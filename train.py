"""
Patch Diffusion × JiT × MMDiT ハイブリッドモデルの訓練スクリプト
タスク: FFHQ 512×512 無条件生成

使い方:
  # 学習開始 (デフォルト: 直接射影, depth=12, hidden=768, ~134M params)
  python train.py

  # パッチ埋め込みの選択:
  #   デフォルト (直接射影): 3072次元 → hidden_size に直接射影。情報損失が少ない
  python train.py
  #   ボトルネック経由: 3072次元 → bottleneck → hidden_size。パラメータ削減
  python train.py --bottleneck_dim 128   # 24倍圧縮 (圧縮しすぎに注意)
  python train.py --bottleneck_dim 512   # 6倍圧縮 (JiTと同等比率)

  # モデルサイズの変更:
  python train.py --depth 16 --hidden_size 1024 --num_heads 16   # ~300M
  python train.py --depth 24 --hidden_size 1024 --num_heads 16   # ~460M

  # バッチサイズやステップ数を変更
  python train.py --batch_size 16 --total_steps 400000

  # AMP無効で学習
  python train.py --no_amp

  # チェックポイントから再開
  python train.py --resume ./runs/patch_dit_ffhq512_YYYYMMDD_HHMMSS/ckpt_0010000.pt

  # TensorBoardでlossグラフ・サンプル画像をリアルタイム監視 (別ターミナルで実行)
  tensorboard --logdir ./runs/<実験ディレクトリ>/tb_logs --port 6006
  # → ブラウザで http://localhost:6006 を開く

  # 出力先は実行ごとに日時付きディレクトリが自動生成される
  # 明示的に指定も可能:
  python train.py --out_dir ./runs/my_experiment
"""

import os
import csv
import math
import argparse
import time
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from model import PatchDiffusionDiT, make_position_grid, sample


class FFHQDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted(Path(root).glob("*.png"))
        if not self.paths:
            self.paths = sorted(Path(root).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class PatchCropper:
    """Patch Diffusion のクロップ戦略。"""

    def __init__(self, img_size=512, patch_size=32, real_p=0.5):
        self.img_size = img_size
        self.patch_size = patch_size
        self.crop_sizes = [img_size, img_size // 2]
        self.crop_probs = [real_p, 1 - real_p]
        self.batch_muls = {
            img_size: 1,
            img_size // 2: 4,
        }

    def sample_crop_size(self):
        return np.random.choice(self.crop_sizes, p=self.crop_probs)

    def crop_batch(self, images, crop_size):
        """
        バッチ内の各画像を独立にランダムクロップし、RoPE用位置を返す。

        Args:
            images: (B, 3, img_size, img_size)
            crop_size: クロップサイズ (ピクセル)
        Returns:
            patches: (B, 3, crop_size, crop_size)
            pos_h: (B, N) パッチグリッド行位置
            pos_w: (B, N) パッチグリッド列位置
        """
        B = images.shape[0]
        p = self.patch_size
        grid_crop = crop_size // p

        if crop_size == self.img_size:
            pos_h, pos_w = make_position_grid(
                grid_crop, grid_crop, device=images.device
            )
            return images, pos_h.expand(B, -1), pos_w.expand(B, -1)

        max_offset = self.img_size - crop_size
        offsets_y = torch.randint(0, max_offset + 1, (B,))
        offsets_x = torch.randint(0, max_offset + 1, (B,))

        patches = []
        all_pos_h = []
        all_pos_w = []

        for i in range(B):
            oy, ox = offsets_y[i].item(), offsets_x[i].item()
            patches.append(images[i, :, oy : oy + crop_size, ox : ox + crop_size])

            ph, pw = make_position_grid(
                grid_crop, grid_crop,
                offset_h=oy // p, offset_w=ox // p,
                device=images.device,
            )
            all_pos_h.append(ph)
            all_pos_w.append(pw)

        patches = torch.stack(patches)
        pos_h = torch.cat(all_pos_h, dim=0)
        pos_w = torch.cat(all_pos_w, dim=0)
        return patches, pos_h, pos_w


def logit_normal_timestep(batch_size, m=0.0, s=1.0, eps=1e-5, device="cpu"):
    """Logit-Normal分布からタイムステップをサンプリング。"""
    u = torch.randn(batch_size, device=device) * s + m
    t = torch.sigmoid(u)
    return t.clamp(eps, 1 - eps)


def compute_v_loss(model, x_0, t, pos_h, pos_w):
    """
    x-prediction + v-loss を計算。

    順方向: z_t = t * x_0 + (1-t) * ε
    モデル: x_pred = model(z_t, t, pos_h, pos_w)
    v_pred = (x_pred - z_t) / (1-t)
    v_target = x_0 - ε
    loss = ||v_target - v_pred||²
    """
    eps = torch.randn_like(x_0)
    t_expand = t.view(-1, 1, 1, 1)

    z_t = t_expand * x_0 + (1 - t_expand) * eps

    x_pred = model(z_t, t, pos_h, pos_w)

    v_target = x_0 - eps
    v_pred = (x_pred - z_t) / (1 - t_expand).clamp(min=1e-5)

    loss = F.mse_loss(v_pred, v_target)
    return loss


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.lerp_(p.data, 1 - decay)


def save_samples(model, step, out_dir, device, n=8):
    """EMAモデルでフル画像をサンプル生成して保存。"""
    imgs = sample(model, batch_size=n, steps=10, device=device)
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
    imgs = (imgs * 255).byte().cpu()

    sample_dir = Path(out_dir) / "samples"
    sample_dir.mkdir(exist_ok=True)

    for i, img in enumerate(imgs):
        img_pil = Image.fromarray(img.permute(1, 2, 0).numpy())
        img_pil.save(sample_dir / f"step{step:07d}_{i}.png")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = FFHQDataset(args.data_dir, transform=transform)
    print(f"Dataset: {len(dataset)} images")

    model = PatchDiffusionDiT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        depth=args.depth,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.1f}M")

    ema_model = deepcopy(model)
    ema_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)
    cropper = PatchCropper(args.img_size, args.patch_size, args.real_p)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    step = 0
    epoch = 0
    log_loss = 0.0
    log_count = 0

    # Resume
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            ema_model.load_state_dict(ckpt["ema_model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            step = ckpt["step"]
            print(f"  Resumed at step {step}")
        else:
            print(f"Warning: {ckpt_path} not found, training from scratch")

    # Loss log (CSV)
    loss_log_path = out_dir / "loss_log.csv"
    if step == 0 or not loss_log_path.exists():
        with open(loss_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "crop_size", "lr", "elapsed_sec"])

    # TensorBoard
    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    start_time = time.time()

    print(f"Training for {args.total_steps} steps (starting from {step})...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Patch sizes: {cropper.crop_sizes} with probs {cropper.crop_probs}")

    model.train()
    while step < args.total_steps:
        epoch += 1
        for images in loader:
            if step >= args.total_steps:
                break

            images = images.to(device, non_blocking=True)

            crop_size = cropper.sample_crop_size()
            batch_mul = cropper.batch_muls[crop_size]

            if batch_mul > 1:
                images = images.repeat(batch_mul, 1, 1, 1)

            patches, pos_h, pos_w = cropper.crop_batch(images, crop_size)

            t = logit_normal_timestep(
                patches.shape[0], m=args.lognorm_m, s=args.lognorm_s, device=device
            )

            # Warmup LR
            if step < args.warmup_steps:
                lr = args.lr * (step + 1) / args.warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.use_amp, dtype=torch.bfloat16):
                loss = compute_v_loss(model, patches, t, pos_h, pos_w)

            scaler.scale(loss / batch_mul).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            update_ema(ema_model, model, decay=args.ema_decay)

            log_loss += loss.item()
            log_count += 1
            step += 1

            if step % args.log_every == 0:
                avg_loss = log_loss / log_count
                elapsed = time.time() - start_time
                cur_lr = optimizer.param_groups[0]["lr"]
                imgs_per_sec = step * args.batch_size / elapsed
                print(
                    f"step={step:>7d}  loss={avg_loss:.4f}  "
                    f"crop={crop_size:>3d}  "
                    f"lr={cur_lr:.2e}  "
                    f"img/s={imgs_per_sec:.1f}"
                )
                with open(loss_log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        step, f"{avg_loss:.6f}", crop_size,
                        f"{cur_lr:.2e}", f"{elapsed:.1f}",
                    ])
                writer.add_scalar("loss/train", avg_loss, step)
                writer.add_scalar("loss/crop_size", crop_size, step)
                writer.add_scalar("lr", cur_lr, step)
                log_loss = 0.0
                log_count = 0

            if step % args.save_every == 0:
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }
                torch.save(ckpt, out_dir / f"ckpt_{step:07d}.pt")
                print(f"Saved checkpoint at step {step}")

            if step % args.sample_every == 0:
                save_samples(ema_model, step, out_dir, device)
                # TensorBoardにもサンプル画像を追加
                sample_imgs = sample(ema_model, batch_size=4, steps=10, device=device)
                sample_imgs = (sample_imgs * 0.5 + 0.5).clamp(0, 1)
                writer.add_images("samples", sample_imgs, step)
                print(f"Saved samples at step {step}")
                model.train()

    # 最終保存
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }, out_dir / "ckpt_final.pt")
    save_samples(ema_model, step, out_dir, device)
    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_dir", type=str, default="./images512x512")
    p.add_argument("--out_dir", type=str,
                   default=f"./runs/patch_dit_ffhq512_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--img_size", type=int, default=512)

    # Model
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--depth", type=int, default=24)
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--bottleneck_dim", type=int, default=None,
                   help="None=直接射影, int=ボトルネック次元 (例: 128, 512)")

    # Patch Diffusion
    p.add_argument("--real_p", type=float, default=0.5)

    # Flow matching
    p.add_argument("--lognorm_m", type=float, default=0.0)
    p.add_argument("--lognorm_s", type=float, default=1.0)

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--total_steps", type=int, default=200000)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_false", dest="use_amp")
    p.add_argument("--num_workers", type=int, default=4)

    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="チェックポイントのパス (例: runs/patch_dit_ffhq512/ckpt_0010000.pt)")

    # Logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--sample_every", type=int, default=5000)

    train(p.parse_args())
