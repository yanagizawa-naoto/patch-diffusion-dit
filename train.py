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

  # === Latentモード (FLUX.1 VAEで事前エンコード済みlatentを使用) ===
  # 事前エンコード:
  python encode_latents.py --img_dir ./images512x512 --out_dir ./latents_flux1_256 --img_size 256
  # latentモードで学習 (img_size/patch_sizeは自動設定):
  python train.py --latent_dir ./latents_flux1_256 --patch_size 2 --batch_size 64

  # === 高速化オプション ===
  # compile単体 (2.0x):
  python train.py --compile
  # max-autotune + 8bit Adam (2.3x、初回コンパイルに数分):
  python train.py --compile --max_autotune --optim_8bit
  # FP8 + compile (2.5x、torch nightly + torchao必要):
  python train.py --fp8 --compile
  # 全部盛り:
  python train.py --fp8 --compile --max_autotune --optim_8bit

  # 出力先は実行ごとに日時付きディレクトリが自動生成される
  # 明示的に指定も可能:
  python train.py --out_dir ./runs/my_experiment

  # === データセット (HuggingFace Hub) ===
  # FFHQ 512×512 JPG画像:
  #   huggingface-cli download --repo-type dataset Naoto-ipu/ffhq-512-jpg --local-dir .
  #   tar xf images512x512_jpg.tar
  # FLUX.1 VAE エンコード済みlatent (256×256, shape 16×32×32):
  #   huggingface-cli download --repo-type dataset Naoto-ipu/ffhq-flux1-latents-256 --local-dir .
  #   tar xf latents_flux1_256.tar
  # 高速ダウンロード (hf_transfer):
  #   pip install hf_transfer
  #   HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ...
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


class LatentDataset(Dataset):
    """事前エンコード済みlatentファイル(.pt)を読むデータセット。"""
    def __init__(self, root, flip=True):
        self.paths = sorted(Path(root).glob("*.pt"))
        self.flip = flip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        latent = torch.load(self.paths[idx], weights_only=True)
        if self.flip and torch.rand(1).item() < 0.5:
            latent = latent.flip(-1)
        return latent


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

        max_offset_grid = (self.img_size - crop_size) // p
        offsets_y = torch.randint(0, max_offset_grid + 1, (B,)) * p
        offsets_x = torch.randint(0, max_offset_grid + 1, (B,)) * p

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


def compute_v_loss(model, x_0, t, pos_h, pos_w, noise_scale=1.0):
    """
    x-prediction + v-loss を計算。

    順方向: z_t = t * x_0 + (1-t) * ε * noise_scale
    モデル: x_pred = model(z_t, t, pos_h, pos_w)
    v_pred = (x_pred - z_t) / (1-t)
    v_target = x_0 - ε * noise_scale
    loss = ||v_target - v_pred||²
    """
    eps = torch.randn_like(x_0) * noise_scale
    t_expand = t.view(-1, 1, 1, 1)

    z_t = t_expand * x_0 + (1 - t_expand) * eps

    x_pred = model(z_t, t, pos_h, pos_w)

    v_target = x_0 - eps
    v_pred = (x_pred - z_t) / (1 - t_expand).clamp(min=0.05)

    loss = F.mse_loss(v_pred, v_target)
    return loss


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.lerp_(p.data, 1 - decay)


def save_samples(model, step, out_dir, device, n=8, vae=None, vae_scaling_factor=None):
    """EMAモデルでサンプル生成して保存。VAE指定時はlatent→画像にデコード。"""
    imgs = sample(model, batch_size=n, steps=50, device=device,
                  vae=vae, vae_scaling_factor=vae_scaling_factor)
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

    # データセット: latentモード or ピクセルモード
    vae = None
    vae_scaling_factor = None
    if args.latent_dir:
        dataset = LatentDataset(args.latent_dir, flip=True)
        # latentの空間サイズを自動検出
        sample_latent = torch.load(dataset.paths[0], weights_only=True)
        latent_channels, latent_h, latent_w = sample_latent.shape
        args.img_size = latent_h
        args.in_channels = latent_channels
        args.patch_size = args.patch_size if args.patch_size <= latent_h else 2
        print(f"Dataset: {len(dataset)} latents, shape={sample_latent.shape}")

        # サンプル生成用にVAEデコーダを読み込み
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            args.vae_id, subfolder="vae", torch_dtype=torch.float16
        ).to(device).eval()
        vae.requires_grad_(False)
        vae_scaling_factor = vae.config.scaling_factor
        print(f"VAE decoder loaded: {args.vae_id} (scaling={vae_scaling_factor})")
    else:
        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = FFHQDataset(args.data_dir, transform=transform)
        args.in_channels = 3
        print(f"Dataset: {len(dataset)} images")

    model = PatchDiffusionDiT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        depth=args.depth,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        bottleneck_dim=args.bottleneck_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.1f}M")

    # EMAはFP8/compile前に作成 (推論用なのでBF16のまま)
    ema_model = deepcopy(model)
    ema_model.requires_grad_(False)

    # Liger-Kernel RMSNorm
    if args.liger:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        from model import RMSNorm as OrigRMSNorm

        class LigerRMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = eps

            @torch._dynamo.disable
            def forward(self, x):
                return LigerRMSNormFunction.apply(x, self.weight, self.eps)

        for name, module in list(model.named_modules()):
            if isinstance(module, OrigRMSNorm):
                parts = name.split('.')
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                ln = LigerRMSNorm(module.weight.shape[0], module.eps)
                ln.weight = module.weight
                setattr(parent, parts[-1], ln)
        print("  Liger RMSNorm enabled")

    # FP8 (学習モデルのみ、0-init層は除外)
    if args.fp8:
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        zero_init_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and torch.all(module.weight == 0):
                zero_init_layers[name] = deepcopy(module)
        convert_to_float8_training(model, config=Float8LinearConfig())
        for name, orig in zero_init_layers.items():
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], orig.to(device))
        print("  FP8 Linear enabled (0-init layers excluded, requires torch nightly + torchao)")

    # torch.compile
    if args.compile:
        if args.max_autotune:
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.conv_1x1_as_mm = True
            model = torch.compile(model, mode="max-autotune", dynamic=True)
            print("  torch.compile enabled (max-autotune, dynamic=True)")
        else:
            model = torch.compile(model, dynamic=True)
            print("  torch.compile enabled (dynamic=True)")

    # lr scaling
    if args.lr_scaling:
        effective_lr = args.lr * args.batch_size / 256
        print(f"  Effective lr: {args.lr} * {args.batch_size}/256 = {effective_lr:.2e} (JiT scaling)")
    else:
        effective_lr = args.lr
        print(f"  Effective lr: {effective_lr:.2e} (scaling off)")

    if args.optim_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(
            model.parameters(), lr=effective_lr, betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        print("  8-bit Adam enabled (bitsandbytes)")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=effective_lr, betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

    # BF16ではGradScaler不要。8bit Adamとも非互換
    use_scaler = args.use_amp and not args.optim_8bit
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
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
    log_step_time = 0.0
    log_count = 0

    # Resume
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            # compile後のモデルは _orig_mod. prefix がつくので変換
            def load_compat(target, state_dict):
                try:
                    target.load_state_dict(state_dict)
                except RuntimeError:
                    # prefix追加/除去を試す
                    new_sd = {}
                    for k, v in state_dict.items():
                        new_sd["_orig_mod." + k] = v
                    try:
                        target.load_state_dict(new_sd)
                    except RuntimeError:
                        new_sd2 = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                        target.load_state_dict(new_sd2)
            load_compat(model, ckpt["model"])
            ema_model.load_state_dict(ckpt["ema_model"])
            if not args.optim_8bit:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                except (ValueError, KeyError, RuntimeError):
                    print("  Warning: optimizer state incompatible, starting fresh optimizer")
            else:
                print("  Skipping optimizer state (8bit Adam is incompatible with saved AdamW state)")
            step = ckpt["step"]
            # resume時にlrを上書き
            for pg in optimizer.param_groups:
                pg["lr"] = effective_lr
            print(f"  Resumed at step {step}, lr={effective_lr:.2e}")
        else:
            print(f"Warning: {ckpt_path} not found, training from scratch")

    # Loss log (CSV)
    loss_log_path = out_dir / "loss_log.csv"
    if step == 0 or not loss_log_path.exists():
        with open(loss_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "crop_size", "lr", "elapsed_sec", "step_time_ms"])

    # TensorBoard
    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    # 設定をファイルに保存 + コンソール出力
    config_path = out_dir / "config.json"
    if step == 0:
        import json
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

    start_time = time.time()

    print(f"Training for {args.total_steps} steps (starting from {step})...")
    print(f"  Out dir: {out_dir}")
    print(f"  Model: depth={args.depth}, hidden={args.hidden_size}, heads={args.num_heads}, "
          f"patch={args.patch_size}, bottleneck={args.bottleneck_dim}")
    print(f"  Params: {n_params:.1f}M")
    print(f"  Batch size: {args.batch_size}, base_lr={args.lr}, effective_lr={effective_lr:.2e}")
    print(f"  Patch Diffusion: sizes={cropper.crop_sizes}, probs={cropper.crop_probs}")
    print(f"  Flow matching: lognorm(m={args.lognorm_m}, s={args.lognorm_s})")
    print(f"  AMP: {args.use_amp}, EMA decay: {args.ema_decay}")

    model.train()
    while step < args.total_steps:
        epoch += 1
        for images in loader:
            if step >= args.total_steps:
                break

            images = images.to(device, non_blocking=True)
            step_start = time.time()

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
                lr = effective_lr * (step + 1) / args.warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.use_amp, dtype=torch.bfloat16):
                loss = compute_v_loss(model, patches, t, pos_h, pos_w,
                                     noise_scale=args.noise_scale)

            scaler.scale(loss / batch_mul).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            update_ema(ema_model, model, decay=args.ema_decay)

            step_time = time.time() - step_start
            log_loss += loss.item()
            log_step_time += step_time
            log_count += 1
            step += 1

            if step % args.log_every == 0:
                avg_loss = log_loss / log_count
                avg_step_ms = log_step_time / log_count * 1000
                cur_lr = optimizer.param_groups[0]["lr"]
                cur_ips = args.batch_size / (log_step_time / log_count)
                print(
                    f"step={step:>7d}  loss={avg_loss:.4f}  "
                    f"crop={crop_size:>3d}  "
                    f"lr={cur_lr:.2e}  "
                    f"step_time={avg_step_ms:.0f}ms  "
                    f"img/s={cur_ips:.0f}"
                )
                elapsed = time.time() - start_time
                with open(loss_log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        step, f"{avg_loss:.6f}", crop_size,
                        f"{cur_lr:.2e}", f"{elapsed:.1f}",
                        f"{avg_step_ms:.1f}",
                    ])
                writer.add_scalar("loss/train", avg_loss, step)
                writer.add_scalar("loss/crop_size", crop_size, step)
                writer.add_scalar("lr", cur_lr, step)
                writer.add_scalar("perf/step_time_ms", avg_step_ms, step)
                writer.add_scalar("perf/img_per_sec", cur_ips, step)
                log_loss = 0.0
                log_step_time = 0.0
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
                save_samples(ema_model, step, out_dir, device,
                             vae=vae, vae_scaling_factor=vae_scaling_factor)
                # TensorBoardにもサンプル画像を追加
                sample_imgs = sample(ema_model, batch_size=4, steps=50, device=device,
                                     vae=vae, vae_scaling_factor=vae_scaling_factor)
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
    save_samples(ema_model, step, out_dir, device,
                 vae=vae, vae_scaling_factor=vae_scaling_factor)
    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_dir", type=str, default="./images512x512")
    p.add_argument("--latent_dir", type=str, default=None,
                   help="事前エンコード済みlatentディレクトリ (指定時はlatentモードで学習)")
    p.add_argument("--vae_id", type=str, default="black-forest-labs/FLUX.1-dev",
                   help="サンプル生成用VAEのHuggingFace ID")
    p.add_argument("--out_dir", type=str,
                   default=f"./runs/patch_dit_ffhq512_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--img_size", type=int, default=512)

    # Model
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--bottleneck_dim", type=int, default=None,
                   help="None=直接射影, int=ボトルネック次元 (例: 128, 512)")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="選択的ドロップアウト率。中間半分のブロックに適用 (JiT-H: 0.2)")
    p.add_argument("--noise_scale", type=float, default=1.0,
                   help="ノイズスケーリング。JiT方式: img_size/256 (256:1.0, 512:2.0)")

    # Patch Diffusion
    p.add_argument("--real_p", type=float, default=0.5)

    # Flow matching
    p.add_argument("--lognorm_m", type=float, default=-0.8,
                   help="logit-normal位置パラメータ (JiT: -0.8)")
    p.add_argument("--lognorm_s", type=float, default=0.8,
                   help="logit-normalスケールパラメータ (JiT: 0.8)")

    # Training (デフォルトはJiT準拠)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="学習率。--lr_scalingありの場合 base lr (実効lr = lr * batch/256)")
    p.add_argument("--lr_scaling", action="store_true", default=False,
                   help="JiT方式のlr scaling (lr * batch/256)。大バッチ時に使用")
    p.add_argument("--no_lr_scaling", action="store_false", dest="lr_scaling")
    p.add_argument("--total_steps", type=int, default=200000)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="0=クリップなし (JiTデフォルト)")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="JiTデフォルト: 0.0")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_false", dest="use_amp")
    p.add_argument("--liger", action="store_true", default=False,
                   help="Liger-Kernel RMSNorm (liger-kernel必要)")
    p.add_argument("--fp8", action="store_true", default=False,
                   help="FP8 Linear (torchao)。0-init層は自動除外。torch nightly + torchao必要")
    p.add_argument("--compile", action="store_true", default=False,
                   help="torch.compile (dynamic=True)")
    p.add_argument("--max_autotune", action="store_true", default=False,
                   help="torch.compile mode=max-autotune。初回コンパイルに数分かかる")
    p.add_argument("--optim_8bit", action="store_true", default=False,
                   help="8-bit Adam (bitsandbytes)。optimizer転送量を4分の1に削減")
    p.add_argument("--num_workers", type=int, default=4)

    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="チェックポイントのパス (例: runs/patch_dit_ffhq512/ckpt_0010000.pt)")

    # Logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--sample_every", type=int, default=5000)

    train(p.parse_args())
