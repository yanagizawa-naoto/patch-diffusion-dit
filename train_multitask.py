"""
マルチタスク継続事前学習:
  タスク0: 顔画像生成 (従来通り)
  タスク1: セグメンテーション (画像条件→マスク生成)

Usage:
    python train_multitask.py \
        --ckpt runs/auto_dpo/round_02/dpo/dpo_epoch_0030.pt \
        --face_dir ./images256_ffhq_celebahq \
        --seg_dir ./celebamask_hq_15class \
        --seg_fraction 0.1 \
        --seg_ratio 0.5 \
        --batch_size 64 \
        --total_steps 50000 \
        --out_dir runs/multitask
"""
import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from model import PatchDiffusionDiT, make_position_grid


class PatchCropper:
    """Patch Diffusion用のランダムクロッパー。"""
    def __init__(self, img_size=512, patch_size=32, real_p=0.5):
        self.img_size = img_size
        self.patch_size = patch_size
        self.crop_sizes = [img_size, img_size // 2]
        self.crop_probs = [real_p, 1 - real_p]
        self.batch_muls = {img_size: 1, img_size // 2: 4}

    def sample_crop_size(self):
        return np.random.choice(self.crop_sizes, p=self.crop_probs)

    def crop_batch(self, images, crop_size):
        B = images.shape[0]
        p = self.patch_size
        grid_crop = crop_size // p

        if crop_size == self.img_size:
            pos_h, pos_w = make_position_grid(grid_crop, grid_crop, device=images.device)
            return images, pos_h.expand(B, -1), pos_w.expand(B, -1), None, None

        max_offset_grid = (self.img_size - crop_size) // p
        offsets_y = torch.randint(0, max_offset_grid + 1, (B,)) * p
        offsets_x = torch.randint(0, max_offset_grid + 1, (B,)) * p

        patches, all_pos_h, all_pos_w = [], [], []
        for i in range(B):
            oy, ox = offsets_y[i].item(), offsets_x[i].item()
            patches.append(images[i, :, oy:oy + crop_size, ox:ox + crop_size])
            ph, pw = make_position_grid(grid_crop, grid_crop,
                                        offset_h=oy // p, offset_w=ox // p,
                                        device=images.device)
            all_pos_h.append(ph)
            all_pos_w.append(pw)

        return torch.stack(patches), torch.cat(all_pos_h), torch.cat(all_pos_w), offsets_y, offsets_x

    def crop_pair(self, images, masks, crop_size):
        """画像とマスクを同じ位置でクロップ。"""
        B = images.shape[0]
        p = self.patch_size
        grid_crop = crop_size // p

        if crop_size == self.img_size:
            pos_h, pos_w = make_position_grid(grid_crop, grid_crop, device=images.device)
            return images, masks, pos_h.expand(B, -1), pos_w.expand(B, -1)

        max_offset_grid = (self.img_size - crop_size) // p
        offsets_y = torch.randint(0, max_offset_grid + 1, (B,)) * p
        offsets_x = torch.randint(0, max_offset_grid + 1, (B,)) * p

        img_patches, mask_patches, all_pos_h, all_pos_w = [], [], [], []
        for i in range(B):
            oy, ox = offsets_y[i].item(), offsets_x[i].item()
            img_patches.append(images[i, :, oy:oy + crop_size, ox:ox + crop_size])
            mask_patches.append(masks[i, :, oy:oy + crop_size, ox:ox + crop_size])
            ph, pw = make_position_grid(grid_crop, grid_crop,
                                        offset_h=oy // p, offset_w=ox // p,
                                        device=images.device)
            all_pos_h.append(ph)
            all_pos_w.append(pw)

        return (torch.stack(img_patches), torch.stack(mask_patches),
                torch.cat(all_pos_h), torch.cat(all_pos_w))


class FaceDataset(Dataset):
    def __init__(self, img_dir, img_size=512):
        self.files = sorted(Path(img_dir).glob("*.*"))
        self.files = [f for f in self.files if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.files[idx]).convert("RGB"))


class SegDataset(Dataset):
    def __init__(self, seg_dir, img_size=512, fraction=0.1):
        seg_dir = Path(seg_dir)
        img_dir = seg_dir / "images"
        mask_dir = seg_dir / "masks_vis"

        mask_files = sorted(mask_dir.glob("*.png"))
        n_use = max(1, int(len(mask_files) * fraction))
        random.seed(42)
        mask_files = random.sample(mask_files, n_use)

        self.pairs = []
        for mf in mask_files:
            stem = mf.stem
            img_path = img_dir / f"{stem}.jpg"
            if not img_path.exists():
                img_path = img_dir / f"{stem}.png"
            if img_path.exists():
                self.pairs.append((img_path, mf))

        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        mask = self.transform(Image.open(mask_path).convert("RGB"))
        return img, mask


def logit_normal_timestep(batch_size, m=-0.8, s=0.8, eps=1e-5, device="cpu"):
    u = torch.randn(batch_size, device=device) * s + m
    t = torch.sigmoid(u)
    return t.clamp(eps, 1 - eps)


def compute_v_loss_multitask(model, x_0, t, pos_h, pos_w, noise_scale=1.0,
                              cond_x=None, task_id=None):
    eps = torch.randn_like(x_0) * noise_scale
    t_expand = t.view(-1, 1, 1, 1)
    z_t = t_expand * x_0 + (1 - t_expand) * eps
    x_pred = model(z_t, t, pos_h, pos_w, cond_x=cond_x, task_id=task_id)
    v_target = x_0 - eps
    v_pred = (x_pred - z_t) / (1 - t_expand).clamp(min=0.05)
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample_segmentation(model, images, steps=50, device="cuda"):
    """画像を条件にマスクを生成。"""
    model.eval()
    B = images.shape[0]
    img_size = model.img_size
    grid = model.grid_size

    pos_h, pos_w = make_position_grid(grid, grid, device=device)
    pos_h = pos_h.expand(B, -1)
    pos_w = pos_w.expand(B, -1)

    task_id = torch.ones(B, dtype=torch.long, device=device)
    x = torch.randn(B, 3, img_size, img_size, device=device)
    dt = 1.0 / steps

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(steps):
            t_cur = i / steps
            t_next = (i + 1) / steps
            t_batch = torch.full((B,), t_cur, device=device)

            x_pred = model(x, t_batch, pos_h, pos_w, cond_x=images, task_id=task_id)
            v = (x_pred - x) / max(1 - t_cur, 0.05)

            x_mid = x + v * dt
            if i < steps - 1:
                t_mid = torch.full((B,), t_next, device=device)
                x_pred_mid = model(x_mid, t_mid, pos_h, pos_w, cond_x=images, task_id=task_id)
                v_mid = (x_pred_mid - x_mid) / max(1 - t_next, 0.05)
                x = x + 0.5 * (v + v_mid) * dt
            else:
                x = x_mid

    return x.float().clamp(-1, 1)


def save_samples(model, seg_dataset, step, out_dir, device, n=4):
    """顔生成サンプル + セグメンテーションサンプルを保存。"""
    from model import sample
    model.eval()

    face_imgs = sample(model, batch_size=n, steps=50, device=device)
    save_image(face_imgs * 0.5 + 0.5, out_dir / "samples" / f"step{step:07d}_face.png",
               nrow=n, padding=2)

    if len(seg_dataset) > 0:
        indices = random.sample(range(len(seg_dataset)), min(n, len(seg_dataset)))
        imgs = torch.stack([seg_dataset[i][0] for i in indices]).to(device)
        gt_masks = torch.stack([seg_dataset[i][1] for i in indices]).to(device)

        pred_masks = sample_segmentation(model, imgs, steps=50, device=device)

        rows = []
        for i in range(len(indices)):
            rows.extend([imgs[i], gt_masks[i], pred_masks[i]])
        grid = torch.stack(rows) * 0.5 + 0.5
        save_image(grid, out_dir / "samples" / f"step{step:07d}_seg.png",
                   nrow=3, padding=2)

    model.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--face_dir", type=str, default="./images256_ffhq_celebahq")
    p.add_argument("--seg_dir", type=str, default="./celebamask_hq_15class")
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--seg_fraction", type=float, default=0.1, help="セグメンテーションデータの使用割合")
    p.add_argument("--seg_ratio", type=float, default=0.3,
                   help="各バッチでセグメンテーションタスクを選ぶ確率")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--total_steps", type=int, default=50000)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--noise_scale", type=float, default=1.0)

    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--real_p", type=float, default=0.5,
                   help="Patch Diffusion: フル画像を使う確率 (0.5=半分はクロップ)")
    p.add_argument("--compile", action="store_true", help="torch.compile")

    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = f"runs/multitask_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    device = "cuda"

    print("Loading model...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model_args = ckpt["args"]
    model = PatchDiffusionDiT(
        img_size=model_args.get("img_size", 512),
        patch_size=model_args.get("patch_size", 32),
        hidden_size=model_args.get("hidden_size", 768),
        depth=model_args.get("depth", 12),
        num_heads=model_args.get("num_heads", 12),
        bottleneck_dim=model_args.get("bottleneck_dim", 128),
        num_experts=model_args.get("num_experts", 0),
        top_k=model_args.get("top_k", 2),
    )
    sd = ckpt.get("ema_model", ckpt["model"])
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device)

    if args.compile:
        model._backbone = torch.compile(model._backbone, dynamic=True)
        print("  torch.compile enabled on _backbone (dynamic=True)")

    model.train()

    grid = model.grid_size
    patch_size = model.patch_size
    cropper = PatchCropper(img_size=args.img_size, patch_size=patch_size, real_p=args.real_p)

    print("Loading datasets...")
    face_dataset = FaceDataset(args.face_dir, img_size=args.img_size)
    seg_dataset = SegDataset(args.seg_dir, img_size=args.img_size, fraction=args.seg_fraction)
    print(f"  Face: {len(face_dataset)} images")
    print(f"  Seg:  {len(seg_dataset)} pairs (fraction={args.seg_fraction})")

    face_loader = DataLoader(face_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    seg_loader = DataLoader(seg_dataset, batch_size=min(args.batch_size, len(seg_dataset)),
                            shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    face_iter = iter(face_loader)
    seg_iter = iter(seg_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "task", "loss", "lr"])

    print(f"\nTraining for {args.total_steps} steps...")
    print(f"  seg_ratio={args.seg_ratio} (probability of segmentation task per step)")

    save_samples(model, seg_dataset, 0, out_dir, device)

    for step in range(1, args.total_steps + 1):
        if step <= args.warmup_steps:
            lr = args.lr * step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / (args.total_steps - args.warmup_steps)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        is_seg = random.random() < args.seg_ratio

        if is_seg:
            try:
                imgs, masks = next(seg_iter)
            except StopIteration:
                seg_iter = iter(seg_loader)
                imgs, masks = next(seg_iter)

            imgs = imgs.to(device)
            masks = masks.to(device)
            imgs_crop, masks_crop, ph, pw = cropper.crop_pair(imgs, masks, args.img_size)
            B = masks_crop.shape[0]
            t = logit_normal_timestep(B, device=device)
            tid = torch.ones(B, dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = compute_v_loss_multitask(
                    model, masks_crop, t, ph, pw,
                    noise_scale=args.noise_scale,
                    cond_x=imgs_crop, task_id=tid,
                )
            task_name = "seg"
        else:
            try:
                face_imgs = next(face_iter)
            except StopIteration:
                face_iter = iter(face_loader)
                face_imgs = next(face_iter)

            face_imgs = face_imgs.to(device)
            crop_size = cropper.sample_crop_size()
            face_patches, ph, pw, _, _ = cropper.crop_batch(face_imgs, crop_size)
            B = face_patches.shape[0]
            t = logit_normal_timestep(B, device=device)
            tid = torch.zeros(B, dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = compute_v_loss_multitask(
                    model, face_patches, t, ph, pw,
                    noise_scale=args.noise_scale,
                    task_id=tid,
                )
            task_name = "face"

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_every == 0:
            print(f"Step {step:6d}/{args.total_steps} | task={task_name:4s} | "
                  f"loss={loss.item():.5f} | lr={lr:.2e}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, task_name, f"{loss.item():.6f}", f"{lr:.2e}"])

        if step % args.sample_every == 0:
            save_samples(model, seg_dataset, step, out_dir, device)

        if step % args.save_every == 0 or step == args.total_steps:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "args": vars(args),
            }, out_dir / f"ckpt_{step:07d}.pt")
            print(f"  Saved checkpoint: step {step}")

    print("Training complete.")


if __name__ == "__main__":
    main()
