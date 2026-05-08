"""
セグメンテーションDPO用: 各画像に対して2つのマスクを生成。

Usage:
    python generate_seg_pairs.py --ckpt runs/multitask/ckpt_0050000.pt --n 100 --out_dir dpo_data/seg_pairs
"""
import argparse
import json
import random
import torch
from pathlib import Path
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

from model import PatchDiffusionDiT, make_position_grid
from train_multitask import SegDataset, sample_segmentation


def load_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = PatchDiffusionDiT(
        img_size=512, patch_size=32, hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128,
    )
    sd = ckpt.get("model", ckpt.get("ema_model"))
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def generate_mask_pair(model, image, steps=50, device="cuda"):
    """同じ画像から2つの異なるマスクを生成（ノイズ違い）。"""
    B = 2
    imgs = image.unsqueeze(0).expand(B, -1, -1, -1)
    grid = model.grid_size
    pos_h, pos_w = make_position_grid(grid, grid, device=device)
    pos_h = pos_h.expand(B, -1)
    pos_w = pos_w.expand(B, -1)

    task_id = torch.ones(B, dtype=torch.long, device=device)
    x = torch.randn(B, 3, model.img_size, model.img_size, device=device)
    dt = 1.0 / steps

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(steps):
            t_cur = i / steps
            t_next = (i + 1) / steps
            t_batch = torch.full((B,), t_cur, device=device)
            x_pred = model(x, t_batch, pos_h, pos_w, cond_x=imgs, task_id=task_id)
            v = (x_pred - x) / max(1 - t_cur, 0.05)
            x_mid = x + v * dt
            if i < steps - 1:
                t_mid = torch.full((B,), t_next, device=device)
                x_pred_mid = model(x_mid, t_mid, pos_h, pos_w, cond_x=imgs, task_id=task_id)
                v_mid = (x_pred_mid - x_mid) / max(1 - t_next, 0.05)
                x = x + 0.5 * (v + v_mid) * dt
            else:
                x = x_mid

    return x[0].float().clamp(-1, 1), x[1].float().clamp(-1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--seg_dir", type=str, default="./celebamask_hq_15class")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out_dir", type=str, default="dpo_data/seg_pairs")
    args = p.parse_args()

    device = "cuda"
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "mask_a").mkdir(parents=True, exist_ok=True)
    (out_dir / "mask_b").mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, device)

    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    seg_dir = Path(args.seg_dir)
    all_pairs = []
    for mf in sorted((seg_dir / "masks_vis").glob("*.png")):
        img_path = seg_dir / "images" / f"{mf.stem}.jpg"
        if img_path.exists():
            all_pairs.append((img_path, mf))
    random.seed(42)
    selected = random.sample(all_pairs, min(args.n, len(all_pairs)))

    print(f"Generating mask pairs for {len(selected)} images...")
    manifest = []

    for idx, (img_path, gt_path) in enumerate(selected):
        img = transform(Image.open(img_path).convert("RGB")).to(device)
        gt = transform(Image.open(gt_path).convert("RGB"))
        mask_a, mask_b = generate_mask_pair(model, img, device=device)

        name = f"{idx:04d}"
        save_image(img * 0.5 + 0.5, out_dir / "images" / f"{name}.png")
        save_image(gt * 0.5 + 0.5, out_dir / "gt" / f"{name}.png")
        save_image(mask_a * 0.5 + 0.5, out_dir / "mask_a" / f"{name}.png")
        save_image(mask_b * 0.5 + 0.5, out_dir / "mask_b" / f"{name}.png")

        manifest.append({"id": name, "source": img_path.name})

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(selected)}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. {len(manifest)} pairs saved to {out_dir}")


if __name__ == "__main__":
    main()
