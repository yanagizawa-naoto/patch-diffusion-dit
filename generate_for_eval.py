"""
DPO評価用画像を生成するスクリプト。
固定seedで生成し、後の比較にも使えるようにする。

Usage:
    python generate_for_eval.py --ckpt runs/.../ckpt_0400000.pt --n 500 --out_dir dpo_data/generated
"""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.utils import save_image

from model import PatchDiffusionDiT, make_position_grid


def load_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    args = ckpt["args"]
    model = PatchDiffusionDiT(
        img_size=args.get("img_size", 512),
        patch_size=args.get("patch_size", 32),
        hidden_size=args.get("hidden_size", 768),
        depth=args.get("depth", 12),
        num_heads=args.get("num_heads", 12),
        bottleneck_dim=args.get("bottleneck_dim", 128),
        num_experts=args.get("num_experts", 0),
        top_k=args.get("top_k", 2),
    )
    sd = ckpt.get("ema_model", ckpt["model"])
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def generate_batch(model, seeds, steps=50, device="cuda"):
    batch_size = len(seeds)
    img_size = model.img_size
    grid = model.grid_size

    pos_h, pos_w = make_position_grid(grid, grid, device=device)
    pos_h = pos_h.expand(batch_size, -1)
    pos_w = pos_w.expand(batch_size, -1)

    noises = []
    for seed in seeds:
        g = torch.Generator(device=device).manual_seed(seed)
        noises.append(torch.randn(1, 3, img_size, img_size, device=device, generator=g))
    x = torch.cat(noises, dim=0)

    dt = 1.0 / steps
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(steps):
            t_cur = i / steps
            t_next = (i + 1) / steps
            t_batch = torch.full((batch_size,), t_cur, device=device)

            x_pred = model(x, t_batch, pos_h, pos_w)
            v = (x_pred - x) / max(1 - t_cur, 0.05)

            x_mid = x + v * dt
            if i < steps - 1:
                t_mid = torch.full((batch_size,), t_next, device=device)
                x_pred_mid = model(x_mid, t_mid, pos_h, pos_w)
                v_mid = (x_pred_mid - x_mid) / max(1 - t_next, 0.05)
                x = x + 0.5 * (v + v_mid) * dt
            else:
                x = x_mid

    return x.float().clamp(-1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n", type=int, default=500, help="生成枚数")
    p.add_argument("--out_dir", type=str, default="dpo_data/generated")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed_offset", type=int, default=0)
    args = p.parse_args()

    device = "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.ckpt}...")
    model = load_model(args.ckpt, device)

    print(f"Generating {args.n} images...")
    generated = 0
    while generated < args.n:
        batch_n = min(args.batch_size, args.n - generated)
        seeds = list(range(args.seed_offset + generated, args.seed_offset + generated + batch_n))
        imgs = generate_batch(model, seeds, steps=args.steps, device=device)

        for i, (img, seed) in enumerate(zip(imgs, seeds)):
            path = out_dir / f"seed{seed:06d}.png"
            save_image(img * 0.5 + 0.5, path)

        generated += batch_n
        print(f"  {generated}/{args.n}")

    print(f"Done. Images saved to {out_dir}")


if __name__ == "__main__":
    main()
