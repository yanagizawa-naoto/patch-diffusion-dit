"""
デノイズ過程を可視化する動画を生成。
16枚の画像を4x4グリッドで2048x2048、ノイズ→クリーンの過程をアニメーション。

Usage:
    python make_denoise_video.py --ckpt runs/auto_dpo/round_02/dpo/dpo_epoch_0030.pt --out denoise.mp4
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import torch
import numpy as np
from PIL import Image

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


def make_grid(images, nrow=4):
    """画像のリストを nrow x ncol のグリッドに結合。"""
    n = len(images)
    ncol = nrow
    h, w = images[0].shape[1], images[0].shape[2]
    grid = torch.zeros(3, h * nrow, w * ncol)
    for idx, img in enumerate(images):
        r = idx // ncol
        c = idx % ncol
        grid[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return grid


def tensor_to_pil(tensor):
    """[-1,1] tensor → PIL Image"""
    img = tensor.clamp(-1, 1) * 0.5 + 0.5
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img)


@torch.no_grad()
def generate_with_trajectory(model, seeds, steps=50, device="cuda"):
    """Heun法でサンプリングし、各ステップの中間状態を返す。"""
    n = len(seeds)
    img_size = model.img_size
    grid = model.grid_size

    pos_h, pos_w = make_position_grid(grid, grid, device=device)
    pos_h = pos_h.expand(n, -1)
    pos_w = pos_w.expand(n, -1)

    noises = []
    for seed in seeds:
        g = torch.Generator(device=device).manual_seed(seed)
        noises.append(torch.randn(1, 3, img_size, img_size, device=device, generator=g))
    x = torch.cat(noises, dim=0)

    trajectory = [x.clone()]
    dt = 1.0 / steps

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(steps):
            t_cur = i / steps
            t_next = (i + 1) / steps
            t_batch = torch.full((n,), t_cur, device=device)

            x_pred = model(x, t_batch, pos_h, pos_w)
            v = (x_pred - x) / max(1 - t_cur, 0.05)

            x_mid = x + v * dt
            if i < steps - 1:
                t_mid = torch.full((n,), t_next, device=device)
                x_pred_mid = model(x_mid, t_mid, pos_h, pos_w)
                v_mid = (x_pred_mid - x_mid) / max(1 - t_next, 0.05)
                x = x + 0.5 * (v + v_mid) * dt
            else:
                x = x_mid

            trajectory.append(x.float().clone())

    return trajectory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="denoise.mp4")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--hold_frames", type=int, default=30, help="最終フレームの静止フレーム数")
    p.add_argument("--seeds", type=str, default=None, help="カンマ区切りのseed (16個)")
    args = p.parse_args()

    device = "cuda"

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = list(range(3000, 3016))
    assert len(seeds) == 16, f"16 seeds required, got {len(seeds)}"

    print(f"Loading model from {args.ckpt}...")
    model = load_model(args.ckpt, device)

    print(f"Generating {len(seeds)} images with {args.steps} steps...")
    trajectory = generate_with_trajectory(model, seeds, steps=args.steps, device=device)
    print(f"Trajectory: {len(trajectory)} frames")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        print("Saving frames...")

        for frame_idx, state in enumerate(trajectory):
            images = [state[i] for i in range(len(seeds))]
            grid = make_grid(images, nrow=4)
            pil_img = tensor_to_pil(grid)
            pil_img.save(tmp_dir / f"frame_{frame_idx:04d}.png")

        for i in range(args.hold_frames):
            frame_idx = len(trajectory) + i
            images = [trajectory[-1][j] for j in range(len(seeds))]
            grid = make_grid(images, nrow=4)
            pil_img = tensor_to_pil(grid)
            pil_img.save(tmp_dir / f"frame_{frame_idx:04d}.png")

        print(f"Encoding video ({args.fps} fps)...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", str(tmp_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-vf", "scale=2048:2048",
            args.out,
        ]
        subprocess.run(cmd, capture_output=True)

    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
