"""
Flow Matching DPO 学習スクリプト。

Diffusion-DPO (Wallace et al., 2023) を Flow Matching + v-prediction に適用。

Loss:
    L = -E[log σ(-β(
        ||v_w - v_θ(z_t^w, t)||² - ||v_w - v_ref(z_t^w, t)||²
      - (||v_l - v_θ(z_t^l, t)||² - ||v_l - v_ref(z_t^l, t)||²)
    ))]

Usage:
    # 基本
    python train_dpo.py --ckpt runs/.../ckpt_0400000.pt \\
                        --pairs dpo_data/pairs.json \\
                        --img_dir dpo_data/generated \\
                        --beta 5000

    # 累積実験: 各ペア数でDPOして比較画像を出力
    python train_dpo.py --ckpt runs/.../ckpt_0400000.pt \\
                        --pairs dpo_data/cumulative/pairs_0010.json \\
                        --img_dir dpo_data/generated \\
                        --beta 5000
"""
import argparse
import json
import copy
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from model import PatchDiffusionDiT, make_position_grid


class PreferencePairDataset(Dataset):
    """preferred/rejected画像ペアのデータセット。条件付き(seg)にも対応。"""

    def __init__(self, pairs_json, img_dir, img_size=512):
        with open(pairs_json) as f:
            self.pairs = json.load(f)
        self.img_dir = Path(img_dir)
        self.has_cond = "image" in self.pairs[0]
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img_w = Image.open(self.img_dir / pair["preferred"]).convert("RGB")
        img_l = Image.open(self.img_dir / pair["rejected"]).convert("RGB")
        if self.has_cond:
            cond = Image.open(self.img_dir / pair["image"]).convert("RGB")
            return self.transform(img_w), self.transform(img_l), self.transform(cond)
        return self.transform(img_w), self.transform(img_l)


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
    model.load_state_dict(sd)
    model.to(device)
    return model


def logit_normal_timestep(batch_size, m=-0.8, s=0.8, eps=1e-5, device="cpu"):
    u = torch.randn(batch_size, device=device) * s + m
    t = torch.sigmoid(u)
    return t.clamp(eps, 1 - eps)


def compute_dpo_loss(model, ref_model, x_w, x_l, pos_h, pos_w, beta=5000, noise_scale=1.0,
                     cond_x=None, task_id=None):
    """
    Flow Matching DPO loss. 条件付き(seg)にも対応。

    Returns:
        loss: scalar
        reward_margin: 暗黙的rewardの差 (モニタリング用, preferred側が高いほど良い)
        model_w_err: 学習モデルのpreferred MSE
        model_l_err: 学習モデルのrejected MSE
    """
    B = x_w.shape[0]
    device = x_w.device

    t = logit_normal_timestep(B, device=device)
    t_expand = t.view(-1, 1, 1, 1)

    eps_w = torch.randn_like(x_w) * noise_scale
    eps_l = torch.randn_like(x_l) * noise_scale

    z_t_w = t_expand * x_w + (1 - t_expand) * eps_w
    z_t_l = t_expand * x_l + (1 - t_expand) * eps_l

    v_target_w = x_w - eps_w
    v_target_l = x_l - eps_l

    x_pred_w = model(z_t_w, t, pos_h, pos_w, cond_x=cond_x, task_id=task_id)
    v_pred_w = (x_pred_w - z_t_w) / (1 - t_expand).clamp(min=0.05)

    x_pred_l = model(z_t_l, t, pos_h, pos_w, cond_x=cond_x, task_id=task_id)
    v_pred_l = (x_pred_l - z_t_l) / (1 - t_expand).clamp(min=0.05)

    with torch.no_grad():
        x_pred_w_ref = ref_model(z_t_w, t, pos_h, pos_w, cond_x=cond_x, task_id=task_id)
        v_pred_w_ref = (x_pred_w_ref - z_t_w) / (1 - t_expand).clamp(min=0.05)

        x_pred_l_ref = ref_model(z_t_l, t, pos_h, pos_w, cond_x=cond_x, task_id=task_id)
        v_pred_l_ref = (x_pred_l_ref - z_t_l) / (1 - t_expand).clamp(min=0.05)

    model_w_err = (v_pred_w - v_target_w).pow(2).mean(dim=[1, 2, 3])
    ref_w_err = (v_pred_w_ref - v_target_w).pow(2).mean(dim=[1, 2, 3])
    model_l_err = (v_pred_l - v_target_l).pow(2).mean(dim=[1, 2, 3])
    ref_l_err = (v_pred_l_ref - v_target_l).pow(2).mean(dim=[1, 2, 3])

    inside_sigmoid = -beta * (
        (model_w_err - ref_w_err) - (model_l_err - ref_l_err)
    )

    loss = -F.logsigmoid(inside_sigmoid).mean()

    with torch.no_grad():
        reward_w = (ref_w_err - model_w_err).mean()
        reward_l = (ref_l_err - model_l_err).mean()
        reward_margin = reward_w - reward_l

    return loss, reward_margin, model_w_err.mean(), model_l_err.mean()


@torch.no_grad()
def generate_comparison(model_base, model_dpo, seeds, steps=50, device="cuda", out_path=None):
    """固定seedでbase vs DPOの比較画像を生成。"""
    img_size = model_base.img_size
    grid = model_base.grid_size
    n = len(seeds)

    pos_h, pos_w = make_position_grid(grid, grid, device=device)
    pos_h = pos_h.expand(n, -1)
    pos_w = pos_w.expand(n, -1)

    noises = []
    for seed in seeds:
        g = torch.Generator(device=device).manual_seed(seed)
        noises.append(torch.randn(1, 3, img_size, img_size, device=device, generator=g))
    x_init = torch.cat(noises, dim=0)

    results = []
    for model_cur in [model_base, model_dpo]:
        model_cur.eval()
        x = x_init.clone()
        dt = 1.0 / steps

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(steps):
                t_cur = i / steps
                t_next = (i + 1) / steps
                t_batch = torch.full((n,), t_cur, device=device)

                x_pred = model_cur(x, t_batch, pos_h, pos_w)
                v = (x_pred - x) / max(1 - t_cur, 0.05)
                x_mid = x + v * dt

                if i < steps - 1:
                    t_mid = torch.full((n,), t_next, device=device)
                    x_pred_mid = model_cur(x_mid, t_mid, pos_h, pos_w)
                    v_mid = (x_pred_mid - x_mid) / max(1 - t_next, 0.05)
                    x = x + 0.5 * (v + v_mid) * dt
                else:
                    x = x_mid

        results.append(x.float().clamp(-1, 1))

    base_imgs, dpo_imgs = results
    comparison = torch.cat([base_imgs, dpo_imgs], dim=0)
    comparison = comparison * 0.5 + 0.5

    if out_path:
        save_image(comparison, out_path, nrow=n, padding=2)
        print(f"Comparison saved: {out_path} (top=base, bottom=DPO)")

    return base_imgs, dpo_imgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="ベースモデルのチェックポイント")
    p.add_argument("--pairs", type=str, required=True, help="ペアデータJSON")
    p.add_argument("--img_dir", type=str, required=True, help="生成画像ディレクトリ")
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--beta", type=float, default=5000, help="KLペナルティ係数")
    p.add_argument("--lr", type=float, default=1e-6, help="学習率")
    p.add_argument("--epochs", type=int, default=50, help="エポック数")
    p.add_argument("--batch_size", type=int, default=2, help="バッチサイズ (ペア数)")
    p.add_argument("--grad_accum", type=int, default=4, help="勾配蓄積ステップ")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--noise_scale", type=float, default=1.0)

    p.add_argument("--compare_seeds", type=str, default="0,1,2,3,4,5,6,7",
                   help="比較画像用の固定seed (カンマ区切り)")
    p.add_argument("--compare_every", type=int, default=10, help="何エポックごとに比較画像を出すか")
    p.add_argument("--save_every", type=int, default=10)
    args = p.parse_args()

    device = "cuda"
    compare_seeds = [int(s) for s in args.compare_seeds.split(",")]

    if args.out_dir is None:
        pairs_name = Path(args.pairs).stem
        args.out_dir = f"runs/dpo_{pairs_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"=== Flow Matching DPO ===")
    print(f"Beta: {args.beta}")
    print(f"LR: {args.lr}")
    print(f"Pairs: {args.pairs}")

    print("Loading models...")
    model = load_model(args.ckpt, device)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    model.train()

    dataset = PreferencePairDataset(args.pairs, args.img_dir, img_size=model.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)
    print(f"Pairs loaded: {len(dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = args.epochs * math.ceil(len(dataset) / args.batch_size)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

    grid = model.grid_size
    pos_h, pos_w = make_position_grid(grid, grid, device=device)

    print("\nGenerating base comparison...")
    generate_comparison(
        ref_model, ref_model, compare_seeds, device=device,
        out_path=out_dir / "samples" / "epoch_0000_base.png"
    )

    log_path = out_dir / "dpo_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,step,loss,reward_margin,model_w_err,model_l_err,lr\n")

    global_step = 0
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_reward = 0
        n_batches = 0

        optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(dataloader):
            if len(batch_data) == 3:
                x_w, x_l, cond = batch_data
                cond = cond.to(device)
                tid = torch.ones(x_w.shape[0], dtype=torch.long, device=device)
            else:
                x_w, x_l = batch_data
                cond = None
                tid = None

            x_w = x_w.to(device)
            x_l = x_l.to(device)
            B = x_w.shape[0]

            pos_h_b = pos_h.expand(B, -1)
            pos_w_b = pos_w.expand(B, -1)

            if warmup_steps > 0 and global_step < warmup_steps:
                lr = args.lr * (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, reward_margin, w_err, l_err = compute_dpo_loss(
                    model, ref_model, x_w, x_l, pos_h_b, pos_w_b,
                    beta=args.beta, noise_scale=args.noise_scale,
                    cond_x=cond, task_id=tid,
                )

            scaler.scale(loss / args.grad_accum).backward()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_reward += reward_margin.item()
            n_batches += 1
            global_step += 1

            with open(log_path, "a") as f:
                f.write(f"{epoch},{global_step},{loss.item():.6f},"
                        f"{reward_margin.item():.6f},{w_err.item():.6f},"
                        f"{l_err.item():.6f},{lr:.2e}\n")

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_reward = epoch_reward / max(n_batches, 1)
        print(f"Epoch {epoch:4d}/{args.epochs} | loss={avg_loss:.4f} | "
              f"reward_margin={avg_reward:.6f} | lr={lr:.2e}")

        if epoch % args.compare_every == 0 or epoch == args.epochs:
            model.eval()
            generate_comparison(
                ref_model, model, compare_seeds, device=device,
                out_path=out_dir / "samples" / f"epoch_{epoch:04d}.png"
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = out_dir / f"dpo_epoch_{epoch:04d}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    print(f"\nDPO training complete. Results in {out_dir}")


if __name__ == "__main__":
    main()
