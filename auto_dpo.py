"""
Iterative Auto-DPO パイプライン。
DINO特徴量スコアリングで自動ペア構成 → DPO学習 → 再生成 → 繰り返し。

Usage:
    # FFHQ特徴量が未計算の場合は先に実行:
    # python dino_scorer.py --precompute --data_dir ./images256_ffhq_celebahq --cache dpo_data/ffhq_features.pt

    python auto_dpo.py \
        --base_ckpt runs/patch_dit_ffhq512_20260505_112612/ckpt_0530000.pt \
        --ffhq_cache dpo_data/ffhq_features.pt \
        --n_rounds 5 \
        --n_generate 10000 \
        --n_pairs 5000 \
        --beta 1000 \
        --out_dir runs/auto_dpo
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, desc=""):
    """コマンドを実行して結果を表示。"""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", type=str, required=True, help="ベースモデルのチェックポイント")
    p.add_argument("--ffhq_cache", type=str, default="dpo_data/ffhq_features.pt",
                   help="FFHQ特徴量キャッシュ")
    p.add_argument("--out_dir", type=str, default="runs/auto_dpo")
    p.add_argument("--python", type=str, default="/home/naoto/venv_fp8/bin/python")

    p.add_argument("--n_rounds", type=int, default=5, help="DPOラウンド数")
    p.add_argument("--n_generate", type=int, default=10000, help="各ラウンドの生成枚数")
    p.add_argument("--n_pairs", type=int, default=5000, help="各ラウンドのペア数")
    p.add_argument("--gen_batch_size", type=int, default=500, help="生成バッチサイズ")
    p.add_argument("--top_pct", type=float, default=0.3, help="preferred候補の上位割合")
    p.add_argument("--bottom_pct", type=float, default=0.3, help="rejected候補の下位割合")

    p.add_argument("--beta", type=float, default=1000)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--dpo_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)

    p.add_argument("--compare_seeds", type=str, default="1060,1180,1190,1130,1031,1090,1140,1005")
    p.add_argument("--k", type=int, default=10, help="K-NN のK")

    args = p.parse_args()
    py = args.python
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    ffhq_cache = Path(args.ffhq_cache)
    if not ffhq_cache.exists():
        print(f"FFHQ feature cache not found: {ffhq_cache}")
        print(f"Run: {py} dino_scorer.py --precompute --data_dir ./images256_ffhq_celebahq --cache {ffhq_cache}")
        sys.exit(1)

    current_ckpt = args.base_ckpt
    log = []

    for round_idx in range(1, args.n_rounds + 1):
        print(f"\n{'#'*60}")
        print(f"  ROUND {round_idx}/{args.n_rounds}")
        print(f"  Current model: {current_ckpt}")
        print(f"{'#'*60}")

        round_dir = out_dir / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        gen_dir = round_dir / "generated"
        scores_path = round_dir / "scores.json"
        pairs_path = round_dir / "pairs.json"
        dpo_dir = round_dir / "dpo"

        # Step 1: 生成
        seed_offset = round_idx * 100000
        run_cmd(
            f"{py} generate_for_eval.py "
            f"--ckpt {current_ckpt} "
            f"--n {args.n_generate} "
            f"--out_dir {gen_dir} "
            f"--batch_size {args.gen_batch_size} "
            f"--seed_offset {seed_offset}",
            f"Round {round_idx} - Step 1: Generate {args.n_generate} images"
        )

        # Step 2: DINOスコアリング
        run_cmd(
            f"{py} dino_scorer.py "
            f"--score "
            f"--img_dir {gen_dir} "
            f"--cache {ffhq_cache} "
            f"--out {scores_path} "
            f"--k {args.k}",
            f"Round {round_idx} - Step 2: DINO scoring"
        )

        # Step 3: ペア構成
        run_cmd(
            f"{py} dino_scorer.py "
            f"--build_pairs "
            f"--scores {scores_path} "
            f"--out {pairs_path} "
            f"--n {args.n_pairs} "
            f"--top_pct {args.top_pct} "
            f"--bottom_pct {args.bottom_pct}",
            f"Round {round_idx} - Step 3: Build {args.n_pairs} pairs"
        )

        # Step 4: DPO学習
        run_cmd(
            f"{py} train_dpo.py "
            f"--ckpt {current_ckpt} "
            f"--pairs {pairs_path} "
            f"--img_dir {gen_dir} "
            f"--beta {args.beta} "
            f"--lr {args.lr} "
            f"--epochs {args.epochs} "
            f"--batch_size {args.dpo_batch_size} "
            f"--grad_accum {args.grad_accum} "
            f"--compare_every {args.epochs} "
            f"--save_every {args.epochs} "
            f"--compare_seeds {args.compare_seeds} "
            f"--out_dir {dpo_dir}",
            f"Round {round_idx} - Step 4: DPO training (beta={args.beta}, {args.epochs} epochs)"
        )

        new_ckpt = dpo_dir / f"dpo_epoch_{args.epochs:04d}.pt"
        if not new_ckpt.exists():
            print(f"ERROR: DPO checkpoint not found: {new_ckpt}")
            sys.exit(1)

        # ログ
        with open(scores_path) as f:
            scores = json.load(f)
        import numpy as np
        score_vals = list(scores.values())
        round_log = {
            "round": round_idx,
            "ckpt_in": current_ckpt,
            "ckpt_out": str(new_ckpt),
            "n_generated": args.n_generate,
            "n_pairs": args.n_pairs,
            "score_mean": float(np.mean(score_vals)),
            "score_median": float(np.median(score_vals)),
            "score_std": float(np.std(score_vals)),
            "score_min": float(np.min(score_vals)),
            "score_max": float(np.max(score_vals)),
        }
        log.append(round_log)
        with open(out_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)

        print(f"\n  Round {round_idx} complete.")
        print(f"  Score: mean={round_log['score_mean']:.4f}, "
              f"median={round_log['score_median']:.4f}, "
              f"min={round_log['score_min']:.4f}")
        print(f"  New checkpoint: {new_ckpt}")

        current_ckpt = str(new_ckpt)

    # 最終サマリー
    print(f"\n{'='*60}")
    print(f"  AUTO-DPO COMPLETE: {args.n_rounds} rounds")
    print(f"{'='*60}")
    print(f"\nScore progression:")
    for entry in log:
        print(f"  Round {entry['round']}: mean={entry['score_mean']:.4f}, "
              f"median={entry['score_median']:.4f}")
    print(f"\nFinal model: {current_ckpt}")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
