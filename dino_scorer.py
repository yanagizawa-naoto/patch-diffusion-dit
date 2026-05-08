"""
DINO v2 ベースの画像品質スコアラー。
FFHQ特徴量分布との距離で生成画像をスコアリングし、DPO用ペアを自動構成。

Usage:
    # Step 1: FFHQ特徴量を事前計算（初回のみ、約10分）
    python dino_scorer.py --precompute --data_dir ./images256_ffhq_celebahq --cache dpo_data/ffhq_features.pt

    # Step 2: 生成画像をスコアリング
    python dino_scorer.py --score --img_dir dpo_data/generated --cache dpo_data/ffhq_features.pt --out dpo_data/auto_scores.json

    # Step 3: スコアからペア構成
    python dino_scorer.py --build_pairs --scores dpo_data/auto_scores.json --out dpo_data/auto_pairs.json --n 10000
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def load_dino(device="cuda"):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model.eval().to(device)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_features(model, img_dir, batch_size=128, device="cuda"):
    """ディレクトリ内の全画像からDINO特徴量を抽出。"""
    transform = get_transform()
    img_dir = Path(img_dir)
    files = sorted(
        [f for f in img_dir.iterdir()
         if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    )

    all_features = []
    all_names = []

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        imgs = []
        for f in batch_files:
            img = Image.open(f).convert("RGB")
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(device)

        features = model(batch)
        features = F.normalize(features, dim=-1)
        all_features.append(features.cpu())
        all_names.extend([f.name for f in batch_files])

        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(batch_files)}/{len(files)}")

    return torch.cat(all_features, dim=0), all_names


def precompute_ffhq(data_dir, cache_path, device="cuda", batch_size=256):
    """FFHQ+CelebAHQ全画像のDINO特徴量を計算してキャッシュ。"""
    print("Loading DINOv2...")
    model = load_dino(device)

    print(f"Extracting features from {data_dir}...")
    features, names = extract_features(model, data_dir, batch_size=batch_size, device=device)

    mean = features.mean(dim=0)
    mean = F.normalize(mean, dim=0)

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features": features,
        "mean": mean,
        "names": names,
    }, cache_path)
    print(f"Saved {len(names)} features to {cache_path} ({features.shape})")


def score_images(img_dir, cache_path, out_path, k=10, device="cuda", batch_size=128):
    """
    生成画像をスコアリング。
    スコア = FFHQ特徴量のK近傍との平均コサイン類似度。
    高い = FFHQに近い = 良い画像。
    """
    print("Loading DINOv2...")
    model = load_dino(device)

    print(f"Loading FFHQ features from {cache_path}...")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    ffhq_features = cache["features"].to(device)
    print(f"  FFHQ features: {ffhq_features.shape}")

    print(f"Extracting features from {img_dir}...")
    gen_features, gen_names = extract_features(model, img_dir, batch_size=batch_size, device=device)
    gen_features = gen_features.to(device)

    print(f"Computing K-NN scores (K={k})...")
    scores = {}
    chunk_size = 500
    for i in range(0, len(gen_features), chunk_size):
        chunk = gen_features[i:i + chunk_size]
        sim = chunk @ ffhq_features.T
        topk_sim, _ = sim.topk(k, dim=1)
        mean_sim = topk_sim.mean(dim=1)

        for j, (name, score) in enumerate(zip(gen_names[i:i + len(chunk)], mean_sim)):
            scores[name] = score.item()

    sorted_scores = sorted(scores.items(), key=lambda x: x[1])

    print(f"\n=== Score Distribution ===")
    vals = list(scores.values())
    import numpy as np
    vals_np = np.array(vals)
    print(f"  Mean:   {vals_np.mean():.4f}")
    print(f"  Median: {np.median(vals_np):.4f}")
    print(f"  Std:    {vals_np.std():.4f}")
    print(f"  Min:    {vals_np.min():.4f}  ({sorted_scores[0][0]})")
    print(f"  Max:    {vals_np.max():.4f}  ({sorted_scores[-1][0]})")
    print(f"\n  Worst 5:")
    for name, s in sorted_scores[:5]:
        print(f"    {name}: {s:.4f}")
    print(f"  Best 5:")
    for name, s in sorted_scores[-5:]:
        print(f"    {name}: {s:.4f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nSaved {len(scores)} scores to {out_path}")

    return scores


def build_pairs_from_scores(scores_path, out_path, n_pairs=10000, top_pct=0.3, bottom_pct=0.3):
    """
    スコア上位 vs 下位からペアをランダムに構成。
    top_pct: preferred候補にする上位割合
    bottom_pct: rejected候補にする下位割合
    """
    with open(scores_path) as f:
        scores = json.load(f)

    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    n = len(sorted_items)
    n_bottom = int(n * bottom_pct)
    n_top = int(n * top_pct)

    bottom = sorted_items[:n_bottom]
    top = sorted_items[-n_top:]

    print(f"Top {n_top} (preferred): score {top[0][1]:.4f} ~ {top[-1][1]:.4f}")
    print(f"Bottom {n_bottom} (rejected): score {bottom[0][1]:.4f} ~ {bottom[-1][1]:.4f}")

    random.seed(42)
    pairs = []
    for _ in range(n_pairs):
        pref_name, pref_score = random.choice(top)
        rej_name, rej_score = random.choice(bottom)
        if pref_name != rej_name:
            pairs.append({
                "preferred": pref_name,
                "rejected": rej_name,
                "score_preferred": pref_score,
                "score_rejected": rej_score,
            })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\nGenerated {len(pairs)} pairs")
    print(f"Saved to {out_path}")
    return pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--precompute", action="store_true", help="FFHQ特徴量を事前計算")
    p.add_argument("--score", action="store_true", help="生成画像をスコアリング")
    p.add_argument("--build_pairs", action="store_true", help="スコアからペア構成")

    p.add_argument("--data_dir", type=str, default="./images256_ffhq_celebahq")
    p.add_argument("--img_dir", type=str, default="dpo_data/generated")
    p.add_argument("--cache", type=str, default="dpo_data/ffhq_features.pt")
    p.add_argument("--scores", type=str, default="dpo_data/auto_scores.json")
    p.add_argument("--out", type=str, default="dpo_data/auto_pairs.json")

    p.add_argument("--k", type=int, default=10, help="K-NN のK")
    p.add_argument("--n", type=int, default=10000, help="生成ペア数")
    p.add_argument("--top_pct", type=float, default=0.3, help="preferred候補の上位割合")
    p.add_argument("--bottom_pct", type=float, default=0.3, help="rejected候補の下位割合")
    p.add_argument("--batch_size", type=int, default=256)

    args = p.parse_args()

    if args.precompute:
        precompute_ffhq(args.data_dir, args.cache, batch_size=args.batch_size)
    elif args.score:
        score_images(args.img_dir, args.cache, args.scores, k=args.k, batch_size=args.batch_size)
    elif args.build_pairs:
        build_pairs_from_scores(args.scores, args.out, n_pairs=args.n,
                                top_pct=args.top_pct, bottom_pct=args.bottom_pct)
    else:
        p.print_help()
