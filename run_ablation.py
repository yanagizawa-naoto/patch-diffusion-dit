"""
事前学習ステップ数 vs セグメンテーション能力のablation study。
各チェックポイントでマルチタスク学習→unseen 200枚でmIoU計算。
"""
import subprocess
import json
import sys
import random
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image

PYTHON = "/home/naoto/venv_fp8/bin/python"
BASE_DIR = Path("runs/patch_dit_ffhq512_20260505_112612")
SEG_DIR = "./celebamask_hq_15class"
FACE_DIR = "./images256_ffhq_celebahq"

STEPS_TO_TEST = [10000, 50000, 100000, 200000, 300000]
SEG_TRAIN_STEPS = 50000

COLOR_TO_CLASS = {
    (0, 0, 0): 0, (204, 178, 153): 1, (255, 140, 80): 2,
    (0, 100, 200): 3, (0, 150, 255): 4, (139, 90, 43): 5,
    (160, 110, 60): 6, (255, 200, 150): 7, (255, 210, 170): 8,
    (200, 50, 50): 9, (255, 80, 80): 10, (220, 40, 40): 11,
    (60, 40, 20): 12, (230, 200, 180): 13,
}
CLASS_NAMES = ['background', 'skin', 'nose', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
               'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'neck']
N_CLASSES = len(CLASS_NAMES)


def rgb_to_class_map(rgb_img, tolerance=40):
    h, w, _ = rgb_img.shape
    class_map = np.zeros((h, w), dtype=np.int32)
    for color, cls_id in COLOR_TO_CLASS.items():
        color_arr = np.array(color, dtype=np.float32)
        dist = np.sqrt(((rgb_img.astype(np.float32) - color_arr) ** 2).sum(axis=-1))
        class_map[dist < tolerance] = cls_id
    return class_map


def train_seg(ckpt_path, out_dir):
    cmd = (
        f"{PYTHON} train_multitask.py "
        f"--ckpt {ckpt_path} "
        f"--face_dir {FACE_DIR} "
        f"--seg_dir {SEG_DIR} "
        f"--seg_fraction 0.1 "
        f"--seg_ratio 0.1 "
        f"--batch_size 64 "
        f"--lr 1e-5 "
        f"--total_steps {SEG_TRAIN_STEPS} "
        f"--sample_every 10000 "
        f"--save_every {SEG_TRAIN_STEPS} "
        f"--log_every 500 "
        f"--out_dir {out_dir}"
    )
    print(f"\n{'='*60}")
    print(f"  Training: {ckpt_path} -> {out_dir}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: Training failed for {ckpt_path}")
        return False
    return True


def evaluate_seg(ckpt_path, n_eval=200):
    sys.path.insert(0, ".")
    from model import PatchDiffusionDiT, make_position_grid
    from train_multitask import SegDataset, sample_segmentation

    device = "cuda"

    train_set = SegDataset(SEG_DIR, img_size=512, fraction=0.1)
    train_stems = set(p[0].stem for p in train_set.pairs)

    seg_dir = Path(SEG_DIR)
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    unseen = [(seg_dir / "images" / f"{mf.stem}.jpg", mf)
              for mf in sorted((seg_dir / "masks_vis").glob("*.png"))
              if mf.stem not in train_stems and (seg_dir / "images" / f"{mf.stem}.jpg").exists()]

    random.seed(42)
    eval_set = random.sample(unseen, min(n_eval, len(unseen)))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = PatchDiffusionDiT(
        img_size=512, patch_size=32, hidden_size=768,
        depth=12, num_heads=12, bottleneck_dim=128,
    )
    sd = ckpt.get("model")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    total_correct = 0
    total_pixels = 0

    batch_size = 8
    for i in range(0, len(eval_set), batch_size):
        batch = eval_set[i:i + batch_size]
        imgs = torch.stack([transform(Image.open(p[0]).convert("RGB")) for p in batch]).to(device)
        pred_masks = sample_segmentation(model, imgs, steps=50, device=device)

        for j, (_, gt_path) in enumerate(batch):
            gt_rgb = np.array(Image.open(gt_path).convert("RGB").resize((512, 512), Image.LANCZOS))
            pred_rgb = ((pred_masks[j].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            gt_cls = rgb_to_class_map(gt_rgb)
            pred_cls = rgb_to_class_map(pred_rgb)
            for c_gt in range(N_CLASSES):
                for c_pred in range(N_CLASSES):
                    confusion[c_gt, c_pred] += ((gt_cls == c_gt) & (pred_cls == c_pred)).sum()
            total_correct += (gt_cls == pred_cls).sum()
            total_pixels += gt_cls.size

    pixel_acc = total_correct / total_pixels
    ious = []
    for c in range(N_CLASSES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        if tp + fp + fn > 0:
            ious.append(tp / (tp + fp + fn))
        else:
            ious.append(0.0)
    miou = np.mean(ious)

    del model
    torch.cuda.empty_cache()

    return {"pixel_accuracy": float(pixel_acc), "miou": float(miou), "per_class_iou": {CLASS_NAMES[c]: float(ious[c]) for c in range(N_CLASSES)}}


def main():
    results = []

    for pretrain_step in STEPS_TO_TEST:
        ckpt_name = f"ckpt_0{pretrain_step:06d}.pt"
        ckpt_path = BASE_DIR / ckpt_name
        out_dir = f"runs/ablation_seg_{pretrain_step // 1000}k"

        final_ckpt = Path(out_dir) / f"ckpt_{SEG_TRAIN_STEPS:07d}.pt"

        if final_ckpt.exists():
            print(f"\nSkipping training for {pretrain_step} (already exists)")
        else:
            success = train_seg(str(ckpt_path), out_dir)
            if not success:
                continue

        print(f"\nEvaluating {pretrain_step}K...")
        metrics = evaluate_seg(str(final_ckpt))
        metrics["pretrain_steps"] = pretrain_step

        print(f"  Pretrain {pretrain_step // 1000}K: "
              f"pixel_acc={metrics['pixel_accuracy']:.4f}, "
              f"mIoU={metrics['miou']:.4f}")

        results.append(metrics)

    # 530Kの結果も追加（既に計算済み）
    existing = Path("assets/seg_metrics.json")
    if existing.exists():
        with open(existing) as f:
            m530 = json.load(f)
        m530["pretrain_steps"] = 530000
        results.append(m530)

    results.sort(key=lambda x: x["pretrain_steps"])

    with open("assets/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("  ABLATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Pretrain':>10s} | {'Pixel Acc':>10s} | {'mIoU':>10s}")
    print("-" * 40)
    for r in results:
        print(f"{r['pretrain_steps']//1000:>8d}K | {r['pixel_accuracy']:>9.1%} | {r['miou']:>9.1%}")

    print(f"\nSaved to assets/ablation_results.json")


if __name__ == "__main__":
    main()
