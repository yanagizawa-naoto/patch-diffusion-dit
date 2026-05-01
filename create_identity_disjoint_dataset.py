"""
CelebAMask-HQから人物重複なしの顔画像+セマンティックマスクペアを作成する。

チェーン:
  CelebAMask-HQ index -> CelebA-HQ-to-CelebA-mapping.txt -> orig_file
  -> identity_CelebA.txt -> person_id
  -> person_idごとに1枚だけ採用
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np
import random

random.seed(42)

BASE = Path("/home/naoto/semantic_segmentaion_augument_diffusion_models")
CELEBA_HQ_DIR = BASE / "CelebAMask-HQ" / "CelebAMask-HQ"
IMG_DIR = CELEBA_HQ_DIR / "CelebA-HQ-img"
MASK_ANNO_DIR = CELEBA_HQ_DIR / "CelebAMask-HQ-mask-anno"
MAPPING_FILE = CELEBA_HQ_DIR / "CelebA-HQ-to-CelebA-mapping.txt"
IDENTITY_FILE = BASE / "identity_CelebA.txt"

OUT_DIR = BASE / "celebamask_hq_identity_disjoint"
OUT_IMG_DIR = OUT_DIR / "images"
OUT_MASK_DIR = OUT_DIR / "masks"

LABEL_MAP = {
    'background': 0,
    'skin': 1,
    'nose': 2,
    'eye_g': 3,
    'l_eye': 4,
    'r_eye': 5,
    'l_brow': 6,
    'r_brow': 7,
    'l_ear': 8,
    'r_ear': 9,
    'mouth': 10,
    'u_lip': 11,
    'l_lip': 12,
    'hair': 13,
    'hat': 14,
    'ear_r': 15,
    'neck_l': 16,
    'neck': 17,
    'cloth': 18,
}

def load_mapping():
    """CelebAMask-HQ index -> CelebA orig_file"""
    mapping = {}
    with open(MAPPING_FILE) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                idx = int(parts[0])
                orig_file = parts[2]
                mapping[idx] = orig_file
    return mapping

def load_identity():
    """CelebA filename -> person_id"""
    identity = {}
    with open(IDENTITY_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                identity[parts[0]] = int(parts[1])
    return identity

def merge_masks(idx):
    """個別パーツマスクを1枚の統合セマンティックマスクに結合する"""
    folder_idx = idx // 2000
    folder = MASK_ANNO_DIR / str(folder_idx)

    merged = np.zeros((512, 512), dtype=np.uint8)

    for label_name, label_id in LABEL_MAP.items():
        if label_name == 'background':
            continue
        mask_file = folder / f"{idx:05d}_{label_name}.png"
        if mask_file.exists():
            mask = np.array(Image.open(mask_file).convert('L'))
            merged[mask > 128] = label_id

    return Image.fromarray(merged, mode='L')

def main():
    print("Loading mapping files...")
    hq_to_celeba = load_mapping()
    celeba_identity = load_identity()

    print(f"CelebAMask-HQ images: {len(hq_to_celeba)}")
    print(f"CelebA identities: {len(set(celeba_identity.values()))}")

    # CelebAMask-HQ index -> person_id
    idx_to_person = {}
    unmapped = 0
    for idx, orig_file in hq_to_celeba.items():
        if orig_file in celeba_identity:
            idx_to_person[idx] = celeba_identity[orig_file]
        else:
            unmapped += 1

    print(f"Mapped: {len(idx_to_person)}, Unmapped: {unmapped}")

    # person_id -> list of CelebAMask-HQ indices
    person_to_indices = defaultdict(list)
    for idx, pid in idx_to_person.items():
        person_to_indices[pid].append(idx)

    unique_persons = len(person_to_indices)
    print(f"Unique persons in CelebAMask-HQ: {unique_persons}")

    # 各person_idから1枚ランダムに選択
    selected_indices = []
    for pid, indices in person_to_indices.items():
        chosen = random.choice(indices)
        selected_indices.append(chosen)

    selected_indices.sort()
    print(f"Selected {len(selected_indices)} identity-disjoint images")

    # 出力ディレクトリ作成
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    # 画像とマスクのコピー/生成
    success = 0
    for i, idx in enumerate(selected_indices):
        src_img = IMG_DIR / f"{idx}.jpg"
        if not src_img.exists():
            print(f"  Warning: {src_img} not found, skipping")
            continue

        # 画像コピー
        dst_img = OUT_IMG_DIR / f"{idx:05d}.jpg"
        shutil.copy2(src_img, dst_img)

        # 統合マスク生成
        merged_mask = merge_masks(idx)
        dst_mask = OUT_MASK_DIR / f"{idx:05d}.png"
        merged_mask.save(dst_mask)

        success += 1
        if (i + 1) % 500 == 0 or (i + 1) == len(selected_indices):
            print(f"  Progress: {i+1}/{len(selected_indices)}")

    print(f"\nDone! {success} identity-disjoint image-mask pairs created.")
    print(f"  Images: {OUT_IMG_DIR}")
    print(f"  Masks:  {OUT_MASK_DIR}")

    # ラベルマップ保存
    with open(OUT_DIR / "label_map.txt", "w") as f:
        for name, lid in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
            f.write(f"{lid}\t{name}\n")
    print(f"  Label map: {OUT_DIR / 'label_map.txt'}")

    # 統計情報
    imgs_per_person = [len(v) for v in person_to_indices.values()]
    print(f"\n--- Statistics ---")
    print(f"Total unique persons: {unique_persons}")
    print(f"Max images per person: {max(imgs_per_person)}")
    print(f"Mean images per person: {sum(imgs_per_person)/len(imgs_per_person):.2f}")
    print(f"Persons with 1 image: {sum(1 for x in imgs_per_person if x == 1)}")
    print(f"Persons with 5+ images: {sum(1 for x in imgs_per_person if x >= 5)}")
    print(f"Persons with 10+ images: {sum(1 for x in imgs_per_person if x >= 10)}")
    print(f"Persons with 15+ images: {sum(1 for x in imgs_per_person if x >= 15)}")

if __name__ == "__main__":
    main()
