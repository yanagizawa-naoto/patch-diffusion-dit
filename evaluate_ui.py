"""
DPO用ペア比較UI。
2枚の画像を並べて表示し、どちらが良いかを選ぶ。

Usage:
    python evaluate_ui.py --img_dir dpo_data/generated --out dpo_data/pairs.json --n 100

    # 途中再開OK
    python evaluate_ui.py --img_dir dpo_data/generated --out dpo_data/pairs.json --n 100 --resume
"""
import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def make_comparison_image(img_path_a, img_path_b, out_path):
    """2枚の画像を左右に並べた比較画像を作成。"""
    a = Image.open(img_path_a)
    b = Image.open(img_path_b)
    w, h = a.size

    gap = 20
    label_h = 40
    canvas = Image.new("RGB", (w * 2 + gap, h + label_h), (40, 40, 40))
    canvas.paste(a, (0, label_h))
    canvas.paste(b, (w + gap, label_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text((w // 2 - 40, 5), "[L] Left", fill=(100, 200, 255), font=font)
    draw.text((w + gap + w // 2 - 45, 5), "[R] Right", fill=(255, 200, 100), font=font)

    canvas.save(out_path)


def evaluate_pairs(img_dir, out_path, n_pairs=100, resume=False):
    img_dir = Path(img_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*.png"))
    if len(images) < 2:
        print(f"Not enough images in {img_dir}")
        return

    pairs = []
    evaluated_set = set()
    if resume and out_path.exists():
        with open(out_path) as f:
            pairs = json.load(f)
        for p in pairs:
            key = tuple(sorted([p["preferred"], p["rejected"]]))
            evaluated_set.add(key)
        print(f"Loaded {len(pairs)} existing pairs")

    random.seed(42)
    all_candidates = []
    img_names = [img.name for img in images]
    for i in range(len(img_names)):
        for j in range(i + 1, len(img_names)):
            all_candidates.append((img_names[i], img_names[j]))
    random.shuffle(all_candidates)

    remaining = [
        (a, b) for a, b in all_candidates
        if tuple(sorted([a, b])) not in evaluated_set
    ]

    target = n_pairs - len(pairs)
    if target <= 0:
        print(f"Already have {len(pairs)} pairs (target: {n_pairs})")
        return

    remaining = remaining[:target * 2]

    print(f"\n=== DPO ペア比較UI ===")
    print(f"目標: {n_pairs} ペア, 評価済み: {len(pairs)}, 残り: {target}")
    print(f"操作: l=左が良い, r=右が良い, s=スキップ, q=保存して終了\n")

    viewer_proc = None
    tmp_path = Path(tempfile.mkdtemp()) / "comparison.png"
    done_now = 0

    try:
        for idx, (name_a, name_b) in enumerate(remaining):
            if done_now >= target:
                break

            if random.random() < 0.5:
                left, right = name_a, name_b
            else:
                left, right = name_b, name_a

            make_comparison_image(
                img_dir / left, img_dir / right, tmp_path
            )

            if viewer_proc is not None:
                viewer_proc.terminate()
                viewer_proc.wait()
            viewer_proc = subprocess.Popen(
                ["eog", str(tmp_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            while True:
                user_input = input(
                    f"  [{len(pairs) + 1}/{n_pairs}] L={left} vs R={right} (l/r/s/q): "
                ).strip().lower()
                if user_input == "q":
                    _save_pairs(pairs, out_path)
                    print(f"\nSaved {len(pairs)} pairs to {out_path}")
                    return
                elif user_input == "s":
                    break
                elif user_input == "l":
                    pairs.append({"preferred": left, "rejected": right})
                    done_now += 1
                    break
                elif user_input == "r":
                    pairs.append({"preferred": right, "rejected": left})
                    done_now += 1
                    break
                else:
                    print("    l(左), r(右), s(skip), q(quit) のいずれかを入力")

            if len(pairs) % 10 == 0:
                _save_pairs(pairs, out_path)
                print(f"  [auto-save] {len(pairs)} pairs saved")

    except KeyboardInterrupt:
        pass
    finally:
        if viewer_proc is not None:
            viewer_proc.terminate()
            viewer_proc.wait()
        tmp_path.unlink(missing_ok=True)

    _save_pairs(pairs, out_path)
    print(f"\nDone. {len(pairs)} pairs saved to {out_path}")


def _save_pairs(pairs, path):
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, default="dpo_data/generated")
    p.add_argument("--out", type=str, default="dpo_data/pairs.json")
    p.add_argument("--n", type=int, default=100, help="目標ペア数")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    evaluate_pairs(args.img_dir, args.out, n_pairs=args.n, resume=args.resume)
