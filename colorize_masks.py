from PIL import Image
import numpy as np
from pathlib import Path

MASK_DIR = Path("/home/naoto/semantic_segmentaion_augument_diffusion_models/celebamask_hq_identity_disjoint/masks")
VIS_DIR = Path("/home/naoto/semantic_segmentaion_augument_diffusion_models/celebamask_hq_identity_disjoint/masks_vis")
VIS_DIR.mkdir(exist_ok=True)

LABELS = {
    0: ('background', (0, 0, 0)),
    1: ('skin', (204, 178, 153)),
    2: ('nose', (255, 140, 80)),
    3: ('eye_g', (100, 200, 255)),
    4: ('l_eye', (0, 100, 200)),
    5: ('r_eye', (0, 150, 255)),
    6: ('l_brow', (139, 90, 43)),
    7: ('r_brow', (160, 110, 60)),
    8: ('l_ear', (255, 200, 150)),
    9: ('r_ear', (255, 210, 170)),
    10: ('mouth', (200, 50, 50)),
    11: ('u_lip', (255, 80, 80)),
    12: ('l_lip', (220, 40, 40)),
    13: ('hair', (60, 40, 20)),
    14: ('hat', (255, 220, 0)),
    15: ('ear_r', (255, 0, 255)),
    16: ('neck_l', (180, 0, 200)),
    17: ('neck', (230, 200, 180)),
    18: ('cloth', (80, 120, 200)),
}

palette = np.zeros((19, 3), dtype=np.uint8)
for lid, (name, color) in LABELS.items():
    palette[lid] = color

def colorize(mask_path):
    mask = np.array(Image.open(mask_path))
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for lid in range(19):
        rgb[mask == lid] = palette[lid]
    return Image.fromarray(rgb)

# 凡例画像を生成
def make_legend():
    cell_h, cell_w = 30, 200
    rows = len(LABELS)
    legend = Image.new('RGB', (cell_w, cell_h * rows), (255, 255, 255))
    pixels = np.array(legend)
    for lid, (name, color) in LABELS.items():
        y = lid * cell_h
        pixels[y:y+cell_h, 0:cell_h, :] = color
    legend = Image.fromarray(pixels)

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(legend)
    for lid, (name, color) in LABELS.items():
        y = lid * cell_h
        draw.rectangle([0, y, cell_h, y + cell_h], fill=color)
        text_color = (0, 0, 0) if sum(color) > 300 else (255, 255, 255)
        draw.text((cell_h + 5, y + 5), f"{lid}: {name}", fill=text_color)
    return legend

masks = sorted(MASK_DIR.glob("*.png"))
print(f"Colorizing {len(masks)} masks...")
for i, p in enumerate(masks):
    vis = colorize(p)
    vis.save(VIS_DIR / p.name)
    if (i + 1) % 1000 == 0 or (i + 1) == len(masks):
        print(f"  {i+1}/{len(masks)}")

legend = make_legend()
legend.save(VIS_DIR / "_legend.png")
print(f"Done! Saved to {VIS_DIR}")
print(f"Legend: {VIS_DIR / '_legend.png'}")
