from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18)

LABELS = {
    0: ('background', (0, 0, 0)),
    1: ('skin', (204, 178, 153)),
    2: ('nose', (255, 140, 80)),
    3: ('eye_g (メガネ)', (100, 200, 255)),
    4: ('l_eye (左目)', (0, 100, 200)),
    5: ('r_eye (右目)', (0, 150, 255)),
    6: ('l_brow (左眉)', (139, 90, 43)),
    7: ('r_brow (右眉)', (160, 110, 60)),
    8: ('l_ear (左耳)', (255, 200, 150)),
    9: ('r_ear (右耳)', (255, 210, 170)),
    10: ('mouth (口内)', (200, 50, 50)),
    11: ('u_lip (上唇)', (255, 80, 80)),
    12: ('l_lip (下唇)', (220, 40, 40)),
    13: ('hair (髪)', (60, 40, 20)),
    14: ('hat (帽子)', (255, 220, 0)),
    15: ('ear_r (イヤリング)', (255, 0, 255)),
    16: ('neck_l (ネックレス)', (180, 0, 200)),
    17: ('neck (首)', (230, 200, 180)),
    18: ('cloth (服)', (80, 120, 200)),
}

cell_h = 40
color_w = 50
text_w = 300
total_w = color_w + text_w
total_h = cell_h * len(LABELS)

legend = Image.new('RGB', (total_w, total_h), (255, 255, 255))
draw = ImageDraw.Draw(legend)

for lid, (name, color) in LABELS.items():
    y = lid * cell_h
    draw.rectangle([0, y, color_w, y + cell_h], fill=color, outline=(128, 128, 128))

    # テキストは白背景に黒文字で統一（読みやすさ優先）
    draw.rectangle([color_w, y, total_w, y + cell_h], fill=(255, 255, 255))
    draw.text((color_w + 8, y + 8), f"{lid}: {name}", fill=(0, 0, 0), font=FONT)

draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=(128, 128, 128))

out = "/home/naoto/semantic_segmentaion_augument_diffusion_models/celebamask_hq_identity_disjoint/masks_vis/_legend.png"
legend.save(out)
print(f"Saved: {out}")
