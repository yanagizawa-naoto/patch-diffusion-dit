from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys

SRC = Path("/home/naoto/semantic_segmentaion_augument_diffusion_models/images1024x1024")
DST = Path("/home/naoto/semantic_segmentaion_augument_diffusion_models/images512x512")

def resize(path):
    out = DST / path.name
    if out.exists():
        return
    img = Image.open(path)
    img = img.resize((512, 512), Image.LANCZOS)
    img.save(out, "PNG")

if __name__ == "__main__":
    files = sorted(SRC.glob("*.png"))
    total = len(files)
    print(f"Resizing {total} images to 512x512...")

    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(resize, files, chunksize=100), 1):
            if i % 5000 == 0 or i == total:
                print(f"Progress: {i}/{total} ({i*100//total}%)")

    print("Done!")
