"""
FLUX.1 VAEで画像をlatentにエンコードして保存するスクリプト。

使い方:
  python encode_latents.py --img_dir ./images512x512 --out_dir ./latents_flux1_256 --img_size 256
  python encode_latents.py --img_dir ./images512x512 --out_dir ./latents_flux1_512 --img_size 512
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm import tqdm


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading FLUX.1 VAE from {args.vae_id}...")
    vae = AutoencoderKL.from_pretrained(
        args.vae_id, subfolder=args.vae_subfolder, torch_dtype=torch.float16
    )
    vae = vae.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    img_dir = Path(args.img_dir)
    paths = sorted(img_dir.glob("*.png"))
    if not paths:
        paths = sorted(img_dir.glob("*.jpg"))

    print(f"Encoding {len(paths)} images at {args.img_size}x{args.img_size}...")

    # バッチ処理
    batch = []
    names = []
    encoded = 0

    for p in tqdm(paths):
        img = Image.open(p).convert("RGB")
        img_t = transform(img)
        batch.append(img_t)
        names.append(p.stem)

        if len(batch) == args.batch_size:
            _encode_batch(vae, batch, names, out_dir, device)
            encoded += len(batch)
            batch, names = [], []

    if batch:
        _encode_batch(vae, batch, names, out_dir, device)
        encoded += len(batch)

    print(f"Done! {encoded} latents saved to {out_dir}")

    # 最初のlatentの情報を表示
    sample = torch.load(out_dir / f"{paths[0].stem}.pt", weights_only=True)
    print(f"Latent shape: {sample.shape}")
    print(f"Latent range: [{sample.min():.3f}, {sample.max():.3f}]")


@torch.no_grad()
def _encode_batch(vae, batch, names, out_dir, device):
    imgs = torch.stack(batch).to(device, dtype=torch.float16)
    latents = vae.encode(imgs).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    for name, lat in zip(names, latents):
        torch.save(lat.cpu().float(), out_dir / f"{name}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, default="./images512x512")
    p.add_argument("--out_dir", type=str, default="./latents_flux1_256")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--vae_id", type=str, default="black-forest-labs/FLUX.1-dev",
                   help="HuggingFace VAE model ID")
    p.add_argument("--vae_subfolder", type=str, default="vae",
                   help="VAEのサブフォルダ (パイプラインリポジトリの場合)")
    main(p.parse_args())
