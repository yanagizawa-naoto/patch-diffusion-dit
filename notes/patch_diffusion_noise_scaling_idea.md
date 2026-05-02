# Patch Diffusionにおけるクロップサイズ依存ノイズスケーリングの提案

## 背景

JiTでは解像度に応じてノイズをスケーリングする:
- 256×256: scale = 1
- 512×512: scale = 2
- 理由: 高解像度ほどピクセル数が多く、ノイズの中から信号を検出しやすくなるため、ノイズを強くしてSNRを一定に保つ
- 公式: scale = √(新解像度² / 基準解像度²)

## Patch Diffusionでも同じ問題が起きる

Patch Diffusionでは訓練時にクロップサイズが変わる:
- Full 256×256 (65K pixels): ピクセル数が多い → 信号検出が容易
- Crop 128×128 (16K pixels): ピクセル数が少ない → 信号検出が困難

同じノイズレベル ε ~ N(0,1) でも、クロップサイズによって実効的なSNRが異なる。
これはクロップ間で学習の難易度が不均一になることを意味する。

## 提案する対策

クロップサイズに応じてノイズをスケーリング:

```python
# 基準を小さいクロップ(128)にする場合:
noise_scale = crop_size / base_crop_size  # 128→1.0, 256→2.0
eps = torch.randn_like(x_0) * noise_scale

# または基準をフル画像(256)にする場合:
noise_scale = math.sqrt(crop_pixels / base_pixels)  # 256→1.0, 128→0.5
eps = torch.randn_like(x_0) * noise_scale
```

## 注意点

- 元のPatch Diffusion論文(U-Net版)はノイズスケーリングなしで動いている
- U-Netは局所的な畳み込みベースなのでこの効果が小さい可能性
- Transformerベース(我々の実装)ではグローバルなAttentionで全トークンを見るため影響が大きい可能性
- 実験で効果を検証する必要がある

## ステータス

未実装・未検証。効果があるかは実験次第。
