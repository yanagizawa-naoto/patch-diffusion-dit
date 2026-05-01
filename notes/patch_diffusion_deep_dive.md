# Patch Diffusion: 論文+実装の完全理解ノート

**論文**: "Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models"
**著者**: Zhendong Wang, Yifan Jiang, Huangjie Zheng, Peihao Wang, Pengcheng He, Zhangyang Wang, Weizhu Chen, Mingyuan Zhou
**所属**: UT Austin + Microsoft Azure AI
**arXiv**: 2304.12526v2 | **会議**: NeurIPS 2023
**公式実装**: https://github.com/Zhendong-Wang/Patch-Diffusion (EDMベース)

---

## 1. コアアイデア

フル画像ではなくランダムクロップしたパッチ単位でscore matchingを行う。
パッチの位置座標を2チャネルとして入力に結合し、モデルが「画像のどの領域か」を認識できるようにする。
推論時はフル座標グリッドを与えるだけで通常通りフル画像を生成する。

---

## 2. 数学的定式化

### 2.1 標準的なScore-based Diffusion (EDM準拠)

デノイザー D_θ(x; σ_t) が以下を最小化:
```
E_{x~p(x)} E_{ε~N(0,σ_t²I)} || D_θ(x+ε; σ_t) - x ||₂²
```

スコア関数:
```
s_θ(x, σ_t) = (D_θ(x; σ_t) - x) / σ_t²
```

### 2.2 パッチワイズScore Matching

パッチ x_{i,j,s} に対して (i,j: 左上座標, s: パッチサイズ):
```
E_{x~p(x), ε~N(0,σ_t²I), (i,j,s)~U} || D_θ(x̃_{i,j,s}; σ_t, i, j, s) - x_{i,j,s} ||₂²
```

- (i,j) は [-1,1] に正規化
- s はカテゴリカル分布からサンプリング
- 座標チャネルの再構成出力はロスから除外

---

## 3. 座標条件付け (Coordinate Conditioning)

### 仕組み
- 画像チャネル(RGB=3ch)に座標チャネル(x,y=2ch)を**結合**して5chとしてU-Netに入力
- 座標は画像の左上=(-1,-1)、右下=(+1,+1) に正規化

### 実装上の重要ポイント
```python
# Patch_EDMPrecond.forward() より
x_in = torch.cat([c_in * x, x_pos], dim=1)  # x_posはc_inでスケーリングしない！
```
- ノイズ画像は c_in = 1/√(σ²+σ_data²) でスケーリングされるが、座標チャネルはスケーリングせず生の[-1,1]のまま結合
- U-Netのin_channelsを img_channels + 2 に設定するだけ
- U-Netアーキテクチャ自体は一切変更なし（EDMのSongUNet/DhariwalUNetそのまま）

### 座標マップの計算 (pachify関数)
```python
x_pos = (pixel_x_coord / (resolution - 1) - 0.5) * 2.0  # -> [-1, 1]
y_pos = (pixel_y_coord / (resolution - 1) - 0.5) * 2.0  # -> [-1, 1]
```
- バッチ内の各サンプルで独立にランダムクロップ位置を決定
- 座標はクロップ位置に応じてオフセット

---

## 4. パッチサイズサンプリング戦略

### デフォルト: 確率的スケジューリング (Stochastic)

画像解像度R, フル画像比率p=0.5 の場合:

**ピクセル空間 (64x64)**:
| パッチサイズ | 確率 | 計算 |
|------------|------|------|
| R (64x64, フル画像) | 0.5 | p |
| R/2 (32x32) | 0.3 | (1-p)×3/5 |
| R/4 (16x16) | 0.2 | (1-p)×2/5 |

**潜在空間 (256x256→32x32 latent)**:
| パッチサイズ | 確率 |
|------------|------|
| 32x32 (フル) | 0.5 |
| 16x16 | 0.5 |

### Progressive スケジューリング (非デフォルト)
小→大の順で固定サイズで訓練。学習率減衰との相性が悪く、Stochasticに劣る。

- CelebA-64: Stochastic FID=1.66 vs Progressive FID=2.05

---

## 5. バッチ乗数 (Batch Multiplier) — 重要な実装トリック

小パッチ時にGPU利用率を維持するため、バッチサイズを比例的に増加:

```python
batch_mul_dict = {512:1, 256:2, 128:4, 64:16, 32:32, 16:64}
```

64x64ベース解像度の場合:
- 64x64パッチ: 1倍
- 32x32パッチ: 2倍
- 16x16パッチ: 4倍

勾配は batch_mul で正規化:
```python
loss.sum().mul(loss_scaling / batch_gpu_total / batch_mul).backward()
```

EMAも調整:
```python
batch_mul_avg = Σ(p_list × [4, 2, 1]) = 2.0  # デフォルト設定時
```

---

## 6. 推論 (Sampling)

推論時は通常のEDMサンプリングと同一。唯一の違いはフル座標グリッドを付加すること:

```python
# フル座標グリッド生成
x_pos = torch.arange(0, resolution) # [0, 1, ..., 63]
y_pos = torch.arange(0, resolution)
x_pos = (x_pos / (resolution - 1) - 0.5) * 2.0  # -> [-1, 1]
y_pos = (y_pos / (resolution - 1) - 0.5) * 2.0  # -> [-1, 1]
```

- 座標マップは全デノイジングステップで不変
- パッチの合成・マージは不要（モデルが直接フル画像を生成）
- NFE=50 (deterministic sampling)
- p=0.5でフル画像訓練が50%含まれるため、特別な遷移は不要

### 画像外挿 (Image Extrapolation)
座標を[-1,1]の外に拡張することで、学習時より大きな画像の生成が可能:
- 256x256で学習 → 384x384, 512x512の生成が可能
- 参照画像を中心に配置し、周囲を生成

---

## 7. Latent Patch Diffusion (LPDM)

256x256画像に対しては、Stable DiffusionのVAEで潜在空間に圧縮してからPatch Diffusionを適用:

```python
img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
latent_scale_factor = 0.18215
# 256x256 → 32x32×4ch latent
images = img_vae.encode(images)['latent_dist'].sample() * latent_scale_factor
```

- VAEは凍結（学習しない）
- U-Netの入力は 4(latent) + 2(座標) = 6ch
- パッチサイズは 32x32(フル) or 16x16 の2段階

---

## 8. 実験結果

### 8.1 フル画像比率 p の影響 (CelebA-64, 16×V100, 200M images)

| p | FID | 訓練時間 |
|---|-----|---------|
| 0.0 (パッチのみ) | 14.51 | 13.6h |
| 0.1 | 3.05 | 20.1h |
| 0.25 | 2.10 | 22.5h |
| **0.5 (デフォルト)** | **1.77** | **24.6h** |
| 0.75 | 1.65 | 42.7h |
| 1.0 (ベースライン) | 1.66 | 48.5h |

→ p=0.5がコスト効率の最適点。ベースラインの約半分の時間で同等品質。

### 8.2 メインの結果

**64x64 ピクセル空間** (16×V100, 200M images):
| データセット | ベースライン FID | Patch Diffusion FID | 速度向上 |
|------------|----------------|-------------------|---------|
| CelebA | 1.66 | 1.77 | ~2倍 |
| FFHQ | 2.60 | 3.11 | ~2倍 |

**256x256 潜在空間** (16×V100):
| データセット | ベースライン FID | LPDM FID | 速度向上 |
|------------|----------------|---------|---------|
| LSUN-Bedroom | 4.32 | **2.75** | ~2倍 |
| LSUN-Church | 4.66 | **2.66** | ~2倍 |

**ImageNet-256 (CFG=1.3)**:
| モデル | FID | GFLOPs |
|-------|-----|--------|
| ADM | 4.59 | 1120 |
| LDM-4 | 3.60 | 103 |
| **LPDM-ADM** | **2.72** | **78** |

### 8.3 小規模データセット (AFHQv2, ~5k枚, 64x64, 75M images)

| データセット | ベースライン FID | Patch Diffusion FID | 速度 |
|------------|----------------|-------------------|------|
| Cat | 4.60 | **3.11** | 2倍 |
| Dog | 4.94 | **4.80** | 2倍 |
| Wild | 2.59 | **1.93** | 2倍 |

→ 小規模データでは品質も向上（パッチクロップがデータ拡張として機能）

---

## 9. 理論的解釈

### MRF (Markov Random Field) 解釈
画像をピクセルグリッドのMRFとしてモデル化:
```
p(x) = (1/Z) Π φ_v(x_v) Π φ_e(x_e)
```
スコア関数が分解可能:
```
∇log p(x) = Σ ∇log φ_v(x_v) + Σ ∇log φ_e(x_e)
```
→ 各パッチは φ_v, φ_e の部分集合に対するscore matchingと解釈できる
→ 座標条件付けは空間的に不変でないスコアを区別するために必要

### 線形回帰解釈
多変量ガウスの場合、パッチスコアマッチングと全体スコアマッチングの違いは測定行列(射影行列P_S)のみ。
画像の冗長性により、部分観測からの全体復元は実現可能。

**注意**: 一般的な場合の収束証明は未解決（future work）。

---

## 10. ハイパーパラメータ一覧

| パラメータ | ピクセル空間 | 潜在空間 |
|----------|------------|---------|
| real_p | 0.5 | 0.5 |
| lr | 2e-4 | 1e-4 |
| batch_size | 256-512 | 1024-4096 |
| dropout | 0.05-0.13 | 0.05-0.13 |
| augment | 0 (OFF) | 0 (OFF) |
| duration | 200M images | 200M images |
| arch | ddpmpp (SongUNet) | adm (DhariwalUNet) |
| NFE (sampling) | 50 | 50 |
| precond | pedm | pedm |

---

## 11. 実装の非自明なポイント

1. **座標はc_inスケーリングをバイパス**: ノイズレベルに依存せず常に[-1,1]
2. **バッチ内で各サンプルが異なるクロップ位置**: 多様性確保
3. **バッチ乗数で小パッチ時のGPU利用率を維持**: 16x16パッチなら4倍のサンプル
4. **勾配はbatch_mulで正規化**: 実効学習率を一定に保つ
5. **デフォルトではFourier位置エンコーディング不使用**: 生の2ch座標を直接入力
6. **U-Net自体は無変更**: 全ての変更はプリコンディショニングラッパー層のみ
7. **Augmentation OFF**: スピードアップはパッチからのみ（拡張と併用しない）
8. **CFG実装**: 無条件予測ではクラスラベルのみゼロ化、座標は維持

---

## 12. 制約と注意点

- 推論コストは変わらない（訓練のみ高速化）
- 十分な訓練予算がある場合、効果は縮小（ベースラインが追いつく）
- 一般的な収束証明なし
- ConvベースのU-Netには直接適用可能だが、DiT等のTransformerベースへの適用は追加設計が必要
- ControlNetファインチューニングにも適用可能（約2倍高速化確認済み）

---

## 13. 我々のプロジェクトへの適用可能性

- FFHQ 70K枚 + CelebAMask-HQ 6K枚で顔画像の拡散モデルを訓練する場合、Patch Diffusionは自然に適用可能
- 特に6K枚のセグメンテーション条件付きモデルではデータ拡張効果が大きい
- 512x512画像に対してはLatent Patch Diffusion (LPDM)が適切
- EDMベースの実装なので、EDM系のコードベースに比較的容易に統合可能
