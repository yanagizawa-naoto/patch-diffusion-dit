# MMDiT (Multimodal Diffusion Transformer): 論文+実装の完全理解ノート

**論文**: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
**著者**: Patrick Esser, Sumith Kulal, Andreas Blattmann, et al. (Stability AI)
**arXiv**: 2403.03206 | **会議**: ICML 2024
**モデル**: Stable Diffusion 3 の技術基盤
**公式実装**: https://github.com/Stability-AI/sd3-ref
**HuggingFace**: diffusers ライブラリに SD3Pipeline として統合済み

---

## 1. 論文の3つの主要貢献

1. **Rectified Flow + Logit-Normal Timestep Sampling**: 61種の定式化を比較し最良の訓練設定を特定
2. **MM-DiT アーキテクチャ**: テキストと画像で別々の重みを持ちつつ、attentionを共有する双方向情報フロー
3. **スケーリング特性**: 450M〜8Bパラメータで滑らかに品質が向上し、飽和なし

---

## 2. Rectified Flow 定式化

### 2.1 順方向プロセス (直線補間)
```
z_t = (1-t) * x_0 + t * ε,  ε ~ N(0, I)
```
- t=0: データ x_0
- t=1: ノイズ ε
- データとノイズを直線で結ぶ（曲線パスより理論的に少ないステップで到達可能）

### 2.2 速度場 (Velocity Field)
モデル v_Θ(z_t, t) が速度を予測。逆ODEで生成:
```
dy_t = v_Θ(y_t, t) dt
```

### 2.3 条件付きフローマッチング損失
```
L_CFM = E_{t, ε, x_0} || v_Θ(z_t, t) - (ε - x_0) ||²
```
- 速度の目標は (ε - x_0) = ノイズとデータの差分
- t=0付近: 目標≈-x_0 (自明)、t=1付近: 目標≈ε (自明)
- **中間のtが最も困難** → ここに訓練を集中すべき

### 2.4 ノイズ予測への変換
速度予測をノイズ予測に変換可能:
```
ε_Θ = -2/(λ'_t * b_t) * (v_Θ - a'_t/a_t * z)
```
統一的な重み付き損失:
```
L_w = -1/2 * E_{t~π(t)} [w_t * λ'_t * ||ε_Θ - ε||²]
```

---

## 3. Logit-Normal Timestep Sampling (核心的イノベーション)

### 定義
```
π_ln(t; m, s) = 1/(s√(2π)) * 1/(t(1-t)) * exp(-(logit(t)-m)² / (2s²))
```
実装: u ~ N(m, s²) → t = sigmoid(u)

### パラメータ
- m (位置): 負→データ寄り、正→ノイズ寄り
- s (スケール): 分布の幅を制御

### 最良設定: rf/lognorm(0.00, 1.00)
- 61種の定式化中ランク1位 (平均ランク1.54)
- 特に**少ステップサンプリング (5ステップ)** で顕著に優位
- 中間timestepに訓練を集中させ、困難な遷移領域の学習を強化

### 比較結果 (25ステップ, ImageNet)
| 定式化 | CLIP | FID |
|-------|------|-----|
| rf (uniform) | 0.247 | 49.70 |
| **rf/lognorm(0.00, 1.00)** | **0.250** | **45.78** |
| rf/mode(1.75) | 0.253 | 44.39 |
| eps/linear | - | 比較的劣る |

---

## 4. 解像度依存タイムステップシフト

訓練解像度 n から推論解像度 m への変換:
```
t_m = (√(m/n) * t_n) / (1 + (√(m/n) - 1) * t_n)
```

SD3では alpha=3.0 (shift=3.0) を使用:
```python
sigma = shift * timestep / (1 + (shift - 1) * timestep)  # shift=3.0
```
256x256で事前学習 → 1024x1024で微調整時に適用。高ノイズレベル側にシフトすることで品質向上。

---

## 5. MM-DiT アーキテクチャ

### 5.1 設計哲学
テキストと画像で**別々の重み**を持ちつつ、**attentionは共有**する双方向情報フロー。
CrossAttention (一方向) でもSimple Concatenation (共有重み) でもない、第三の選択肢。

### 5.2 モデルサイズの決定方法 — depth一つで全て決まる
```python
hidden_size = 64 * depth
num_heads = depth
num_layers = depth  # MM-DiTブロック数
head_dim = 64  # 固定
```

| depth | hidden_size | heads | 概算パラメータ |
|-------|------------|-------|-------------|
| 15 | 960 | 15 | ~450M |
| 18 | 1,152 | 18 | ~900M |
| 24 | 1,536 | 24 | ~1.5B (SD3 Medium) |
| 30 | 1,920 | 30 | ~2.4B |
| 38 | 2,432 | 38 | ~8B |

### 5.3 入力処理

**画像パッチ埋め込み**:
```python
# 16ch VAE latent → 2x2パッチ → 線形射影
Conv2d(in_channels=16, out_channels=hidden_size, kernel_size=2, stride=2)
# (B, 16, H, W) → (B, H/2 * W/2, hidden_size)
# 例: 1024x1024画像 → latent 128x128 → 4096パッチトークン
```

**テキスト埋め込み**:
```python
# 3つのテキストエンコーダ出力を結合
lg_out = cat([CLIP_L_out, CLIP_G_out], dim=-1)  # (B, 77, 2048)
lg_out = zero_pad(lg_out, target_dim=4096)       # (B, 77, 4096) ← ゼロパディング!
context = cat([lg_out, T5_out], dim=-2)           # (B, 77+T5_len, 4096)
context = Linear(4096, hidden_size)(context)      # (B, seq_len, hidden_size)

# プーリング済みテキスト (adaLN条件付け用)
pooled = cat([CLIP_L_pooled, CLIP_G_pooled], dim=-1)  # (B, 2048)
```

**タイムステップ + プーリング済みテキスト → 条件ベクトル c**:
```python
c = TimestepMLP(sinusoidal(t)) + VectorMLP(pooled)  # (B, hidden_size)
```
→ 単純な加算。このcが全ブロックのadaLN変調を駆動。

### 5.4 位置エンコーディング
- 2D正弦波位置埋め込み (sin/cos、最大グリッドサイズからセンタークロップ)
- pos_embed_max_size=192 → 最大3072px相当
- 可変アスペクト比対応: バケットサンプリング + 位置グリッドのセンタークロップ

### 5.5 MM-DiTブロック (双方向ストリーム) — アーキテクチャの核心

各ブロックの処理フロー:

```
入力: context (テキスト), x (画像), c (条件)
    ↓
1. [テキスト] adaLN変調 → Q_c, K_c, V_c (テキスト固有の重み)
   [画像]   adaLN変調 → Q_x, K_x, V_x (画像固有の重み)
    ↓
2. Q = cat(Q_c, Q_x)  # 結合
   K = cat(K_c, K_x)
   V = cat(V_c, V_x)
    ↓
3. Attention(Q, K, V)  # 全トークンが相互にattend (双方向!)
    ↓
4. 出力を分割 → attn_c, attn_x
    ↓
5. [テキスト] gate * attn_c + residual → MLP → gate * mlp_out + residual
   [画像]   gate * attn_x + residual → MLP → gate * mlp_out + residual
    ↓
出力: context', x'
```

**重要**: テキストと画像は**別々のLayerNorm, QKV射影, MLP, adaLN変調パラメータ**を持つが、**attentionだけは共有**。

### 5.6 条件付けメカニズム: adaLN-Zero (DiTから継承、MMDiT固有ではない)

**注意: adaLN-ZeroはMMDiT独自の設計ではなく、DiT (Peebles & Xie, 2023) で導入された条件付け手法。**
MMDiTはこれをそのまま採用している。MMDiT固有の革新は5.5の双方向ストリーム設計。

条件付け手法の比較:
```
[単純加算]      output = features + cond                    → 加算のみ
[Cross-Attn]   output = CrossAttn(Q=feat, K=cond, V=cond)  → SD1/SDXLのテキスト条件付け
[AdaLN]        γ,β = Linear(cond)                          → LayerNormのアフィン変換
               output = γ * LayerNorm(x) + β
[adaLN-Zero]   γ,β,α = Linear(cond)  α=gate, 初期値0      → AdaLN + gate付き残差制御
               output = x + α * Layer(γ * LayerNorm(x) + β)
```

MMDiTでの具体的な使い方 — 各ブロック・各ストリームで6つの変調パラメータを生成:
```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
    Linear(hidden_size, 6 * hidden_size)(SiLU(c)).chunk(6)
```

適用:
```python
# Pre-attention: LayerNormの出力をスケール&シフト (← これがAdaLN部分)
x_modulated = LayerNorm(x) * (1 + scale_msa) + shift_msa
# Post-attention: gateで残差接続の強度を制御 (← これがZero初期化部分)
x = x + gate_msa * attn_out  # gate初期値=0 → 学習初期は恒等変換
# Post-MLP: 同様
x = x + gate_mlp * MLP(LayerNorm(x) * (1 + scale_mlp) + shift_mlp)
```

### 5.7 QK正規化 (RMSNorm)
```python
# Q, Kにhead_dim次元でRMSNormを適用 (学習可能スケール付き)
q = RMSNorm(head_dim, learnable=True)(q)
k = RMSNorm(head_dim, learnable=True)(k)
```
- attention logitの発散を防止
- bf16混合精度でAdamW(eps=1e-15)の安定訓練を可能に
- 高解像度微調整時に必須

### 5.8 最終ブロックの特殊処理
最後のブロック (i == depth-1) はテキストストリームが `pre_only=True`:
- テキスト側はQ,K,Vのみ計算（MLP/post-attention なし）
- joint attention後、テキスト出力は破棄
- 画像ストリームのみ最終層へ

### 5.9 最終層 (FinalLayer)
```python
shift, scale = Linear(hidden_size, 2*hidden_size)(SiLU(c)).chunk(2)
x = LayerNorm(x) * (1 + scale) + shift  # 2パラメータのみ (gateなし)
x = Linear(hidden_size, patch_size² * out_channels)(x)  # (B, N, 2*2*16=64)
x = unpatchify(x)  # (B, 16, H, W)
```

---

## 6. 16チャネルVAE

### SD1/SDXLの4chから16chへ拡張
| 指標 | 4ch | 8ch | 16ch |
|-----|-----|-----|------|
| FID (再構成) | 2.41 | 1.56 | **1.06** |
| SSIM | 0.75 | 0.79 | **0.86** |
| PSNR | 25.12 | 26.40 | **28.62** |

16chを選択した理由: 再構成品質が高く、より容量の大きいモデルのスケーリング性能が向上。

### 潜在空間の正規化
```python
scale_factor = 1.5305
shift_factor = 0.0609
latent_normalized = (latent - shift_factor) * scale_factor  # エンコード後
latent_recovered = latent / scale_factor + shift_factor      # デコード前
```
空潜在変数は0ではなく0.0609で初期化。

---

## 7. テキストエンコーダ構成

| エンコーダ | hidden_size | layers | 出力 | 用途 |
|----------|------------|--------|------|------|
| CLIP-L/14 | 768 | 12 | pooled(768) + seq(77,768) | adaLN + context |
| CLIP-G/14 | 1280 | 32 | pooled(1280) + seq(77,1280) | adaLN + context |
| T5-XXL | 4096 | 24 | seq(T5_len, 4096) | contextのみ |

- 各エンコーダは訓練中に独立して46.3%の確率でドロップ
- 推論時: T5なし(CLIP-only)でも審美性は同等、テキスト追従は微減、タイポグラフィは明確に低下
- CLIP出力はpenultimate layer (-2) を使用

---

## 8. テンソルシェイプの完全フロー (1024x1024, depth=24)

| ステップ | テンソル | シェイプ |
|--------|--------|---------|
| 画像latent入力 | x | (B, 16, 128, 128) |
| パッチ埋め込み後 | x | (B, 4096, 1536) |
| + 位置エンコーディング | x | (B, 4096, 1536) |
| テキスト (射影前) | context | (B, 154, 4096) |
| テキスト (射影後) | context | (B, 154, 1536) |
| 条件ベクトル | c | (B, 1536) |
| Joint Attention Q,K,V | q,k,v | (B, 4250, 1536) |
| SDPA用reshape | q,k,v | (B, 24, 4250, 64) |
| adaLN変調パラメータ/ブロック | mod | (B, 9216) = 6×1536 |
| 最終層出力 | x | (B, 4096, 64) |
| Unpatchify後 | output | (B, 16, 128, 128) |

---

## 9. アーキテクチャ比較 (CC12Mでのアブレーション)

| 方式 | 説明 | 結果 |
|-----|------|------|
| Vanilla DiT | テキスト・画像を単純結合、共有重み | 最悪 |
| UViT | ハイブリッド | 初期学習は速いがプラトー |
| CrossDiT | テキスト→画像のcross-attention | UViTより全体的に良い |
| **MM-DiT (2重み)** | **別重み + 共有attention** | **大幅に最良** |
| MM-DiT (3重み) | Q,K,Vも別重み | わずかに改善、VRAM増 |

→ 2重みの双方向設計が最適。

---

## 10. スケーリング研究結果

### 検証損失
- モデルサイズ・訓練ステップ増加で滑らかに減少、飽和なし
- 8Bモデルで約5×10²² FLOPs

### GenEval ベンチマーク (512x512)
| モデル | Overall | Single | Two | Count | Color | Position | ColorAttr |
|-------|---------|--------|-----|-------|-------|----------|-----------|
| DALL-E 3 | 0.67 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 |
| d=24 | 0.62 | 0.98 | 0.74 | 0.63 | 0.67 | 0.34 | 0.36 |
| d=38 | 0.68 | 0.98 | 0.84 | 0.66 | 0.74 | 0.40 | 0.43 |
| **d=38+DPO 1024** | **0.74** | **0.99** | **0.94** | 0.72 | **0.89** | 0.33 | **0.60** |

### サンプリング効率 — 大きいモデルほど少ステップで良い
| モデル | 5/50ステップ劣化 | パス長 |
|-------|----------------|-------|
| d=15 | 4.30% | 191.13 |
| d=30 | 3.59% | 187.96 |
| d=38 | **2.71%** | **185.96** |

大きいモデルはより直線的な軌道を学習 → 少ステップサンプリングに有利。

---

## 11. 訓練詳細

| 項目 | 設定 |
|-----|------|
| Optimizer | AdamW (eps=1e-15) |
| 精度 | bf16混合精度 |
| 事前学習 | 256x256, batch 4096, 500kステップ |
| 微調整 | 1024x1024, 可変アスペクト比, バケットサンプリング |
| タイムステップシフト | alpha=3.0 (微調整時) |
| キャプション | 合成キャプション(CogVLM生成) 50% + オリジナル 50% |
| テキストエンコーダドロップ | 各エンコーダ独立に46.3% |
| CFGドロップ | 詳細非公開 |

合成キャプション混合による改善:
- 全体: 43.27% → **49.78%**
- 位置理解: 6.50% → **18.00%**
- 色属性: 11.75% → **24.75%**

---

## 12. 実装の非自明なポイント

1. **両ストリームは同じ条件ベクトルcを受け取る** (adaLN-Zero自体はDiT由来): ただし別々のLinear射影があるため、実際の変調は異なる
2. **CLIP埋め込みはゼロパディングで4096次元に**: T5と次元を揃えるための簡易的だが有効な手法
3. **T5はadaLNに寄与しない**: pooled出力がないため。sequence attentionのみで情報を提供
4. **最終ブロックでcontextは破棄**: テキストの更新は不要なので計算を節約
5. **MLP活性化はGELU(tanh近似)、拡張率4倍**: SwiGLUもサポートされるがデフォルトではない
6. **QKVは単一Linear(dim, 3*dim)で融合**: 分割して使用
7. **位置エンコーディングはセンタークロップ**: 可変解像度対応のため最大グリッドから中央を切り出す
8. **空latentは0.0609で初期化**: VAEのshift_factorに合わせる
9. **Attention は is_causal=False**: テキスト↔画像は完全双方向
10. **SD3.5ではdual attention追加**: 特定ブロックで画像のみの追加self-attention

---

## 13. SD3.5 の拡張 (Dual Attention)

SD3.5では一部のブロックに画像ストリーム専用の追加self-attentionを導入:
```python
# 9個のadaLNパラメータ (通常6 + 追加self-attention用3)
# Joint attention後、画像トークンのみで追加attentionを実行
attn_out2 = SelfAttention(image_tokens_only)
x = x + gate_msa2 * attn_out2
```

---

## 14. 我々のプロジェクトへの適用可能性

### 直接利用
- SD3のVAE (16ch) は高品質な潜在空間を提供 → Latent Diffusionの基盤として利用可能
- テキスト条件付けの代わりにセグメンテーションマスク条件付けに改造可能

### MMDiTの設計からの学び
- 双方向情報フローは、マスク↔画像の相互条件付けに有効な可能性
- adaLN-Zero変調 (DiT由来) は安定した条件付けメカニズム
- QK正規化はbf16訓練の安定性に必須
- Rectified Flow + Logit-Normal は拡散モデルの標準的な選択肢として採用すべき
- depth一つでモデルサイズを制御する設計は、スケーリング実験に便利

### 注意点
- 8Bモデルの訓練には大規模計算リソースが必要
- テキストエンコーダの代わりにセグメンテーション条件を入れる場合、context embedderの設計変更が必要
- 小規模データ(6K枚)でのMMDiT訓練はオーバーフィットのリスク → 小さいdepthが適切
