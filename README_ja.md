# PatchDiffusionDiT: ピクセル空間Flow Matching + DPO + マルチタスクセグメンテーション

[English](README.md) | **日本語**

FFHQ + CelebA-HQ（10万枚の顔画像、512×512）でフルスクラッチ学習した、130Mパラメータのピクセル空間Diffusion Transformer。
事前学習後、**Diffusion-DPO**（Flow Matching向けに改変）で人間の嗜好にアラインメントし、さらにわずか104枚のアノテーション付き画像で**セマンティックセグメンテーション**をマルチタスク事後学習で追加。

## 主要な結果

### 顔画像生成（Flow Matching, Heunサンプラー）

<p align="center">
<img src="assets/denoise_after_dpo.gif" width="600">
</p>

### DPOによる破綻画像の改善

同一シードでの生成結果。500件の人手ペア評価によるDiffusion-DPO適用前後の比較：

<p align="center">
<img src="assets/dpo_tweet.png" width="500">
</p>

### たった104枚の学習データでセマンティックセグメンテーション

以下の画像は全て**未知画像**（104枚の学習セットに含まれていない）：

<p align="center">
<img src="assets/seg_showcase.png" width="800">
</p>

### セグメンテーション品質の推移（0 / 6K / 12K / 50Kステップ）

<p align="center">
<img src="assets/seg_step0000.png" width="180">
<img src="assets/seg_step6000.png" width="180">
<img src="assets/seg_step12000.png" width="180">
<img src="assets/seg_step50000.png" width="180">
</p>

## アーキテクチャ

| パラメータ | 値 |
|-----------|------|
| モデル | PatchDiffusionDiT |
| パラメータ数 | 約130M |
| 画像サイズ | 512 × 512（ピクセル空間） |
| パッチサイズ | 32 |
| 隠れ層次元 | 768 |
| Transformer深さ | 12層 |
| アテンションヘッド数 | 12 |
| ボトルネック次元 | 128 |
| 位置エンコーディング | 2D RoPE |
| FFN | SwiGLU |
| 正規化 | RMSNorm + AdaLN-Zero |
| 予測タイプ | x-prediction with v-loss |
| サンプラー | Heun法（2次ODEソルバー） |

### マルチタスク設計

無条件顔生成と条件付きセグメンテーションの両方を、以下の仕組みで実現：

- **`task_emb`**: 学習可能なタスクembedding（0 = 顔生成、1 = セグメンテーション）。AdaLNを通じてtimestep embeddingに加算
- **`modality_emb`**: 学習可能なモダリティembedding（0 = 条件トークン、1 = デノイズ対象）。パッチトークンに加算
- セグメンテーション時：画像トークンとノイズ付きマスクトークンを結合し、self-attentionで一括処理（MMDiTスタイル）。共有された2D RoPE位置座標により空間的対応を保持

両embeddingはゼロ初期化されており、事前学習済みチェックポイントとの後方互換性を維持。

## 3段階パイプライン

### Stage 1: 事前学習（Flow Matching）

- **データ**: FFHQ（7万枚）+ CelebA-HQ（3万枚）= 10万枚の顔画像（512×512）
- **手法**: Rectified Flowによる Flow Matching、v-loss、logit-normalタイムステップサンプリング
- **Patch Diffusion**: 50%の確率で半解像度のランダムクロップ（256×256）で学習
- **学習**: 53万ステップ、バッチサイズ128、lr=1e-4、AdamW 8bit、BF16 + torch.compile
- **ハードウェア**: RTX 6000 Ada 48GB（約364 img/s、10Kステップあたり約58分）

<p align="center">
<img src="assets/loss_curve.png" width="800">
</p>

**発見**: 40万ステップ以降もlossは減少し続けた（0.0225 → 0.0224）が、**生成品質は目に見えて改善しなかった**。人間が知覚できない高周波の微細なディテールに計算を費やしていた。これがDPOへの動機となった。

### Stage 2: Diffusion-DPO（嗜好アラインメント）

[Diffusion-DPO](https://arxiv.org/abs/2311.12908)（Wallace et al., 2023）を、DDPMのノイズ予測からFlow Matchingの速度(velocity)予測に改変：

**原論文（DDPM）**:
```
L = -log σ(-βT(||ε_w - ε_θ||² - ||ε_w - ε_ref||² - (||ε_l - ε_θ||² - ||ε_l - ε_ref||²)))
```

**本実装（Flow Matching）**:
```
L = -log σ(-β(||v_w - v_θ||² - ||v_w - v_ref||² - (||v_l - v_θ||² - ||v_l - v_ref||²)))
```

主な変更点：
- ノイズ予測MSEを速度予測MSEに置換
- Tファクターを除去（連続時間のため離散ステップ数が不要）
- 前向き過程: DDPMスケジュールの代わりに `z_t = t * x_0 + (1-t) * ε`

**DPO学習の詳細**:
- ブラウザベースのA/B比較UIで500件の人手ペア評価
- β=1000、lr=1e-6、50エポック
- 参照モデル: 事前学習済みベースモデルのfrozenコピー

**自動DPOパイプライン**（`auto_dpo.py`）:
DINOv2特徴量を使った完全自動の反復DPOパイプラインも構築：
1. 現在のモデルから1万枚生成
2. 各画像をFFHQ特徴量へのK-NNコサイン類似度でスコアリング（DINOv2 ViT-B/14）
3. スコア上位30% vs 下位30%で5000ペアを構成
4. DPO学習
5. 更新したモデルを参照モデルとして繰り返し

**ハイパーパラメータの感度**:
| β | lr | 結果 |
|------|----|--------|
| 5000 | 1e-6 | 保守的すぎ、目に見える変化なし |
| 1000 | 1e-6 | 良いバランス、微妙だが計測可能な改善 |
| 100 | 1e-6 | 崩壊（色が壊れ、顔が歪む） |
| 5000 | 1e-7 | 自動5000ペアで安定 |

### Stage 3: マルチタスクセグメンテーション

セマンティックセグメンテーションを**条件付き画像生成タスク**として定式化：顔画像を条件として、デノイズ過程を通じて対応するRGBセグメンテーションマスクを生成する。

- **データ**: CelebAMask-HQ、利用可能なアノテーションの**10%（104枚）のみ使用**
- **15クラス**: 背景、肌、鼻、左右の目、左右の眉、左右の耳、口、上唇/下唇、髪、首
- **学習**: 5万ステップ、seg_ratio=0.1（10%のステップがセグメンテーション、90%が顔生成）
- **条件付け**: 画像パッチは条件トークンとして機能（ノイズなし）、マスクパッチがデノイズされる。空間的アラインメントのため同一のRoPE位置を共有

**なぜ104枚で十分なのか**: モデルは53万ステップの生成学習で顔の構造を既に理解している。セグメンテーションで必要なのは、顔の解剖学ではなく、カラーコーディングのマッピングを学ぶことだけ。

**セグメンテーションDPO**: 各100件の人手マスク嗜好ペアによるDPOを2ラウンド実施し、セグメンテーション品質をさらに改善。

**定量評価**（200枚の未知画像）:

| 指標 | 値 |
|------|------|
| **Pixel Accuracy** | **78.8%** |
| **mIoU** | **43.2%** |

クラス別IoU:

| クラス | IoU | クラス | IoU |
|--------|-----|--------|-----|
| 背景 | 71.2% | 肌 | 73.7% |
| 鼻 | 80.0% | 髪 | 58.7% |
| 左目 | 62.0% | 右目 | 61.5% |
| 下唇 | 64.7% | 上唇 | 43.6% |
| 首 | 37.5% | 右眉 | 25.0% |
| 右耳 | 18.3% | 左眉 | 8.1% |
| 左耳 | 0.0% | 口 | 0.0% |

大きな領域（肌、髪、背景、鼻）は高精度。小さなパーツ（眉、耳、口の内部）は104枚の学習データでは困難。全データセット（1042枚）を使用すれば大幅な改善が見込まれる。

## 暗記分析

モデルが訓練データを暗記せず、新規の顔を生成していることを検証：

<p align="center">
<img src="assets/memorization_check_100.png" width="600">
</p>

最近傍訓練画像との平均コサイン類似度: 0.937（256×256ピクセル空間）。目視検査により、全ての生成画像が訓練データとは異なる人物であることを確認。なお、ピクセル空間のコサイン類似度はアラインされた顔データセットでは信頼性の低い暗記指標である — 正面向きの顔で肌色が類似していれば、人物が異なっていても自然と高い類似度が出る。このため、視覚的な最近傍検査で新規性を確認している。

## 制限事項

- 生成品質の定量評価（FID/KID）は未計算。結果は主に定性的なもの。
- セグメンテーションはアラインされた正面顔画像（CelebAMask-HQドメイン）に限定。任意のポーズや顔以外の画像への汎化は想定していない。
- 小さな顔パーツ（眉、耳、口の内部）は限られた学習セット（104枚）のためIoUが低い。
- モデルは無条件かつドメイン特化型（顔のみ）であり、汎用的なtext-to-imageモデルではない。
- DPOの改善は現在のスケールでは微妙であり、より大きな効果にはより多くの嗜好データやより大きなモデルが必要な可能性がある。

## 使い方

### 必要なライブラリ

```bash
pip install torch torchvision Pillow numpy tensorboard flask
# オプション: bitsandbytes (8bit optimizer), liger-kernel (fused RMSNorm)
```

### 事前学習

```bash
python train.py \
    --data_dir ./images256_ffhq_celebahq \
    --batch_size 128 --lr 1e-4 \
    --compile --max_autotune --optim_8bit --liger --preload \
    --bottleneck_dim 128 --img_size 512 --patch_size 32 \
    --total_steps 400000
```

### DPOファインチューニング

```bash
# 1. 評価用画像を生成
python generate_for_eval.py --ckpt runs/.../ckpt_final.pt --n 500

# 2. Web UIで人手評価
python evaluate_web.py --img_dir dpo_data/generated --out dpo_data/pairs.json --n 500
# http://localhost:8501 を開き、好みの画像をクリックまたは矢印キーで選択

# 3. DPO学習
python train_dpo.py \
    --ckpt runs/.../ckpt_final.pt \
    --pairs dpo_data/pairs.json \
    --img_dir dpo_data/generated \
    --beta 1000 --lr 1e-6 --epochs 50
```

### 自動DPO（DINOベース）

```bash
# FFHQ特徴量の事前計算（初回のみ、約10分）
python dino_scorer.py --precompute --data_dir ./images256_ffhq_celebahq

# 反復自動DPOの実行
python auto_dpo.py \
    --base_ckpt runs/.../ckpt_final.pt \
    --n_rounds 5 --n_generate 10000 --n_pairs 5000 --beta 5000
```

### マルチタスクセグメンテーション

```bash
python train_multitask.py \
    --ckpt runs/.../dpo_epoch_0030.pt \
    --face_dir ./images256_ffhq_celebahq \
    --seg_dir ./celebamask_hq_15class \
    --seg_fraction 0.1 --seg_ratio 0.1 \
    --batch_size 64 --lr 1e-5 --total_steps 50000 --compile
```

### デノイズ動画の生成

```bash
python make_denoise_video.py --ckpt runs/.../ckpt_final.pt --out denoise.mp4
```

## プロジェクト構成

```
model.py                 # PatchDiffusionDiTアーキテクチャ
train.py                 # 事前学習（Flow Matching + Patch Diffusion）
train_dpo.py             # Flow Matching向けDiffusion-DPO
train_multitask.py       # マルチタスク学習（顔生成 + セグメンテーション）
generate_for_eval.py     # 固定シードでのバッチ画像生成
evaluate_web.py          # DPO用ブラウザベースA/B比較UI
evaluate_seg_web.py      # セグメンテーション用ブラウザベース比較UI
generate_seg_pairs.py    # セグメンテーションDPO用マスクペア生成
dino_scorer.py           # DINOv2ベースの自動画像品質スコアリング
auto_dpo.py              # 反復自動DPOパイプライン
make_denoise_video.py    # デノイズ過程の可視化
assets/                  # 画像、グラフ、学習曲線
```

## 参考文献

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al., 2022)
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) (Peebles & Xie, 2022)
- [Patch Diffusion](https://arxiv.org/abs/2304.12526) (Wang et al., 2023)
- [Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908) (Wallace et al., 2023)
- [Image Generators are Generalist Vision Learners](https://arxiv.org/abs/2604.20329) (Gabeur et al., 2026)
- [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) (Lee et al., 2020)

## ライセンス

MIT
