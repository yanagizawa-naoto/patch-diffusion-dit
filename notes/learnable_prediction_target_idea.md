# 学習可能な予測対象 (Learnable Prediction Target) のアイデア

## 背景

拡散モデルの予測対象は手動で選択される:
- x-prediction: y = x₀
- ε-prediction: y = ε
- v-prediction: y = x₀ - ε

これらは全て y = a·x₀ + b·ε の線形結合であり、係数が異なるだけ。
v-predictionは定数係数(1,-1)で特殊に綺麗なケースだが、z_t = t·x₀ + (1-t)·ε
のように係数がtに依存すること自体は拡散モデルの枠組みとして自然。

## 提案

予測対象をデータ駆動で学習する:

```
y(t) = cos(θ(t))·x₀ + sin(θ(t))·ε
```

θ(t)は小さなMLPで学習:

```python
theta_net = MLP(input_dim=1, output_dim=1)  # t → θ

def get_target(t, x_0, eps):
    theta = theta_net(t)
    a = torch.cos(theta)
    b = torch.sin(theta)
    return a * x_0 + b * eps
```

cos/sinの正規化により a² + b² = 1 が保証され、自明解(a→0, b→0)を防ぐ。

## 既存の予測対象との関係

```
θ = 0:      y = x₀         (x-prediction)
θ = π/2:    y = ε           (ε-prediction)
θ = -π/4:   y = (x₀-ε)/√2  (v-prediction、スケール違い)
θ = f(t):   timestep依存の最適予測対象 (新規)
```

## 損失の重み付けとの関係

数学的には予測対象を変えることと損失の重みを変えることは等価:

```
||y - y_pred||² = (a - b·t/(1-t))² × ||x₀ - x_pred||²
```

しかし最適化のランドスケープ（勾配の流れ方）は異なるため、
学習のしやすさに差が出る可能性がある。
v-predictionがε-predictionより安定するのも同じ理由。

## Min-SNR + v-loss の場合

Min-SNR重み付きv-loss は以下の予測と等価:

```
y(t) = √min(SNR(t), γ) · v = √min(t²/(1-t)², γ) · (x₀ - ε)
```

つまり「SNRでスケールされた速度」を予測していることになる。

## MuLANとの違い

```
本提案:  同じノイズ、予測対象をtに応じて変える
MuLAN:   ノイズをピクセルごとに変える、予測対象は固定

→ 直交した改善軸。組み合わせも可能:
  y(t, i, j) = cos(θ(t, i, j))·x₀[i,j] + sin(θ(t, i, j))·ε[i,j]
  → ピクセル位置にも依存する空間適応型予測対象
```

## 期待される成果

1. 学習後のθ(t)カーブを分析 → 最適な予測対象のtimestep依存性を発見
2. v-predictionが本当に最適なのかをデータ駆動で検証
3. データセット/解像度/タスクごとの最適予測対象の違いを調査
4. 「等価なはずの損失でも学習ダイナミクスが異なる」ことの実証

## 既存研究の状況

2025年5月時点で、予測対象自体を学習可能にする論文は確認できず。
関連研究:
- EDM (Karras 2022): preconditioning係数を解析的に導出
- VDM (Kingma 2021): ノイズスケジュールを学習 (間接的に等価)
- Min-SNR (Hang 2023): 損失重みを設計 (明示的な予測対象学習ではない)
- MuLAN (Sahoo 2024): ピクセルごとのノイズを学習 (直交した軸)

## ステータス

未実装。論文化の可能性あり。
