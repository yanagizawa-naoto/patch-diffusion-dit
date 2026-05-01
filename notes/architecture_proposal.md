# アーキテクチャ提案: Patch Diffusion × JiT × MMDiT ハイブリッド

## タスク
FFHQ 512×512 無条件画像生成

## 設計方針

### JiTから採用
- ピクセル空間で直接学習（VAEなし）
- ボトルネックパッチ埋め込み（高次元パッチの低ランク圧縮）
- 大パッチサイズでトークン数を抑制（patch_size=32 → 16×16=256トークン）
- x-prediction + v-loss（モデルはクリーン画像を予測、損失は速度空間で計算）
- 2D RoPE, RMSNorm, SwiGLU等のモダンTransformer設計

### MMDiTから採用
- 生成位置の注入: RoPEの位置パラメータでパッチ位置を表現
  → Patch Diffusionの座標条件付けをアーキテクチャ変更なしで実現
- adaLN-Zero条件付け（DiT由来、timestep注入用）

### Patch Diffusionから採用
- 訓練時にランダムクロップしたパッチで学習（データ拡張効果 + 高速化）
- パッチサイズ確率: full(512)=50%, half(256)=30%, quarter(128)=20%
- バッチ乗数: 小パッチ時にバッチサイズ増加でGPU利用率維持
- 推論時はフル座標でフル画像生成

## フロー定式化
- 順方向: z_t = t * x_0 + (1-t) * ε (JiT規約)
- 速度目標: v = x_0 - ε
- モデル出力: x_pred
- 速度導出: v_pred = (x_pred - z_t) / (1-t)
- 損失: ||v - v_pred||² = 1/(1-t)² * ||x_0 - x_pred||²
- タイムステップ: logit-normal sampling (m=0.0, s=1.0)

## モデル構成
- patch_size = 32
- depth = 12
- hidden_size = 768
- num_heads = 12
- head_dim = 64
- bottleneck_dim = 128
- FFN: SwiGLU (intermediate ≈ 2048)
- 概算パラメータ: ~137M
