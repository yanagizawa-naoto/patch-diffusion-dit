# FP8トレーニング検証結果

## 環境
- GPU: RTX 6000 Ada Generation (48GB, sm89)
- PyTorch: 2.12.0.dev20260408+cu128 (nightly)
- torchao: 0.17.0
- モデル: PatchDiffusionDiT, depth=12, hidden=768, bottleneck=128 (130M params)
- データ: 256×256, Patch Diffusion (crop 256/128)

## 動作する構成

### FP8 + compile(dynamic=True) + warmup
```python
# 1. 0-init層をFP8から除外
zero_layers = {}
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and torch.all(module.weight == 0):
        zero_layers[name] = deepcopy(module)
convert_to_float8_training(model, config=Float8LinearConfig())
for name, orig in zero_layers.items():
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], orig.to(device))

# 2. compile (dynamic=Trueが必須)
model = torch.compile(model, dynamic=True)

# 3. warmup (nan-skipping付き、5-10ステップ)
for i in range(10):
    loss = compute_v_loss(model, ...)
    if not torch.isnan(loss):
        loss.backward()
        optimizer.step()
```

### ベンチマーク結果 (RTX 6000 Ada, depth=12, batch=128)
```
BF16:                     408 ms/step,  314 img/s (1.00x)
BF16 + compile:           176 ms/step,  726 img/s (2.31x)
FP8 + compile(dynamic):   163 ms/step,  786 img/s (2.50x)
```

## nanの原因と対策

### 原因1: adaLN-Zeroの0初期化
- adaLN/final_adaLN/final_proj/t_embedの重みが全て0
- FP8スケーリングで scale = max(|w|) / 448 = 0 / 448 = 0 → 0除算 → nan
- **対策**: 0初期化された層をFP8から除外

### 原因2: torch.compile初回コンパイル時の不安定
- 複数のcropサイズで初回コンパイルされる際にnanが出ることがある
- **対策**: warmupフェーズでnan-skipping付き5-10ステップを実行

### 原因3: dynamic=None + recompile_limit
- dynamic=Noneだとcropサイズ変更のたびにrecompile
- 8回超えるとdynamo fallback → FP8 state破壊 → nan
- **対策**: dynamic=Trueを使用

### 原因4: fast_accum + dynamic=True の特定サイズ
- batch=128, crop=128 (matmul 8192×768) で約50%のnanが出る場合あり
- warmup後は安定する傾向
- **対策**: warmupで両cropサイズを事前コンパイル

## FP8が効かない理由 (RTX 6000 Ada, batch=128)

### matmulは全体の~14%しかない
```
depth=12, batch=128 の1ステップ:
  Total: 176ms (compile後)
  matmul: ~25ms (14%)
  非matmul: ~151ms (86%) ← RoPE, RMSNorm, softmax, adaLN, メモリ等

FP8でmatmulが半分になっても:
  25ms → 12.5ms = 12.5ms節約
  全体: 176ms → 163ms = 8%改善のみ
```

## 最適な高速化の組み合わせ (RTX 6000 Ada, depth=12, batch=128)

### ベンチマーク結果
```
BF16:                          346 ms/step,  370 img/s (1.00x)
Liger RMSNorm (no compile):   248 ms/step,  516 img/s (1.39x)
compile(dynamic=True):         170 ms/step,  754 img/s (2.04x)
compile + 8bit Adam:           170 ms/step,  753 img/s (2.03x)
compile(max-autotune):         161 ms/step,  796 img/s (2.15x)
FP8 + compile:                 163 ms/step,  786 img/s (2.12x)
max-autotune + 8bit Adam:      150 ms/step,  853 img/s (2.31x) ← 最速
```

### なぜmax-autotune + 8bit Adamが最速か
- max-autotune: Tritonカーネルの最適パラメータ(BLOCK_M/N/K等)を自動探索
- 8bit Adam: optimizer stateの転送量を4分の1に削減 → メモリ帯域ボトルネック緩和
- FP8: matmulは14%しかないのでオーバーヘッドと相殺、compile単体より遅い

### 8bit Adamの品質検証
100ステップでBF16 AdamWとの差: 最大0.01 (許容範囲)

### プロファイリング結果 (compile後)
```
GEMM (matmul):         44% ← FP8の対象だが全体の半分以下
Triton融合カーネル:     17% ← compile済み、追加最適化困難
Flash Attention:        4.5%
メモリ転送/その他:      34.5% ← 8bit Adamで一部削減
```

### Liger-Kernelの状況
- RMSNorm 7x高速化の実績あり
- ただしtorch.compileと併用不可 (SubgraphTracerエラー)
- compileなしだと1.39x (compileの2.04xに負ける)
- 将来torch.compile互換になれば有望

### Fused Optimizer-in-Backward
- PyTorch公式チュートリアルに記載
- hookのreturn typeエラーで動作せず
- 将来のPyTorchバージョンで修正される可能性

## 今後FP8で圧勝するための条件
1. 大モデル (depth=24+, hidden=1024+): matmul比率が上がる
2. 大バッチ (256+): matmulが支配的になる
3. H100/B300: FP8 Tensor Coreがより効率的
4. FP4 (Blackwell sm100+): さらに2倍高速化
5. torch.compileの最適化改善: 非matmul部分のoverhead削減

## 動作しない構成

| 構成 | 結果 | 原因 |
|------|------|------|
| FP8+compile, 0-init層FP8化 | nan | 0除算 |
| FP8+compile, dynamic=None | nan | recompile_limit超過 |
| FP8+compile, warmupなし | nan確率高 | compile初回不安定 |
| FP8+compile, batch_mul有効 | nan | バッチサイズ動的変更 |
| FP8+compile, dynamic=True + fast_accum=False | nan | torchao/compileバグ |
| FP8+compile, rowwise scaling | エラー | axiswise+compileは未サポート |

## 関連Issue
- pytorch/pytorch #150859: RMSNorm + compile + Float8 NaN (closed, 2.12.0で修正済み)
- pytorch/pytorch #154028: FP8 E4M3 overflow → NaN (software clamping未実装)
- pytorch/ao #561: Float8 amax/scale buffer精度問題
- arXiv 2409.12517: SwiGLUがFP8で外れ値を増幅
