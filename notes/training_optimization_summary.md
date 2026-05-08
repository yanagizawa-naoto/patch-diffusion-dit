# 学習高速化 完全まとめ

## 環境
- GPU: RTX 6000 Ada Generation (48GB GDDR6, 960 GB/s, BF16 364 TFLOPS)
- モデル: PatchDiffusionDiT, depth=12, hidden=768, bottleneck=128 (130M params)
- データ: 256×256, FFHQ+CelebA-HQ 100K枚, Patch Diffusion (crop 256/128)
- venv: `/home/naoto/venv_fp8/` (PyTorch 2.12.0 nightly + torchao + bitsandbytes + liger-kernel)

---

## 最適設定（確定）

```bash
/home/naoto/venv_fp8/bin/python train.py \
    --batch_size 64 --lr 2e-4 \
    --compile --max_autotune --optim_8bit --liger --preload \
    --bottleneck_dim 128 --depth 12 --hidden_size 768 --num_heads 12 \
    --img_size 256 --data_dir ./images256_ffhq_celebahq \
    --resume runs/patch_dit_ffhq512_20260504_121120/ckpt_0120000.pt
```

**実測性能: 522ms/step, 122 img/s（元の52 img/sから2.35倍）**

---

## 高速化の内訳

| 最適化 | 効果 | 説明 |
|--------|------|------|
| torch.compile + max-autotune | 最大 (~2.3x) | kernel fusion, Triton autotune |
| batch 32→64 + lr=2e-4 | 2x img/s | linear lr scaling |
| 8-bit Adam (bitsandbytes) | 微小 | optimizer stateメモリ帯域削減 |
| Liger RMSNorm | ~10% | compile graph breakだが単体カーネルが速い |
| RAM preload (--preload) | 15% | 70K画像をuint8でRAMに全読み込み、DataLoaderバイパス |

---

## ボトルネック分析（実測）

### プロファイリング結果 (crop=512, batch=64)
```
matmul (GEMM):       306ms  (52.7%)  ← ハードウェア限界
RMSNorm (Liger):     131ms  (22.5%)  ← 最速実装済み
triton_fused:         78ms  (13.4%)  ← compile fusion済み
memory_ops (RoPE等):  54ms  ( 9.2%)  ← compile済み
optimizer:             5ms  ( 0.8%)
other:                 8ms  ( 1.3%)
```

### GPU利用率
```
理論演算量:   23.2 TFLOPS (fwd+bwd)
理論最小時間:  63.7ms
実測時間:     604ms
GPU演算利用率: 10.5% ← 残り89.5%はメモリ帯域待ち
```

### 根本的制約
RTX 6000 Ada の GDDR6 (960 GB/s) がボトルネック。Tensor Core (364 TFLOPS) は余っているが、データ供給が追いつかない。

---

## 試して効果がなかったもの

### FP8 (torchao Float8Linear)
- **結果:** BF16と同速 (617ms vs 619ms)
- **原因:** FP8はmatmul演算を2倍にするが、ボトルネックはメモリ帯域。演算が余っているのでFP8で演算力を上げても意味がない
- **RTX 6000 Adaの演算/帯域比:** BF16で379 FLOPS/byte, FP8で759 FLOPS/byte → メモリ飢餓

### FP8 (NVIDIA Transformer Engine)
- **結果:** BF16と同速または微減
- **原因:** 同上。TE FP8 fused Linear (LayerNormLinear含む) でも帯域制限で速度変わらず

### COAT (NVlabs, FP8 activation compression)
- **結果:** compile非互換で使えない
- **原因:** COATのTritonカーネルがtorch.compileのTritonと衝突。compileなしだと1.42xだがcompile(2.3x)に負ける

### 手動FP8 activation保存 (autograd.Function)
- **結果:** 1.43x遅くなる
- **原因:** Pythonレベルのquantize/dequantizeオーバーヘッドが帯域削減効果を上回る

### Gradient Checkpointing
- **結果:** 69%遅くなる (881ms vs 522ms)
- **原因:** 再計算コストが帯域削減効果を大幅に上回る。batch=128は可能になるが1819msで遅い

### F.rms_norm + compile (Ligerなし)
- **結果:** 9%遅い (570ms vs 522ms)
- **原因:** Ligerの@torch._dynamo.disableがgraph breakを起こすが、Ligerカーネル自体が十分速く、compileのグラフ範囲が小さい方がむしろ効率的

### Static bucket compile (dynamic=False)
- **結果:** 0%改善 (dynamic=Trueと同速)
- **原因:** max-autotuneがdynamic=True下で既に各shape用の最適カーネルを生成

### SwiGLU w1+w3結合
- **結果:** 8%悪化
- **原因:** compile最適化パターンが崩れる

### v-loss代数簡約
- **結果:** 上記に含む（悪化）
- **原因:** compileが元の構造で既に最適なカーネルを生成

### RoPE inv_freq キャッシュ
- **結果:** CUDA Graphsと衝突してクラッシュ
- **原因:** グローバル辞書のテンソルがCUDA Graphに上書きされる

---

## FP4 Training 調査結果

### NVFP4 (NVIDIA Transformer Engine)

#### 対応GPU
| GPU | SM | 共有メモリ/block | FP4 batch=32 | FP4 batch=64+ |
|-----|-----|-----------------|--|--|
| RTX 5060 Ti | sm_120 | 99 KB | ✓ (SW fallback) | ✗ smem不足 |
| RTX 5090 | sm_120 | 99 KB | ✓ (SW fallback) | ✗ smem不足 |
| RTX PRO 6000 | sm_120 | 99 KB | ✓ (推定) | ✗ (推定) |
| B200/GB200 | sm_100 | 228 KB | ✓ | ✓ |
| B300/GB300 | sm_100 | 228 KB | ✓ | ✓ |

#### 根本原因
TEのNVFP4 Hadamard Transformカーネルが要求する共有メモリサイズがsm_120 (99KB) を超える。sm_100 (228KB) なら収まる。batch次元でsmem使用量が決まり、Dは無関係。

#### NVIDIAのFP4論文 (arxiv 2509.25149)
- GPU: GB200/GB300のみ。Consumer GPUへの言及なし
- 手法: 全GEMM入力をNVFP4化、Hadamard transform、2Dブロックスケーリング、stochastic rounding
- 精度: FP8比1%未満の相対誤差
- ソフトウェア: Transformer Engine

#### セットアップ手順 (B200/B300用、将来参照)
```bash
# CUDA 13.1+環境で
pip install torch --index-url https://download.pytorch.org/whl/cu128
# TE をソースビルド (compute_120f or compute_100a)
git clone --depth 1 https://github.com/NVIDIA/TransformerEngine.git /tmp/te
# CMakeLists.txt の project() 後に追加:
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_100a,code=sm_100a")
export NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS='100'
export CPATH=.../nvidia/nccl/include:$CPATH
export LIBRARY_PATH=.../nvidia/nccl/lib:$LIBRARY_PATH
cd /tmp/te && pip install . --no-build-isolation
```

#### train.py FP4対応 (実装済み)
```bash
python train.py --fp4 --compile --max_autotune --optim_8bit ...
```
`--fp4` フラグで全nn.LinearをTE Linearに置換、NVFP4BlockScaling recipeで学習。

### MXFP4 (torchao)
- **結果:** `NotImplementedError: MXFP4 scaling only supported in CUDA for B200/B300`
- RTX 50xxではハードウェア非対応

---

## GPU別の推定性能

スケーリングモデル: `performance = (BW/960)^0.7 × (TFLOPS/364)^0.3`

| GPU | BW (GB/s) | BF16 img/s | FP4 img/s | 備考 |
|-----|-----------|------------|-----------|------|
| RTX 6000 Ada | 960 | **122** (実測) | - | 現在使用中 |
| RTX 5090 | 1,792 | ~163 | ✗ smem制限 | FP4使えない |
| RTX PRO 6000 BW | 1,792 | ~172 | ✗ smem制限 | 同上 |
| H100 SXM | 3,350 | ~357 | - | FP4非対応 (Hopper) |
| B200 | 8,000 | ~729 | ~1,100 | FP4完全対応 |

---

## コスト分析

### 200Kステップの学習
| 構成 | 所要時間 | 費用 |
|------|---------|------|
| RTX 6000 Ada (現状) | 29.0時間 | 電気代のみ |
| H100 cloud (Vast.ai $1.4/hr) | ~10時間 | ~2,000円 |
| B200 cloud (推定 $4/hr) | ~5時間 | ~3,000円 |

### RTX 5090 x2 DDP (買い切り)
- 初期投資: ~155万円
- img/s: ~296 (BF16 DDP)
- 損益分岐: 14ヶ月 (24h稼働) or 43ヶ月 (8h/日)

---

## データローディング最適化

### 問題
元のパイプライン: 70K PNGをディスクから毎step読み込み → DataLoader経由 → GPU転送
プロファイリングでステップ時間の40%以上がデータI/O

### 解決策 (`--preload`)
```python
class FFHQDatasetRAM(Dataset):
    # 起動時に70K画像をuint8 [N,3,H,W] でRAMに全読み込み (55GB)
    # get_batch(): ランダムインデックスで直接バッチ取得、GPU上でnormalize
```

### 効果
- データ読み込み: 950ms → 16ms (60倍高速化)
- ステップ時間: 617ms → 522ms (15%改善)
- RAM使用: +55GB (256GB中)

---

## B300 実測結果 (2026-05-05)

### BF16 vs TE FP8 vs TE FP4
```
D=768 (130M):   BF16 3,702 img/s, FP8 1.09x, FP4 0.92x (FP4効かない)
D=2048 (2.5B):  BF16 362 img/s,   FP8 1.34x, FP4 1.42x (FP4がFP8を超える)
D=3072 (12B):   BF16 122 img/s,   FP8 1.42x, FP4 1.62x (FP4圧勝)
```

### FP4が効く条件
- D(hidden_size) ≥ 2048 が必須
- D=768ではbatchをいくら増やしてもFP4 < BF16
- D ≥ 2048 かつ 大batch で FP4 > FP8

### B300 vs RTX 6000 Ada
```
RTX 6000 Ada BF16+compile: 122 img/s
B300 BF16+compile:          4,894 img/s (40倍)
B300 BF16 (no compile):     3,702 img/s (30倍)
```

---

## 結論

1. **RTX 6000 Adaでの限界は522ms/step (122 img/s)** — ソフトウェア最適化は尽くした
2. **ボトルネックはGDDR6の帯域幅 (960 GB/s)** — GPU演算の10.5%しか使えていない
3. **FP4 trainingはB200/B300専用** — RTX 50xxは共有メモリ制限 (99KB vs 228KB) で不可
4. **FP4が効くにはD ≥ 2048が必要** — 現在のD=768では効果なし
5. **スケールアップ時はpatch=16, D=2048+が最効率** — 詳細は notes/future_scaling_plan.md
