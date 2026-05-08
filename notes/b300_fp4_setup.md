# B300 FP4/FP8 Training セットアップ手順

## 前提
- GPU: NVIDIA B300 SXM6 AC (275GB VRAM, CC 10.3, CUDA 13.0 driver)
- OS: Ubuntu 24.04
- Python: 3.12
- ディスク空き: 15GB以上（CUDA toolkit ~4GB + TEビルド ~5GB + pip ~5GB）
- CPU RAM: 32GB以上（TEビルド時ptxasが最大7GB×複数並列）
- ネットワーク: pipダウンロード用

## 注意事項
- **スポットインスタンス注意**: TEビルドに15分かかる。途中で切断されたら最初からやり直し
- **TE gitバージョン**: `--depth 1`で最新を取得。もし将来ビルドが壊れたら特定コミット指定:
  `git clone --depth 1 --branch v2.16.0 https://github.com/NVIDIA/TransformerEngine.git /tmp/te`
  (2026-05-05時点のmainはv2.16.0.dev0+528f16c)
- **transformer_engine_cu13をpip installしないこと**: ソースビルドと競合してversion mismatchエラー
- **LD_LIBRARY_PATHは毎回設定が必要**: .bashrcに追加推奨:
  `echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH' >> ~/.bashrc`

## 確認済み (2026-05-05)
- TE FP4 (NVFP4BlockScaling) forward+backward: batch=256 OK
- TE FP8 (DelayedScaling): OK
- BF16 + compile: 5,485 img/s (batch=512)

---

## Step 0: 前提確認

```bash
# GPU確認
nvidia-smi | head -15
# python3確認
python3 --version
# pip動作確認 (--break-system-packages が必要な環境かチェック)
python3 -m pip --version
```

## Step 1: 基本パッケージ

```bash
# 必ずこの順序で実行（torchが先、torchaoが後）
apt-get install -y wget g++ gcc git

python3 -m pip install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install --break-system-packages torchao bitsandbytes pybind11 cmake ninja

# 確認
python3 -c "import torch; print(f'PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name()}, SM: {torch.cuda.get_device_capability()}')"
```

## Step 2: CUDA 13.1 Toolkit (nvcc + cuBLAS 13.2)

```bash
# NVIDIA apt repo追加
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update -qq

# CUDA 13.1 toolkit (nvcc含む) + cuBLAS 13.1 (v13.2.2.2)
apt-get install -y cuda-toolkit-13-1 libcublas-13-1 cuda-cudart-13-1
```

## Step 3: CUDA 13 ライブラリをTEが見つけられる場所に配置

```bash
# TEはnvidia/cu13/lib/ を探す
mkdir -p /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
cp /usr/local/cuda-13.1/lib64/libcudart.so.13* /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/
cp /usr/local/cuda-13.1/lib64/libcublas.so.13* /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/
cp /usr/local/cuda-13.1/lib64/libcublasLt.so.13* /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/
```

## Step 4: Transformer Engine ソースビルド

```bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export CPATH=/usr/local/cuda-13.1/include:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/include:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LIBRARY_PATH
export CUDACXX=/usr/local/cuda-13.1/bin/nvcc
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS='100'

git clone --depth 1 https://github.com/NVIDIA/TransformerEngine.git /tmp/te
cd /tmp/te && python3 -m pip install --break-system-packages . --no-build-isolation --no-cache-dir
```

注意: ビルドに10-15分かかる（gelu.cu等のsm_100カーネルが重い）

## Step 5: Python 3.12 torch.compile バグパッチ

```bash
# CSE typing bug (PyTorch 2.11/2.12 + Python 3.12.3)
sed -i 's/CSE\[Any\]/CSE[Any, Any]/g' /usr/local/lib/python3.12/dist-packages/torch/_inductor/codegen/cutedsl/cutedsl_kernel.py
```

## Step 6: cuDNN SDPA 無効化 (B300 CC 10.3で非対応)

コード内で:
```python
torch.backends.cuda.enable_cudnn_sdp(False)
```

## Step 7: TE sanity check パッチ (ソースビルド時のみ)

```bash
# ソースビルドではPyPIメタデータがないのでチェックをスキップ
# パスはインストール方法で変わる:
#   通常install: /usr/local/lib/python3.12/dist-packages/transformer_engine/common/__init__.py
#   editable(-e): /tmp/te/transformer_engine/common/__init__.py
# 実際のパスを確認:
TE_INIT=$(python3 -c "import transformer_engine.common; import os; print(os.path.join(os.path.dirname(transformer_engine.common.__file__), '__init__.py'))")
echo "TE init: $TE_INIT"
sed -i 's/assert te_installed_via_pypi/# assert te_installed_via_pypi/' "$TE_INIT"
sed -i 's/assert version("transformer-engine") == te_core_version/# assert version("transformer-engine") == te_core_version/' "$TE_INIT"
```

---

## 動作確認

```python
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

torch.backends.cuda.enable_cudnn_sdp(False)
device = 'cuda'

# FP4
fp4_recipe = recipe.NVFP4BlockScaling()
te_lin = te.Linear(768, 2304, bias=False, params_dtype=torch.bfloat16).to(device)
x = torch.randn(256, 768, device=device, dtype=torch.bfloat16, requires_grad=True)
with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
    y = te_lin(x)
y.sum().backward()
print(f'FP4 OK: {y.shape}')

# FP8
fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID, amax_history_len=16)
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    y = te_lin(x)
y.sum().backward()
print(f'FP8 OK: {y.shape}')
```

---

## ベンチマーク実行

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
python3 bench_b300.py
```

---

## トラブルシューティング

| エラー | 原因 | 解決 |
|--------|------|------|
| `cudart shared object not found` | nvidia/cu13/lib/ にlibcudart.so.13がない | Step 3を実行 |
| `cublasLtGroupedMatrixLayoutInit_internal` | cuBLAS 13.0が古い | libcublas-13-1をインストール |
| `cudnn.h: No such file or directory` | CPATHにcudnn includeがない | Step 4のCPATH設定 |
| `nvcc fatal: Unknown option` | cmake 4.3とnvcc 13.0の互換性 | cuda-toolkit-13-1をインストール(nvcc 13.1) |
| `CSE[Any]` TypeError | Python 3.12 typing bug | Step 5のパッチ |
| `cuDNN Frontend error: No valid execution plans` | B300でcuDNN SDPAが未対応 | Step 6の無効化 |
| `PyPI package assertion` | ソースビルドでメタデータなし | Step 7のパッチ |
| `sm_100a generated for non family-specific and family-specific` | CMakeLists.txt にcompute_100fが重複追加 | パッチせずTE本来のCMakeに任せる |

---

## 実測値 (2026-05-05)

```
BF16 (no compile):   4,097 img/s (batch=512)
BF16 + compile:      5,485 img/s (batch=512)
TE FP8:              動作確認済み（速度未計測 - スポット切断）
TE FP4:              動作確認済み（速度未計測 - スポット切断）
torchao FP8:         BF16より遅い（オーバーヘッド大、使わないこと）

参考: RTX 6000 Ada = 122 img/s → B300 BF16は45倍速い
```

---

## B200での差異

B200 (CC 10.0) の場合:
- `NVTE_CUDA_ARCHS='100'` は同じ
- sm_100a が直接動く（compute_100f不要）
- その他の手順は全て同じ
