# クラウドGPUセットアップ手順

## SSH接続
```bash
ssh root@<IP_ADDRESS>
```

## 1. パッケージインストール
```bash
apt-get update -qq && apt-get install -y -qq tmux
pip install --break-system-packages --ignore-installed typing-extensions \
  torch torchvision tensorboard huggingface_hub hf_transfer
```

## 2. カラーターミナル設定
```bash
echo 'export PS1="\[\e[1;32m\]\u@gpu\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]\$ "' >> ~/.bashrc
echo 'alias ls="ls --color=auto"' >> ~/.bashrc
source ~/.bashrc
```

## 3. HuggingFaceログイン
```bash
mkdir -p ~/.cache/huggingface
echo '<HF_TOKEN>' > ~/.cache/huggingface/token
```
※ トークンはローカルの `cat ~/.cache/huggingface/token` で確認

## 4. データダウンロード
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Naoto-ipu/ffhq-celebahq-256 --repo-type dataset --local-dir /workspace/data
cd /workspace/data && tar xf images256_ffhq_celebahq.tar
```

## 5. コード転送
GitHubがprivateなのでscpで転送:
```bash
# ローカルから実行
scp model.py train.py encode_latents.py root@<IP>:/workspace/
```

## 6. 学習開始
```bash
tmux new-session -d -s train
tmux send-keys -t train 'cd /workspace && python3 train.py \
  --data_dir /workspace/data/images256_ffhq_celebahq \
  --img_size 256 --batch_size 32 --lr 1e-4 \
  --bottleneck_dim 128 --total_steps 100000 --save_every 5000' Enter
```

### チェックポイントからresume
```bash
python3 train.py --resume /workspace/runs/<RUN_DIR>/ckpt_XXXXX.pt \
  --data_dir /workspace/data/images256_ffhq_celebahq \
  --img_size 256 --batch_size 32 --lr 1e-4 \
  --bottleneck_dim 128 --total_steps 400000 --save_every 5000
```

## 7. ローカルへの同期 (ローカル側で実行)
```bash
# sync_b300.sh のIPを変更して使用
bash sync_b300.sh
```

## 注意点
- `pip install` で `typing-extensions` の競合が出る場合は `--ignore-installed` を付ける
- `huggingface-cli` は非推奨、`hf` コマンドを使用
- Startup Scriptは環境によって動かないことがある → 手動セットアップが確実
- スポットインスタンスはチェックポイント転送中に停止する可能性あり
  → `--save_every 5000` で頻繁に保存
  → 最後のチェックポイントは破損リスクあり、1つ前を使う
- チェックポイントは1個約2GB (130Mモデル)

## GPU別の実測スループット (batch=32, depth=12, 130M)
```
RTX 6000 Ada:  ~90 img/s
B300 SXM6:     ~643 img/s (単独), ~472 img/s (2実験並列)
GH200 PCIe:    未計測
```

## コスト比較 (Spheron, 2026年5月時点)
```
B300 SXM6 spot:    $3.00/hr
GH200 PCIe:        $1.97/hr
H100 PCIe:         $2.01/hr
A100 80G SXM spot: $0.61/hr
```
