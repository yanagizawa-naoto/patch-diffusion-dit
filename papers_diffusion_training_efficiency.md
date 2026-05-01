# 大規模拡散モデルの訓練効率化に関する論文リスト

調査日: 2026-04-30

---

## 1. アーキテクチャ効率化

| # | 論文 | 著者 | 年 | 会議 | arXiv | 要点 |
|---|------|------|-----|------|-------|------|
| 1 | Scalable Diffusion Models with Transformers (DiT) | Peebles & Xie | 2023 | ICCV | 2212.09748 | U-NetをTransformerに置換。強いスケーリング特性。FID 2.27 (ImageNet 256) |
| 2 | All are Worth Words: A ViT Backbone for Diffusion Models (U-ViT) | Fan Bao et al. | 2023 | CVPR | 2209.12152 | 純粋なViTバックボーン、down/upsamplingが不要。FID 2.29 |
| 3 | PixArt-alpha: Fast Training of Diffusion Transformer for Photorealistic T2I Synthesis | Junsong Chen et al. | 2023 | ICLR'24 | 2310.00426 | SD v1.5の訓練コストの10.8%で同等品質（675 vs 6,250 A100 GPU days）。3段階分解訓練 |
| 4 | Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3) | Patrick Esser et al. (Stability AI) | 2024 | ICML | 2403.03206 | Dual-stream Transformer + Rectified Flow。450M〜8Bパラメータでスケーリング検証 |
| 5 | MaskDiT: Fast Training of Diffusion Models with Masked Transformers | Zheng et al. | 2023 | arXiv | 2306.09305 | 50%パッチマスキングで訓練コスト70%削減 |
| 6 | DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention | Zhu et al. | 2025 | CVPR | 2405.18428 | Gated Linear Attentionで線形計算量を実現 |

## 2. MoE（Mixture-of-Experts）スケーリング

| # | 論文 | 著者 | 年 | 会議 | arXiv | 要点 |
|---|------|------|-----|------|-------|------|
| 7 | EC-DiT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing | Haotian Sun et al. (Google) | 2024 | ICLR'25 | 2410.02098 | Expert-Choice Routingで97Bパラメータまで効率的にスケール |
| 8 | Efficient Training of Diffusion Mixture-of-Experts Models: A Practical Recipe | Yahui Liu et al. | 2025 | arXiv | 2512.01252 | DeepSeek型MoEの実践的設計指針 |

## 3. 訓練スケジュール・損失関数の最適化

| # | 論文 | 著者 | 年 | 会議 | arXiv | 要点 |
|---|------|------|-----|------|-------|------|
| 9 | Efficient Diffusion Training via Min-SNR Weighting Strategy | Tiankai Hang et al. | 2023 | ICCV | 2303.09556 | SNRベースの損失重み付けで3.4倍高速化、FID 2.06 |
| 10 | Analyzing and Improving the Training Dynamics of Diffusion Models (EDM2) | Tero Karras et al. (NVIDIA) | 2024 | CVPR Oral | 2312.02696 | 訓練ダイナミクスの体系的修正。活性化/重みの大きさを制御 |
| 11 | FasterDiT: Towards Faster DiT Training without Architecture Modification | Jingfeng Yao et al. | 2024 | arXiv | 2410.10356 | ノイズスケジュール最適化でDiTを7倍高速に収束 |
| 12 | A Closer Look at Time Steps is Worthy of Triple Speed-Up (SpeeD) | Kai Wang et al. (NUS) | 2024 | CVPR'25 | 2405.17403 | タイムステップの非対称サンプリングで3倍高速化 |
| 13 | Representation Alignment for Generation (REPA) | Sihyun Yu et al. | 2025 | ICLR Oral | 2410.06940 | DINOv2との表現アライメントで17.5倍高速収束、FID 1.42 |
| 14 | Adaptive Non-Uniform Timestep Sampling for Accelerating Diffusion Model Training | Myunsoo Kim et al. | 2025 | CVPR'25 | 2411.09998 | 勾配分散に基づく適応的タイムステップサンプリング |

## 4. ステップ数削減・蒸留

| # | 論文 | 著者 | 年 | 会議 | arXiv | 要点 |
|---|------|------|-----|------|-------|------|
| 15 | Consistency Models | Yang Song et al. (OpenAI) | 2023 | ICML | 2303.01469 | 1ステップ生成の基盤的フレームワーク |
| 16 | Improved Techniques for Training Consistency Models (iCT) | Yang Song & Prafulla Dhariwal | 2024 | ICLR Oral | - | Consistency Modelsを3.5〜4倍改善 |
| 17 | Latent Consistency Models (LCM) | Simian Luo et al. | 2023 | CVPR'24 | 2310.04378 | Latent空間でのConsistency蒸留、2-4ステップ生成 |
| 18 | LCM-LoRA: A Universal Stable-Diffusion Acceleration Module | Simian Luo et al. | 2023 | arXiv | 2311.05556 | LoRAベースの蒸留、プラグアンドプレイ加速 |
| 19 | Adversarial Diffusion Distillation (ADD / SDXL Turbo) | Axel Sauer et al. (Stability AI) | 2023 | ECCV'24 | 2311.17042 | 敵対的蒸留で1ステップリアルタイム生成を初めて実現 |
| 20 | SDXL-Lightning: Progressive Adversarial Diffusion Distillation | Shanchuan Lin et al. (ByteDance) | 2024 | arXiv | 2402.13929 | Progressive adversarial distillationで1-8ステップ1024px生成 |
| 21 | Improved Distribution Matching Distillation (DMD2) | Tianwei Yin et al. (MIT/Adobe) | 2024 | NeurIPS Oral | 2405.14867 | 1ステップFID 1.28 (ImageNet 64)、教師を超える性能 |
| 22 | SlimFlow: Training Smaller One-Step Diffusion Models with Rectified Flow | Yuanzhi Zhu et al. (ETH) | 2024 | ECCV | 2407.12718 | ステップ数とモデルサイズの同時圧縮 |

## 5. データ効率・スケーリング則

| # | 論文 | 著者 | 年 | 会議 | arXiv | 要点 |
|---|------|------|-----|------|-------|------|
| 23 | Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models | Zhendong Wang et al. | 2023 | NeurIPS | 2304.12526 | パッチ単位訓練で2倍以上高速、5000枚からでも訓練可能 |
| 24 | Data Curation via Joint Example Selection (JEST) | Siddharth Joshi et al. (DeepMind) | 2024 | NeurIPS | 2406.17711 | バッチ単位データキュレーションで13倍少ないイテレーション |
| 25 | Scaling Laws for Diffusion Transformers | Zhengyang Liang et al. | 2024 | arXiv | 2410.08184 | DiTのスケーリング則を定式化（Chinchilla的） |
| 26 | On the Scalability of Diffusion-based Text-to-Image Generation | Hao Li et al. (Amazon) | 2024 | CVPR | 2404.02883 | データ品質 > 量。45%小さいU-Netで同等性能 |
| 27 | Autoguided Online Data Curation for Diffusion Model Training | Valeria Pais et al. | 2025 | arXiv | 2509.15267 | Autoguidance + オンラインデータ選択 |

## 6. 分散訓練インフラ

| # | 論文 | 著者 | 年 | 会議 | arXiv | 要点 |
|---|------|------|-----|------|-------|------|
| 28 | DiffusionPipe: Training Large Diffusion Models with Efficient Pipelines | Ye Tian et al. (Amazon) | 2024 | MLSys | 2405.01248 | パイプラインバブルに非訓練部分を配置、1.41倍高速化 |
| 29 | DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models | Muyang Li et al. (MIT) | 2024 | CVPR Highlight | 2402.19481 | Displaced Patch Parallelismで8 A100で6.1倍高速化 |

## サーベイ論文

- **Efficient Diffusion Models: A Survey** (TMLR 2025, arXiv: 2502.06805)
  - GitHub: https://github.com/AIoT-MLSys-Lab/Efficient-Diffusion-Model-Survey
