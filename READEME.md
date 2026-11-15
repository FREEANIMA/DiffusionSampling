## MMDIT
![MMDIT](assets/sample.png)

## Architecture
- Autoencoder: DC-AE f32c32 (compression ratio 32 / 32 latent channels)
- Text encoder: T5 Large
- Backbone: MMDiT with joint attention (custom implementation, none basemodel)
- Paramters: 4B

## Environment
- Framework: PyTorch (no custom CUDA kernels)
- GPU: RTX PRO 6000 (96GB VRAM), Google Colab

## Training
- Hardware: 1x RTX PRO 6000 (96GB)
- Batch: micro-batch 4 × grad accumulation 16 = effective batch 64
- Speed: ~2.3 s / optimizer step
- Max resolution: 1024 × 1024
- Dataset: 2M images
- Objective: flow matching (shift 3.0)
- Precision: bf16
- Optimizer: AdamW, lr 1e-4, EMA OK
- Stage 1 (pre-training): 600K steps, patch size 2
- Stage 2 (fine-tuning): 11K steps, patch size 1 (Reset Patch embedding Weights, Out proj Weights)

## Loss by Noise Level
| Noise Level | Average Loss |
|:---:|---:|
| 1.0 – 0.9 | 4.1765 |
| 0.9 – 0.8 | 3.0710 |
| 0.8 – 0.7 | 2.4875 |
| 0.7 – 0.6 | 2.0503 |
| 0.6 – 0.5 | 1.8212 |
| 0.5 – 0.4 | 1.6026 |
| 0.4 – 0.3 | 1.5205 |
| 0.3 – 0.2 | 1.3906 |
| 0.2 – 0.1 | 1.2685 |

## Patch Size 2 + DC-AE f32c32
