# MMDIT

![MMDIT](assets/sample.png)

## Architecture

- **Autoencoder:** DC-AE f32c32 (32× compression, 32 latent channels)
- **Text Encoder:** T5-Large
- **Backbone:** MMDiT with Joint Attention
- **Implementation:** Custom implementation (no pretrained base model)
- **Parameters:** 4B

## Environment

- **Framework:** PyTorch (no custom CUDA kernels)
- **GPU:** NVIDIA RTX PRO 6000 (96GB VRAM)
- **Platform:** Google Colab

## Training

### Configuration

- **Batch Size:** Micro-batch 4 × Gradient Accumulation 16 = Effective Batch 64
- **Training Speed:** ~2.3 s / optimizer step
- **Max Resolution:** 1024 × 1024
- **Dataset:** 2M images
- **Objective:** Flow Matching (shift = 3.0)
- **Precision:** BF16
- **Optimizer:** AdamW
- **Learning Rate:** 1e-4
- **EMA:** Enabled

### Training Stages

| Stage | Description | Steps | Patch Size |
|:---:|---|---:|:---:|
| Stage 1 | Pre-training | 600K | 2 |
| Stage 2 | Fine-tuning | 11K | 1 |

**Stage 2 Notes:**
- Reset patch embedding weights.
- Reset output projection weights.

### Loss by Noise Level

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

## Generation Results

### Patch Size 2 + DC-AE f32c32

![Patch Size 2](assets/patch_size_2.png)

- **Image** 2k -> 50k -> 300k -> 450k-> 600k
- **Issue:** Facial features, particularly the eyes, tend to collapse or become distorted.
- **Observation:** Limited spatial detail makes accurate reconstruction of fine facial features challenging.

### Patch Size 1 + DC-AE f32c32

![Patch Size 1](assets/patch_size_1.png)

- **Image** 10k -> 80k -> 110k
- **Fine-tuning Result:** Fine-tuning with patch size 1 improved facial feature reconstruction.
- **Improvement:** Eye and facial distortions were partially resolved, with better preservation of fine details.