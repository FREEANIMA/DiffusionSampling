# Diffusion MMDiT Text2img with DC-AE f32c32

[VIEW UPDATE](CHANGELOG.md)
[VIEW TRAINSTATE](/docs/training.md)
[VIEW ARCHITECTURE](/docs/architecture.md)
[VIEW UI](/docs/ui.md)

## Installation
```bash
python -m pip install -r requirements.txt
```

![MMDIT](doc/images/sample.png)

## Architecture

- **Autoencoder:** DC-AE f32c32 (32× compression, 32 latent channels)
- **Text Encoder:** T5-Large
- **Backbone:** MMDiT with Joint Attention
- **Implementation:** Custom implementation (no pretrained base model)
- **Parameters:** 4B

---

## Train Environment

- **Framework:** PyTorch (no custom CUDA kernels)
- **GPU:** NVIDIA RTX PRO 6000 (96GB VRAM) x 1
- **Platform:** Google Colab
- **Batch Size:** Micro-batch 4 × Gradient Accumulation 8 = Effective Batch 32

- **Training Speed:** ~2.3 s / optimizer step
- **Max Resolution:** 1024 × 1024
- **Dataset:** 2M images
- **Objective:** Flow Matching
- **Precision:** BF16
- **Optimizer:** AdamW
- **Learning Rate:** 1e-4 ~ 5e~5
- **EMA:** Enabled

### Training Stages

| Stage | Description | Steps | Patch Size |
|:---:|---|---:|:---:|
| Stage 1 | Pre-training | 600K | 2 |
| Stage 2 | Fine-tuning | 11K | 1 |

**Stage 2 Notes:**
- Reset patch embedding weights.
- Reset output projection weights.

---

### Current Limitations

- **Full-body Generation:** Full-body generation remains challenging due to insufficient full-body samples in the training dataset.
- **Observation:** The current dataset contains insufficient full-body examples to learn consistent full-body composition and anatomical details.
- **Future Work:** Increase the proportion and diversity of full-body images in the training dataset.

- **GITHUB link** https://github.com/FREEANIMA/DiffusionSampling.git
