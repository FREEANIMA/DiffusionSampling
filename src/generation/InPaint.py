import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.utilities import sigma_schedular, steps_sampling, img_resize, gpu_context
INIT_PROMPT = "general img; "

class InPainting:
    def __init__(self, base_root, model, text_encoder, vae, config, console=None):
        self.base_root = base_root
        self.shift: float = 3.0
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.f = config["compression_ratio"]
        self.in_channels = config["in_channels"]
        self.patch_size = config["patch_size"]
        self.allowable_size = config["allowable_size"]

        self.model = model
        self.text_encoder = text_encoder
        self.vae = vae
        self.console= console

    @torch.inference_mode()
    def vae_encode(self, img):
        with gpu_context(self.vae, self.device):
            return self.vae.encoder(img)

    @torch.inference_mode()
    def text_encode(self, tags):
        with gpu_context(self.text_encoder, self.device):
            cond_emb, cond_mask = self.text_encoder.encoder(tags)
            uncond_emb, uncond_mask = self.text_encoder.encoder([INIT_PROMPT])

        return cond_emb, uncond_emb, cond_mask, uncond_mask
    
    @torch.inference_mode()
    def vae_decode(self, x):
        with gpu_context(self.vae, self.device):
            decoded = self.vae.decoder(x)

        img = self.vae._tensor_to_pil(decoded)
        return img

    def mask_generation(self, mask_arr, latent):
        if isinstance(mask_arr, np.ndarray):
            m = torch.from_numpy(mask_arr).float()
        else:
            m = mask_arr.float()

        if m.max() > 1.0:
            m = m / 255.0

        if m.ndim == 2:
            m = m[None, None]
        elif m.ndim == 3:
            m = m[None]

        m = m.to(latent.device, latent.dtype)
        m_lat = F.interpolate(m, size=latent.shape[-2:], mode="nearest")
        return (m_lat > 0.5).float()

    @torch.inference_mode()
    def sampling(self, tags, calculate_type, cfg_scale, steps, img_bytes, start_t, seed=None, mask=None, on_step=None, save_dir=None):
        arr, H, W = img_resize(img_bytes, self.allowable_size, self.patch_size, self.f)

        # 변경 후
        if mask is None:
            raise ValueError("mask가 필요합니다.")

        mask = mask.convert("L").resize((W, H), Image.Resampling.NEAREST)
        mask_arr = np.array(mask, dtype=np.uint8)

        img = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        img = img.to(device=self.device, dtype=self.dtype)

        latent = self.vae_encode(img)
        mask_lat = self.mask_generation(mask_arr, latent)

        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        noise = torch.randn(
            latent.shape, generator=gen, device=latent.device, dtype=latent.dtype
        )

        sigmas = sigma_schedular(step=steps, device=self.device, shift=self.shift)
        start_idx = int(torch.argmin((sigmas - start_t).abs()).item())
        sigmas = sigmas[start_idx:]
        start_sigma = sigmas[0]
        orig_at_start = (1 - start_sigma) * latent + start_sigma * noise
        x_t = (1 - mask_lat) * latent + mask_lat * orig_at_start

        total = len(sigmas) - 1
        done = 0
        if on_step:
            on_step(0, total)
            
        cond_emb, uncond_emb, cond_mask, uncond_mask = self.text_encode(tags=tags)

        def after_step(x, sigma, sigma_next):
            nonlocal done
            done += 1
            if on_step:
                on_step(min(done, total), total)

            s = sigma.view(1, 1, 1, 1)
            orig_t = (1 - s) * latent + s * noise
            return (1 - mask_lat) * orig_t + mask_lat * x

        with gpu_context(self.model, self.device):
            x = steps_sampling(
                model=self.model,
                x=x_t,
                sigmas=sigmas,
                calculate_type=calculate_type,
                cfg_scale=cfg_scale,
                cond_emb=cond_emb, cond_mask=cond_mask,
                uncond_emb=uncond_emb, uncond_mask=uncond_mask,
                gen=gen,
                dtype=self.dtype,
                after_step=after_step,
            )

        del cond_emb, uncond_emb, cond_mask, uncond_mask
        del latent, mask_lat, noise, x_t, orig_at_start, sigmas

        img_out = self.vae_decode(x)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_dir, stamp)
        os.makedirs(save_dir, exist_ok=True)

        save_img_name = os.path.join(save_dir, "img.png")
        img_out.save(save_img_name)

        self.console.put(f"img saved: {save_img_name}")

        return img_out, save_img_name