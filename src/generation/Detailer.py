import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from src.utilities import sigma_schedular, steps_sampling, img_pad
INIT_PROMPT = "general img; "

class Detailering:
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
        return self.vae.encoder(img)

    @torch.inference_mode()
    def vae_decode(self, latent):
        decoded = self.vae.decoder(latent)
        return self.vae._tensor_to_pil(decoded)

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
    def text_encode(self, tags):
        cond_emb, cond_mask = self.text_encoder.encoder(tags)
        uncond_emb, uncond_mask = self.text_encoder.encoder([INIT_PROMPT])

        return cond_emb, uncond_emb, cond_mask, uncond_mask

    @torch.inference_mode()
    def sampling(self, tags, calculate_type, cfg_scale, steps, img_bytes, start_t, seed=None, on_step=None):

        arr, H, W, padding = img_pad(
            img_bytes,
            self.patch_size,
            self.f,
        )

        left, top, right, bottom = padding

        img = torch.from_numpy(arr).permute(2, 0, 1).float()
        img = img.div_(255.0).unsqueeze(0)
        img = img.to(device=self.device, dtype=self.dtype)
        latent = self.vae_encode(img)
        latent = self.vae.upscale_latent(
            latent,
            max_side=32,
            mode="bilinear",
        )

        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)

        noise = torch.randn(
            latent.shape,
            generator=gen,
            device=latent.device,
            dtype=latent.dtype,
        )

        sigmas = sigma_schedular(
            step=steps,
            device=self.device,
            shift=self.shift,
        )

        start_idx = int(torch.argmin((sigmas - start_t).abs()).item())
        sigmas = sigmas[start_idx:]

        start_sigma = sigmas[0]
        x_t = (1 - start_sigma) * latent + start_sigma * noise

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

            return x

        x = steps_sampling(
            model=self.model,
            x=x_t,
            sigmas=sigmas,
            calculate_type=calculate_type,
            cfg_scale=cfg_scale,
            cond_emb=cond_emb,
            cond_mask=cond_mask,
            uncond_emb=uncond_emb,
            uncond_mask=uncond_mask,
            gen=gen,
            dtype=self.dtype,
            after_step=after_step,
        )

        img_out = self.vae_decode(x)
        scale = 2

        original_w = W - left - right
        original_h = H - top - bottom

        img_out = img_out.crop((
            left * scale,
            top * scale,
            (left + original_w) * scale,
            (top + original_h) * scale,
        ))

        return img_out

    def merge_img(self, origin_img, img_out, square, save_dir, feather=16, img_in=None):
        left, top, right, bottom = square
        w, h = right - left, bottom - top

        out = origin_img.convert("RGB").copy()
        patch = img_out.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)

        f = max(1, min(feather, w // 4, h // 4))
        alpha = Image.new("L", (w, h), 0)
        alpha.paste(255, (f, f, w - f, h - f))
        alpha = alpha.filter(ImageFilter.GaussianBlur(f / 2))

        out.paste(patch, (left, top), alpha)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_dir, stamp)
        os.makedirs(save_dir, exist_ok=True)

        save_img_name = os.path.join(save_dir, "merged.png")
        out.save(save_img_name)
        img_out.save(os.path.join(save_dir, "patch.png"))
        if img_in is not None:
            img_in.save(os.path.join(save_dir, "input.png"))

        if self.console:
            self.console.put(f"merged saved: {save_img_name}")

        return out, save_img_name