import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageFilter
import common.globals as g

from src.utilities import sigma_schedular, steps_sampling, img_resize, gpu_context, _save_img
INIT_PROMPT = "general img; "

class Detailering:
    def __init__(self, base_root, model, text_encoder, vae, config, console=None):
        self.base_root = base_root
        self.shift: float = 2.0
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
    def text_encode(self, tags):
        with gpu_context(self.text_encoder.text_encoder, self.device):
            cond_emb, cond_mask = self.text_encoder.encoder(tags)
            uncond_emb, uncond_mask = self.text_encoder.encoder([INIT_PROMPT])

        return cond_emb, uncond_emb, cond_mask, uncond_mask
    
    @torch.inference_mode()
    def vae_decode(self, x):
        with gpu_context(self.vae.vae, self.device):
            decoded = self.vae.decoder(x)

        img = self.vae._tensor_to_pil(decoded)
        return img

    @torch.inference_mode()
    def sampling(self, tags, calculate_type, cfg_scale, steps, img_bytes, start_t, seed=None, on_step=None):
        arr, H, W = img_resize(img_bytes, self.allowable_size, self.patch_size, self.f)
        img = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        img = img.to(device=self.device, dtype=self.dtype)

        with gpu_context(self.vae.vae, self.device):
            latent = self.vae_encode(img)

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

        if g.INIT_DEVICE == "cpu":
            self.model.to(self.device)
            self.model.blocks.to("cpu")
            torch.cuda.empty_cache() 
            
        x, latents, sigmas_steps = steps_sampling(
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
        if g.INIT_DEVICE == "cpu":
            self.model.to("cpu")

        del cond_emb, uncond_emb, cond_mask, uncond_mask
        del img, latent, noise, sigmas, x_t

        imgs = []
        for latent in latents:
            imgs.append(self.vae_decode(latent))

        del latents

        return self.vae_decode(x), imgs, sigmas_steps

    def merge_img(self, origin_img, img_out, square, save_dir, imgs, sigmas_steps, feather=16, img_in=None):
        left, top, right, bottom = square
        w, h = right - left, bottom - top

        out = origin_img.convert("RGB").copy()
        patch = img_out.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)

        f = max(1, min(feather, w // 4, h // 4))
        alpha = Image.new("L", (w, h), 0)
        alpha.paste(255, (f, f, w - f, h - f))
        alpha = alpha.filter(ImageFilter.GaussianBlur(f / 2))

        out.paste(patch, (left, top), alpha)

        save_dir = os.path.join(self.base_root, "detailer")
        _save_img(img=img_out, filename="patch", save_dir=save_dir)
        _save_img(img=img_in, filename="input", save_dir=save_dir)
        save_img_name = _save_img(img=out, filename="merged", save_dir=save_dir)
        _save_img(imgs=imgs, filename="detail_mp4", save_dir=save_dir, sigmas_steps=sigmas_steps)

        self.console.put(f"merged saved: {save_img_name}")

        return out, save_img_name