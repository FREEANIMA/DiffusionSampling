import os
import random
import numpy as np
import torch
from PIL import Image, ImageFilter
import common.globals as g
import torch.nn.functional as F

from src.utilities import sigma_schedular, steps_sampling, img_resize, gpu_context, _save_img, _save_text
INIT_PROMPT = "general img; "

class Detailering:
    def __init__(self, base_root, model, text_encoder, vae, config, console=None):
        self.base_root = base_root
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

        img = self.vae._tensor_to_pils(decoded)
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

        m_lat = F.adaptive_max_pool2d(m, output_size=latent.shape[-2:])
        return (m_lat > 0.5).float()

    def set_init(self, tags, seed, steps, shift, B, H, W):
        tags_batch = tags * B
        cond_emb, uncond_emb, cond_mask, uncond_mask = self.text_encode(tags=tags_batch)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        rng = random.Random(seed)
        seeds = [seed] + [rng.randint(0, 2**32 - 1) for _ in range(B - 1)]
        gens = [torch.Generator(device=self.device).manual_seed(s) for s in seeds]
        noise = torch.cat([
            torch.randn(
                (1, self.in_channels, H // self.f, W // self.f),
                generator=g_,
                device=self.device,
                dtype=torch.float32,
            )
            for g_ in gens
        ], dim=0)

        sigmas = sigma_schedular(step=steps, device=self.device, shift=shift)

        return cond_emb, uncond_emb, cond_mask, uncond_mask, noise, gens, sigmas, seeds

    def save_imgs(self, tags, cfg_scale, final_tags, imgs, steps, calculate_type, shift, seeds, video_imgs, sigmas_steps=None):
        save_dir = os.path.join(self.base_root, "detailer")
        save_paths = _save_img(imgs=imgs, filename="detailer", save_dir=save_dir, video_imgs=video_imgs, sigmas=sigmas_steps)

        _save_text(
            os.path.dirname(save_paths[0]),
            tags=tags,
            tags_list=final_tags,
            cfg_scale=cfg_scale,
            steps=steps,
            calc_type=calculate_type,
            shift=shift,
            seeds=seeds,
        )

        self.console.put(f"img saved: {save_paths[0]}")

        return save_paths

    @torch.inference_mode()
    def sampling(self, arr, tags, shift, H, W, calculate_type, cfg_scale, steps, start_t, 
                 B=1, eta=0.3, seed=None, mask=None, on_step=None, eta_mode="cos", frame_save=False):
        if mask is None:
            raise ValueError("mask가 필요합니다.")

        mask = mask.convert("L").resize((W, H), Image.Resampling.NEAREST)
        mask_arr = np.array(mask, dtype=np.uint8)

        img = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        img = img.to(device=self.device, dtype=self.dtype)

        latent = self.vae_encode(img)
        mask_lat = self.mask_generation(mask_arr, latent)

        cond_emb, uncond_emb, cond_mask, uncond_mask, noise, gens, sigmas, seeds = self.set_init(tags, seed, steps, shift, B, H, W)

        sigmas = sigma_schedular(step=steps, device=self.device, shift=shift)
        start_idx = int(torch.argmin((sigmas - start_t).abs()).item())
        sigmas = sigmas[start_idx:]
        if len(sigmas) < 2:
            raise ValueError("start_t가 너무 작아 샘플링할 step이 없습니다.")
        start_sigma = sigmas[0]

        x_t = (1 - start_sigma) * latent + start_sigma * noise

        total = len(sigmas) - 1
        done = 0
        if on_step:
            on_step(0, total)
            
        def after_step(x, sigma, sigma_next):
            nonlocal done
            done += 1
            if on_step:
                on_step(min(done, total), total)

            s = sigma_next.view(1, 1, 1, 1) if torch.is_tensor(sigma_next) else float(sigma_next)
            orig_t = (1 - s) * latent + s * noise
            return (1 - mask_lat) * orig_t + mask_lat * x

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
            cond_emb=cond_emb, cond_mask=cond_mask,
            uncond_emb=uncond_emb, uncond_mask=uncond_mask,
            gens=gens,
            dtype=self.dtype,
            after_step=after_step,
            eta=eta,
            eta_mode=eta_mode
        )
        if g.INIT_DEVICE == "cpu":
            self.model.to("cpu")

        del cond_emb, uncond_emb, cond_mask, uncond_mask, latent, mask_lat, noise, x_t, sigmas, latents
        imgs = self.vae_decode(x)

        video_imgs = None
        if frame_save:
            video_imgs = []
            for latent in latents:
                video_imgs.append(self.vae_decode(latent)[0])

        return imgs, sigmas_steps, seeds, video_imgs

    def merge_img(self, origin_img, fixed_imgs, box, sigmas_steps, tags, calculate_type, shift, seeds,
                  steps, final_tags, cfg_scale, paste_mask=None, video_imgs=None):
        left, top, right, bottom = box
        w, h = right - left, bottom - top

        if not isinstance(fixed_imgs, (list, tuple)):
            fixed_imgs = [fixed_imgs]

        imgs = []
        for i, patch in enumerate(fixed_imgs):
            patch = patch.convert("RGB")
            if patch.size != (w, h):
                patch = patch.resize((w, h), Image.Resampling.LANCZOS)

            out = origin_img.copy()
            out.paste(patch, (left, top), paste_mask)

            imgs.append(out)

        save_paths = self.save_imgs(
            tags=tags, final_tags=final_tags, imgs=imgs, 
            calculate_type=calculate_type, shift=shift, video_imgs=video_imgs, 
            sigmas_steps=sigmas_steps, cfg_scale=cfg_scale, steps=steps, seeds=seeds,
            )

        return imgs, save_paths