import os
import torch
import random
from src.utilities import sigma_schedular, steps_sampling, gpu_context, _save_img, _save_text, _save_plot
import common.globals as g
import matplotlib.pyplot as plt

INIT_PROMPT = "general img; "

class Sampling:
    def __init__(self, base_root, model, text_encoder, vae, config, console=None):
        self.base_root = base_root
        self.device = "cuda"
        self.dtype=torch.bfloat16
        self.f = config["compression_ratio"]
        self.in_channels = config["in_channels"]
        self.patch_size = config["patch_size"]

        self.model = model
        self.text_encoder = text_encoder
        self.vae = vae

        self.console = console

    def _snap(self, v, unit):
        return max(round(v / unit), 1) * unit

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

    def save_imgs(self, tags, cfg_scale, final_tags, imgs, steps, calculate_type, shift, seeds, video_imgs, eta, eta_mode, sigmas_steps=None):
        save_dir = os.path.join(self.base_root, "generation")
        save_paths = _save_img(imgs=imgs, filename="sample", save_dir=save_dir, video_imgs=video_imgs, sigmas=sigmas_steps)

        _save_text(
            os.path.dirname(save_paths[0]),
            tags=tags,
            tags_list=final_tags,
            cfg_scale=cfg_scale,
            steps=steps,
            calc_type=calculate_type,
            shift=shift,
            seeds=seeds,
            eta=eta,
            eta_mode=eta_mode
        )

        self.console.put(f"img saved: {save_paths[0]}")

        return save_paths

    @torch.inference_mode()
    def sampling(self, tags, final_tags, shift, calculate_type, cfg_scale, steps, H, W, B=1, 
                 eta=0.3, seed=None, on_step=None, frame_save=False, eta_mode="cos"):
        cond_emb, uncond_emb, cond_mask, uncond_mask, noise, gens, sigmas, seeds = self.set_init(tags, seed, steps, shift, B, H, W)

        total = len(sigmas) - 1
        done = 0
        if on_step:
            on_step(0, total)

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
            x=noise,
            sigmas=sigmas,
            calculate_type=calculate_type,
            cfg_scale=cfg_scale,
            cond_emb=cond_emb,
            cond_mask=cond_mask,
            uncond_emb=uncond_emb,
            uncond_mask=uncond_mask,
            gens=gens,
            dtype=self.dtype,
            after_step=after_step,
            eta=eta,
            eta_mode=eta_mode
        )

        if g.INIT_DEVICE == "cpu":
            self.model.to("cpu")

        del cond_emb, uncond_emb, cond_mask, uncond_mask
        del noise, sigmas

        imgs = self.vae_decode(x)

        video_imgs = None
        if frame_save:
            video_imgs = []
            for latent in latents:
                video_imgs.append(self.vae_decode(latent)[0])

        del latents

        save_paths = self.save_imgs(
            tags=tags, cfg_scale=cfg_scale, final_tags=final_tags,
            imgs=imgs, steps=steps, calculate_type=calculate_type,
            shift=shift, seeds=seeds,
            video_imgs=video_imgs, sigmas_steps=sigmas_steps,
            eta=eta, eta_mode=eta_mode, 
        )

        return imgs, save_paths