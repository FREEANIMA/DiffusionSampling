import os
from tqdm.auto import tqdm
from datetime import datetime
import torch
import random
from src.utilities import sigma_schedular, steps_sampling, gpu_context

INIT_PROMPT = "general img; "

class Sampling:
    def __init__(self, base_root, model, text_encoder, vae, config, console=None):
        self.base_root = base_root
        self.shift: float = 3.0
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

    @torch.inference_mode()
    def sampling(self, tags, calculate_type, cfg_scale, steps, H, W, seed=None, on_step=None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        cond_emb, uncond_emb, cond_mask, uncond_mask = self.text_encode(tags=tags)


        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)

        noise = torch.randn(
            (1, self.in_channels, H // self.f, W // self.f),
            generator=gen,
            device=self.device,
            dtype=torch.float32,
        )

        sigmas = sigma_schedular(step=steps, device=self.device, shift=self.shift)

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

        with gpu_context(self.model, self.device):
            x = steps_sampling(
                model=self.model,
                x=noise,
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

        del cond_emb, uncond_emb, cond_mask, uncond_mask
        del noise, sigmas

        img = self.vae_decode(x)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.base_root, "generation", stamp)
        os.makedirs(save_dir, exist_ok=True)
        save_img_name = os.path.join(save_dir, "img.png")

        img.save(save_img_name)

        self.console.put(f"img saved: {save_img_name}")

        return img, save_img_name