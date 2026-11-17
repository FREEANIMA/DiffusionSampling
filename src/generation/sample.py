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

        img = self.vae._tensor_to_pil(decoded)
        return img

    @torch.inference_mode()
    def sampling(self, tags, final_tags, shift, calculate_type, cfg_scale, steps, H, W,  eta=0.3, seed=None, on_step=None):

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

        sigmas = sigma_schedular(step=steps, device=self.device, shift=shift)

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
            gen=gen,
            dtype=self.dtype,
            after_step=after_step,
            eta=eta
        )

        if g.INIT_DEVICE == "cpu":
            self.model.to("cpu")

        del cond_emb, uncond_emb, cond_mask, uncond_mask
        del noise, sigmas

        img = self.vae_decode(x)

        imgs = []
        for latent in latents:
            imgs.append(self.vae_decode(latent))

        # latent_diff = []

        # for i in range(len(latents) - 1):
        #     diff = (latents[i + 1].float() - latents[i].float()).abs()
        #     latent_diff.append(diff)

        # self.inspect_plot(latent_diff=latent_diff)
        
        save_dir = os.path.join(self.base_root, "generation")
        save_img_name = _save_img(img=img, filename="sample", save_dir=save_dir, imgs=None, sigmas=sigmas_steps)

        _save_text(
            os.path.dirname(save_img_name),
            tags=tags,
            tags_list=final_tags,
            cfg_scale=cfg_scale,
            steps=steps,
            calc_type=calculate_type,
            shift=shift,
            seed=seed,
        )

        self.console.put(f"img saved: {save_img_name}")

        del latents

        return img, save_img_name

    def inspect_plot(self, latent_diff):
        maps = [
            diff.mean(dim=1)[0].cpu().numpy()
            for diff in latent_diff
        ]

        cols = 5
        rows = (len(maps) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))

        if rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        vmax = max(m.max() for m in maps)

        for i, (ax, m) in enumerate(zip(axes, maps)):
            ax.imshow(m, cmap="inferno", vmin=0, vmax=vmax)
            ax.set_title(f"Step {i}")
            ax.axis("off")

        for ax in axes[len(maps):]:
            ax.axis("off")

        fig.tight_layout()

        save_dir = os.path.join(self.base_root, "generation")
        save_plot_name = _save_plot(
            filename="sample_plot",
            save_dir=save_dir,
            plots=fig
        )

        plt.close(fig)

        self.console.put(f"plot saved: {save_plot_name}")

        return save_plot_name