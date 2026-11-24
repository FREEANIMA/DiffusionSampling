import torch
from PIL import Image

from .types import SamplingInput, InitOutput
from .setup import setup_init

from src.core import log
from src.external import TE, VAE
from src.network import MMDIT, LLM

from .steps import sampling

class InPaint:
    def __init__(self, model:MMDIT, llm:LLM, te:TE, vae:VAE, console=None):
        self.console = console
        self.te = te
        self.vae = vae
        self.llm = llm
        self.model = model

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
        save_dir = os.path.join(self.base_root, "inpainting")
        save_paths = _save_img(imgs=imgs, filename="inpainting", save_dir=save_dir, video_imgs=video_imgs, sigmas=sigmas_steps)

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
    def sampling(self, , img_bytes, start_t, B=1, eta=0.3, seed=None, mask=None, on_step=None, save_dir=None, frame_save=False, eta_mode="cos"):
