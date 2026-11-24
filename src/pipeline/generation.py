import torch
from PIL import Image

from .types import Txt2ImgInput, InitOutput, InpaintInput
from .setup import setup_init

from src.core import log
from src.external import TE, VAE
from src.network import MMDIT, LLM

from .steps import sampling
from .utils import img_resize, mask_generation

class Generation:
    def __init__(self, model:MMDIT, llm:LLM, te:TE, vae:VAE, console=None):
        self.console = console
        self.te = te
        self.vae = vae
        self.llm = llm
        self.model = model
        self.model.eval()

    def vae_decode(self, x: torch.Tensor, latents: list[torch.Tensor], video_save: bool) -> tuple[list[Image.Image], list[Image.Image]]:
        imgs: list[Image.Image] = self.vae._tensor_to_pils(self.vae.decoder(x))

        frames: list[Image.Image] = []
        if video_save and latents:
            for latent in latents:
                frames.extend(self.vae._tensor_to_pils(self.vae.decoder(latent)))

        return imgs, frames

    def txt2img(self, txt2img_input:Txt2ImgInput) -> tuple[list[Image.Image], list[Image.Image], list[float]]:
        init_out:InitOutput = setup_init(txt2img_input)

        x, latents, sigmas_steps = sampling(
            model=self.model,
            txt2img_input=txt2img_input,
            init_out=init_out,
            dtype=self.dtype,
        )

        return self.vae_decode(x, latents, txt2img_input.video_save), sigmas_steps

    def inpaint(self, txt2img_input:Txt2ImgInput, inpainting_input:InpaintInput) -> tuple[list[Image.Image], list[Image.Image], list[float]]:
        arr, H, W = img_resize(inpainting_input.img_bytes, self.allowable_size, self.patch_size, self.f)

        mask = mask.convert("L").resize((W, H), Image.Resampling.NEAREST)
        mask_arr = np.array(mask, dtype=np.uint8)

        img = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        img = img.to(device=self.device, dtype=self.dtype)

        latent = self.vae_encode(img)
        mask_lat = self.mask_generation(mask_arr, latent)

        cond_emb, uncond_emb, cond_mask, uncond_mask, noise, gens, sigmas, seeds = self.set_init(tags, seed, steps, shift, B, H, W)

        start_idx = int(torch.argmin((sigmas - start_t).abs()).item())
        sigmas = sigmas[start_idx:]

        x, latents, sigmas_steps = sampling(
            model=self.model,
            txt2img_input=txt2img_input,
            init_out=init_out,
            dtype=self.dtype,
        )

        return self.vae_decode(x, latents, txt2img_input.video_save), sigmas_steps
        

    