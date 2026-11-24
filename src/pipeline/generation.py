import torch
from PIL import Image
import numpy as np

from .types import Txt2ImgInput, InitOutput, InpaintInput, InpaintCtx
from .setup import setup_init

from src.core import log, state, error_report
from src.external import TE, VAE
from src.network import MMDIT, LLM

from .steps import sampling
from .utils import img_resize, mask_generation, upscale_to, img_to_bytes

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

    def txt2img(self, txt2img_input, start_t=0, inpaint_ctx: InpaintCtx | None = None) -> tuple[list[Image.Image], list[Image.Image], list[float]]:
        try:
            init_out:InitOutput = setup_init(txt2img_input)

            start_idx = int(torch.argmin((init_out.sigmas - start_t).abs()).item())
            init_out.sigmas = init_out.sigmas[start_idx:]

            x, latents, sigmas_steps = sampling(
                model=self.model,
                txt2img_input=txt2img_input,
                init_out=init_out,
                inpaint_ctx=inpaint_ctx
            )

            imgs, frames = self.vae_decode(x, latents, txt2img_input.video_save)

            return imgs, frames, sigmas_steps
        
        except Exception as e:
            error_report(e, "txt2img")
            log(self.console, e)
            raise

    def inpaint(self, txt2img_input:Txt2ImgInput, inpaint_input:InpaintInput, upscale=False) -> tuple[list[Image.Image], list[Image.Image], list[float]]:
        img = inpaint_input.img

        try:
            if upscale:
                img = upscale_to(inpaint_input.img, target=1024)

            img_bytes = img_to_bytes(img, ".png")

            f = state.config.compression_ratio * state.config.patch_size
            arr, H, W = img_resize(img_bytes, state.config.allowable_size, state.config.patch_size, f)
            mask = inpaint_input.mask.convert("L").resize((W, H), Image.Resampling.NEAREST)
            mask_arr = np.array(mask, dtype=np.uint8)

            np_tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            np_tensor = np_tensor.to(device=state.config.device, dtype=state.config.dtype)

            latent = self.vae.encoder(np_tensor)
            mask_lat = mask_generation(mask_arr, latent)

            ctx = InpaintCtx(z0=latent, mask=mask_lat, noise=torch.randn_like(latent))

            return self.txt2img(txt2img_input, start_t=inpaint_input.start_t, inpaint_ctx=ctx)
        
        except Exception as e:
            error_report(e, "txt2img")
            log(self.console, e)
            raise    