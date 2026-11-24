import torch
from PIL import Image
import numpy as np
import cv2


from src.core import log, state, error_report
from src.external import TE, VAE
from src.network import MMDIT, LLM

from src.sampling.engine_internal import (
    Txt2ImgInput, InitOutput, InpaintInput, InpaintCtx,
    setup_init, sampling, arr_resize, mask_generation
)

class SamplingEngine:
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
            init_out:InitOutput = setup_init(model=self.te, config=state.config, txt2img_input=txt2img_input)

            start_idx = int(np.argmin(np.abs(init_out.sigmas - start_t)))
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
            arr, H, W = arr_resize(
                img,
                state.config.allowable_size,
                state.config.patch_size,
                state.config.compression_ratio,
                upscale=upscale
            )

            txt2img_input.H = H
            txt2img_input.W = W
            
            mask = inpaint_input.mask.convert("L").resize((W, H), Image.Resampling.NEAREST)
            mask_arr = np.array(mask, dtype=np.uint8)

            np_tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            np_tensor = np_tensor.to(device=state.config.device, dtype=state.config.dtype)

            latent = self.vae.encoder(np_tensor)
            mask_lat = mask_generation(mask_arr, latent)

            B = len(txt2img_input.tags)

            ctx = InpaintCtx(
                z0=latent.expand(B, -1, -1, -1),
                mask=mask_lat.expand(B, -1, -1, -1)
            )

            return self.txt2img(txt2img_input, start_t=inpaint_input.start_t, inpaint_ctx=ctx)
        
        except Exception as e:
            error_report(e, "inpaint")
            log(self.console, e)
            raise    
