import torch
from PIL import Image
import numpy as np
from src.core import log, state, error_report, dbg
from src.external import TE, VAE
from src.network import MMDIT, LLM
from src.tag import TagGenration
from src.storage import download_model, load_ema_only
import numpy as np
from src.sampling.engine_internal import (
    Txt2ImgInput, InitOutput, InpaintInput, InpaintCtx,
    setup_init, sampling, arr_resize, mask_generation
)

class SamplingEngine:
    def __init__(self, console=None):
        self.console = console

        # test
        self.model = None

        self.te = TE()
        self.vae = VAE()

        self.llm:TagGenration = TagGenration(console=console)

        download_model(config=state.config, filename="ema_model.pt")
        self.model:MMDIT = load_ema_only(
            config=state.config,
            network=MMDIT,
            console=console
        )
        

    def vae_decode(self, x: torch.Tensor, latents: list[torch.Tensor], frame_save: bool) -> tuple[list[Image.Image], list[Image.Image]]:
        imgs: list[Image.Image] = self.vae._tensor_to_pils(self.vae.decoder(x))

        frames: list[Image.Image] = []
        if frame_save and latents:
            for latent in latents:
                frames.extend(self.vae._tensor_to_pils(self.vae.decoder(latent)))

        return imgs, frames

    def txt2img(self, txt2img_input:Txt2ImgInput, start_t=0, inpaint_ctx: InpaintCtx | None = None) -> tuple[list[Image.Image], list[Image.Image], list[float]]:
        try:
            if not txt2img_input.prompt_static:
                self.llm.generation(
                    tags=txt2img_input.prompts, 
                    nsfw=txt2img_input.nsfw,
                    H=txt2img_input.H,
                    W=txt2img_input.W
                )
  
            init_out:InitOutput = setup_init(model=self.te, config=state.config, txt2img_input=txt2img_input)

            start_idx = int(np.argmin(np.abs(init_out.sigmas - start_t)))
            init_out.sigmas = init_out.sigmas[start_idx:]
            init_out.cfgs = init_out.cfgs[start_idx:]
            init_out.etas = init_out.etas[start_idx:]

            x, latents, sigmas_steps = sampling(
                model=self.model,
                txt2img_input=txt2img_input,
                init_out=init_out,
                inpaint_ctx=inpaint_ctx
            )

            imgs, frames = self.vae_decode(x, latents, txt2img_input.frame_save)

            return imgs, frames, sigmas_steps
        
        except Exception as e:
            error_report(e, "txt2img")
            log(self.console, e)
            raise

    def inpaint(self, txt2img_input:Txt2ImgInput, inpaint_input:InpaintInput) -> tuple[list[Image.Image], list[Image.Image], list[float]]:
        print(txt2img_input.H)
        print(txt2img_input.W)
        if inpaint_input.crop_img is not None:
            img = inpaint_input.crop_img
        else:
            img = inpaint_input.img

        origin_img = inpaint_input.img

        try:
            arr, H, W = arr_resize(
                np.asarray(img.convert('RGB')),   # PIL → (H, W, 3) uint8 ndarray
                1024, # 현재 한계치
                state.config.patch_size,
                state.config.compression_ratio,
                upscale=inpaint_input.upscale
            )

            print(f"H : {H}")
            print(f"W : {W}")
            txt2img_input.H = H
            txt2img_input.W = W
            print(txt2img_input.H)
            print(txt2img_input.W)
     
            mask = inpaint_input.mask.convert("L")
            if inpaint_input.crop_img is not None:
                x, y, w, h = inpaint_input.crop_box
                mask = mask.crop((x, y, x + w, y + h))
                
            mask = mask.resize((W, H), Image.Resampling.NEAREST)
            mask_arr = np.array(mask, dtype=np.uint8)

            np_tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            np_tensor = np_tensor.to(device=state.config.device, dtype=state.config.torch_dtype)

            latent = self.vae.encoder(np_tensor)
            mask_lat = mask_generation(mask_arr, latent)

            B = len(txt2img_input.prompts)

            ctx = InpaintCtx(
                z0=latent.expand(B, -1, -1, -1),
                mask=mask_lat.expand(B, -1, -1, -1)
            )

            return self.txt2img(txt2img_input, start_t=inpaint_input.start_t, inpaint_ctx=ctx)
        
        except Exception as e:
            error_report(e, "inpaint")
            log(self.console, e)
            raise    
