import torch
from tqdm.auto import tqdm

from src.network import MMDIT
from src.core import state
from .types import Txt2ImgInput, InitOutput

@torch.inference_mode()
@torch.amp.autocast(device_type=state.config.device, dtype=state.config.torch_dtype)
def one_steps(model:MMDIT, x, txt2img_input:Txt2ImgInput, init_out:InitOutput, sigma) -> torch.Tensor:
    x_in = x.to(state.config.torch_dtype)
    B, _, _, _ = x_in.shape
    sigma_1d = torch.full((2 * B,), sigma, device=state.config.device, dtype=torch.float32)

    out, _ = model.forward(
        x= torch.cat([x_in, x_in], dim=0),
        sigma= sigma_1d,
        txt_emb= torch.cat([init_out.cond_emb, init_out.uncond_emb], dim=0),
        txt_mask= torch.cat([init_out.cond_mask, init_out.uncond_mask], dim=0),
    )

    c, u = out.float().chunk(2, dim=0)
    v = u + txt2img_input.cfg_scale * (c - u)

    return v

def sampling(model:MMDIT, txt2img_input:Txt2ImgInput, init_out:InitOutput) -> torch.Tensor:
    sigmas = init_out.sigmas
    n = len(sigmas) - 1

    latents = []
    sigmas_steps = []
    x = init_out.noise

    for i in tqdm(range(n), desc="sampling"):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        sigmas_steps.append(float(sigma_next))

        v = one_steps(
            model=model,
            x=x,
            txt2img_input=txt2img_input,
            init_out=init_out,
            sigma=sigma,
        )

        x0_hat = (x - float(sigma) * v).detach().float().cpu()

        if txt2img_input.calculate_type == "euler":

        elif txt2img_input.calculate_type == "heun":

        elif txt2img_input.calculate_type == "ddpm_plus":

        elif txt2img_input.calculate_type ==


        latents.append(x0_hat)

    return x, latents, sigmas_steps