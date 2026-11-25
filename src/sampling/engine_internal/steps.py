import torch
from tqdm.auto import tqdm

from src.network import MMDIT
from src.core import state
from .types import Txt2ImgInput, InitOutput, InpaintCtx
from .solvers import euler, heun_step, dpmpp_2m

@torch.inference_mode()
@torch.amp.autocast(device_type=state.config.device, dtype=state.config.torch_dtype)
def one_steps(model:MMDIT, x, init_out:InitOutput, sigma, cfg) -> torch.Tensor:
    x_in = x.to(state.config.torch_dtype)
    B, _, _, _ = x_in.shape

    sigma_1d = torch.full((2 * B,), sigma, device=state.config.device, dtype=torch.float32)

    out, _ = model.forward(
        x= torch.cat([x_in, x_in], dim=0),
        sigma= sigma_1d,
        txt_emb= torch.cat([init_out.cond_emb, init_out.uncond_emb], dim=0),
        txt_mask= torch.cat([init_out.cond_mask, init_out.uncond_mask], dim=0),
        max_side= init_out.max_side
    )

    c, u = out.float().chunk(2, dim=0)
    v = u + cfg * (c - u)

    return v

def sampling(model, txt2img_input, init_out, inpaint_ctx=None, on_step=None):
    sigmas = init_out.sigmas
    n = len(sigmas) - 1
    calc = txt2img_input.calculate_type

    latents = []
    sigmas_steps = []
    noise = init_out.noise

    if inpaint_ctx is not None:
        z0 = inpaint_ctx.z0.to(device=noise.device, dtype=noise.dtype)
        mask = inpaint_ctx.mask.to(device=noise.device, dtype=noise.dtype)
        def clamp_known(x, s):
            return mask * x + (1 - mask) * ((1 - s) * z0 + s * noise)
    else:
        clamp_known = None

    x = noise if clamp_known is None else clamp_known(noise, float(sigmas[0]))

    x0_prev = None
    h_prev = None

    for i in tqdm(range(n), desc="sampling"):
        sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
        cfg = float(init_out.cfgs[i])

        first_v = []

        def model_fn(x_in, s, cfg=cfg):
            if clamp_known is not None:
                x_in = clamp_known(x_in, s)
            v = one_steps(model, x_in, init_out, s, cfg)
            if not first_v:
                first_v.append((x_in, v))
            return v

        if calc == "heun":
            x = heun_step(model_fn, x, sigma, sigma_next)

        elif calc == "dpmpp_2m":
            v = model_fn(x, sigma)
            x0_cur = x - sigma * v
            x, h = dpmpp_2m(x, x0_cur, x0_prev, sigma, sigma_next, h_prev)
            # 상태 갱신: renoise/클램프가 없을 때만 prev 유지
            if clamp_known is None:
                x0_prev, h_prev = x0_cur, h
            else:
                x0_prev, h_prev = None, None
        else:  # euler
            v = model_fn(x, sigma)
            x = euler(x, v, sigma, sigma_next)

        x_in0, v0 = first_v[0]
        x0_hat = x_in0 - sigma * v0

        if clamp_known is not None:
            x = clamp_known(x, sigma_next)

        sigmas_steps.append(sigma_next)
        latents.append(x0_hat)

        tqdm.write(
            f"[{i:02d}] s={sigma:.3f}->{sigma_next:.3f} cfg={cfg:.1f} "
            f"x_in.std={x_in0.std().item():.3f} x_in.max={x_in0.abs().max().item():.2f} | "
            f"v.std={v0.std().item():.3f} v.max={v0.abs().max().item():.2f} | "
            f"x0hat.std={x0_hat.std().item():.3f} | "
            f"x_next.std={x.std().item():.3f} x_next.max={x.abs().max().item():.2f} | "
            f"nan={bool(torch.isnan(x).any())}"
        )

        if on_step:
            on_step(i + 1, n)

    return x, latents, sigmas_steps
    