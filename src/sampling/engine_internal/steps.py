import torch
from tqdm.auto import tqdm

from src.network import MMDIT
from src.core import state
from .types import Txt2ImgInput, InitOutput, InpaintCtx
from .solvers import euler, heun_step, midpoint_step

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
    )

    c, u = out.float().chunk(2, dim=0)
    v = u + cfg * (c - u)

    return v



def sampling(model, txt2img_input: Txt2ImgInput, init_out: InitOutput, inpaint_ctx: InpaintCtx = None, on_step=None):
    sigmas = init_out.sigmas
    n = len(sigmas) - 1
    calc = txt2img_input.calculate_type

    latents = []
    sigmas_steps = []
    noise = init_out.noise

    if inpaint_ctx is not None:
        z0 = inpaint_ctx.z0.to(device=noise.device, dtype=noise.dtype)
        mask = inpaint_ctx.mask.to(device=noise.device, dtype=noise.dtype)   # 1 = 수정, 0 = 보존

        def clamp_known(x, s):
            return mask * x + (1 - mask) * ((1 - s) * z0 + s * noise)
    else:
        clamp_known = None

    x = noise if clamp_known is None else clamp_known(noise, float(sigmas[0]))

    tqdm.write(f"sigmas={[round(float(s),3) for s in sigmas]}")
    tqdm.write(f"x0 std={x.std().item():.3f} dtype={x.dtype} sigma dtype={sigmas.dtype}")

    for i in tqdm(range(n), desc="sampling"):
        sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])

        cfg = init_out.cfgs[i]
        eta = init_out.etas[i]

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
        else:
            v = model_fn(x, sigma)
            x = euler(x, v, sigma, sigma_next)

        x_in0, v0 = first_v[0]
        x0_hat = x_in0 - sigma * v0
        tqdm.write(
            f"[{i:02d}] s={sigma:.3f}->{sigma_next:.3f} cfg={float(cfg):.1f} eta={float(eta):.3f} | "
            f"x_in.std={x_in0.std().item():.3f} x_in.max={x_in0.abs().max().item():.2f} | "
            f"v.std={v0.std().item():.3f} v.max={v0.abs().max().item():.2f} | "
            f"x0hat.std={x0_hat.std().item():.3f} | "
            f"x_next.std={x.std().item():.3f} x_next.max={x.abs().max().item():.2f} | "
            f"nan={bool(torch.isnan(x).any())}"
        )

        if clamp_known is not None:
            x = clamp_known(x, sigma_next) 

        sigmas_steps.append(sigma_next)
        latents.append(x0_hat)
        
        if on_step:
            on_step(i + 1, n)
            
    return x, latents, sigmas_steps