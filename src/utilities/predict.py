import torch
from tqdm.auto import tqdm
from .solvers import euler, heun, ddpm_plus, renoise, sigma_lo_of
import common.globals as g


@torch.inference_mode()
def velocity_predict(model, x, sigma, cond_emb, cond_mask, uncond_emb, uncond_mask, cfg_scale, dtype):
    x_in = x.to(dtype)
    sig = sigma.expand(x.shape[0])

    use_cfg = cfg_scale != 1.0

    def fwd(emb, mask):
        if g.INIT_DEVICE == "cpu":
            out, _ = model.forward_cpu(x=x_in, sigma=sig, txt_emb=emb, txt_mask=mask)
        else:
            out, _ = model.forward(x=x_in, sigma=sig, txt_emb=emb, txt_mask=mask)
        return out.float()

    with torch.amp.autocast(device_type=x.device.type, dtype=dtype):
        c = fwd(cond_emb, cond_mask)
        v = c
        if use_cfg:
            u = fwd(uncond_emb, uncond_mask)
            v = u + cfg_scale * (c - u)

    return v


@torch.inference_mode()
def steps_sampling(model, x, sigmas, calculate_type, cfg_scale, cond_emb, cond_mask,
                   uncond_emb, uncond_mask, gen, dtype, eta, after_step=None):
    model.eval()
    n = len(sigmas) - 1
    kw = dict(model=model, cond_emb=cond_emb, cond_mask=cond_mask,
              uncond_emb=uncond_emb, uncond_mask=uncond_mask,
              cfg_scale=cfg_scale, dtype=dtype)

    latents = []
    sigmas_steps = []

    try:
        for i in tqdm(range(n), desc="sampling"):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]

            sigmas_steps.append(float(sigma_next))
            v = velocity_predict(x=x, sigma=sigma, **kw)

            last = (i == n - 1)

            use_noise = eta > 0 and not last
            sigma_lo = sigma_lo_of(sigma_next, eta) if use_noise else sigma_next
            z = (torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
                 if use_noise else None)

            if calculate_type == "euler":
                x = euler(x, v, sigma, sigma_lo)
                if use_noise:
                    x = renoise(x, z, sigma_lo, sigma_next)

            elif calculate_type == "heun":
                if last:
                    x = euler(x, v, sigma, sigma_next)
                else:
                    x_e = euler(x, v, sigma, sigma_lo)
                    v2 = velocity_predict(x=x_e, sigma=sigma_lo, **kw)
                    x = heun(x, v, v2, sigma, sigma_lo)
                    del x_e, v2
                if use_noise:
                    x = renoise(x, z, sigma_lo, sigma_next)

            elif calculate_type == "ddpm_plus":
                x = ddpm_plus(x, v, z, sigma, sigma_next, eta)

            else:
                raise ValueError(f"Unsupported sampling method: {calculate_type}")

            del z

            if after_step is not None:
                x = after_step(x, sigma, sigma_next)

            latents.append(x.detach().float().cpu())
    finally:
        torch.cuda.empty_cache()

    return x, latents, sigmas_steps