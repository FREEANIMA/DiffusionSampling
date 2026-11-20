import torch
from tqdm.auto import tqdm
from .solvers import euler, heun, ddpm_plus, renoise, sigma_lo_of
import common.globals as g


@torch.inference_mode()
def velocity_predict(model, x, sigma, cond_emb, cond_mask, uncond_emb, uncond_mask, cfg_scale, dtype):
    x_in = x.to(dtype)
    use_cfg = cfg_scale != 1.0
    B = x_in.shape[0]

    def _rep(t):
        return t.expand(B, *t.shape[1:]) if t.shape[0] == 1 else t

    cond_emb, cond_mask = _rep(cond_emb), _rep(cond_mask)
    uncond_emb, uncond_mask = _rep(uncond_emb), _rep(uncond_mask)

    def fwd(x_, emb, mask):
        sig = sigma.expand(x_.shape[0])
        fn = model.forward_cpu if g.INIT_DEVICE == "cpu" else model.forward
        out, _ = fn(x=x_, sigma=sig, txt_emb=emb, txt_mask=mask)
        return out.float()

    with torch.amp.autocast(device_type=x.device.type, dtype=dtype):
        if use_cfg:
            out = fwd(
                torch.cat([x_in, x_in], dim=0),
                torch.cat([cond_emb, uncond_emb], dim=0),
                torch.cat([cond_mask, uncond_mask], dim=0),
            )
            c, u = out.chunk(2, dim=0)
            v = u + cfg_scale * (c - u)
        else:
            v = fwd(x_in, cond_emb, cond_mask)

    return v

@torch.inference_mode()
def steps_sampling(model, x, sigmas, calculate_type, cfg_scale, cond_emb, cond_mask, uncond_emb, uncond_mask, gens, dtype, eta, 
                   after_step=None, eta_sigma_min=0.6, cfg_min_sigma=0.0):
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
            cfg_i = cfg_scale if float(sigma) > cfg_min_sigma else 1.0
            kw_i = {**kw, "cfg_scale": cfg_i}
            v = velocity_predict(x=x, sigma=sigma, **kw_i)
            last = (i == n - 1)

            use_noise = eta > 0 and (float(sigma_next) > eta_sigma_min) and not last
            sigma_lo = sigma_lo_of(sigma_next, eta) if use_noise else sigma_next
            z = (torch.cat([
                    torch.randn((1, *x.shape[1:]), generator=g_, device=x.device, dtype=x.dtype)
                    for g_ in gens
                ], dim=0) if use_noise else None)
            
            if calculate_type == "euler":
                x = euler(x, v, sigma, sigma_lo)
                if use_noise:
                    x = renoise(x, z, sigma_lo, sigma_next)

            elif calculate_type == "heun":
                if last:
                    x = euler(x, v, sigma, sigma_next)
                else:
                    x_e = euler(x, v, sigma, sigma_lo)
                    cfg_i2 = cfg_scale if float(sigma_lo) > cfg_min_sigma else 1.0
                    v2 = velocity_predict(x=x_e, sigma=sigma_lo, **{**kw, "cfg_scale": cfg_i2})
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

            # latents.append(x.detach().float().cpu())
    finally:
        torch.cuda.empty_cache()

    return x, latents, sigmas_steps