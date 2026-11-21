import math
import torch
from tqdm.auto import tqdm
from .solvers import euler, heun, renoise, sigma_lo_of, ab2
import common.globals as g

@torch.inference_mode()
def velocity_predict(model, x, sigma, cond_emb, cond_mask, uncond_emb, uncond_mask, cfg_scale, dtype, max_side):
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
        out, _ = fn(x=x_, sigma=sig, txt_emb=emb, txt_mask=mask, max_side=max_side)
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

def eta_at(progress, eta, power=1.0):
    x = min(max(float(progress), 0.0), 1.0)
    w = math.log(2.0 / (x + 1.0)) / math.log(2.0)
    return eta * (w ** power)

def eta_at_cos(sigma_next, eta, mu=0.6, half_width=0.5, floor=0.5, sigma_fade=0.15):
    sn = float(sigma_next)
    d = min(abs(sn - mu) / half_width, 1.0)
    bump = 0.5 * (1.0 + math.cos(math.pi * d))
    w = floor + (1.0 - floor) * bump
    t = min(max(sn / sigma_fade, 0.0), 1.0)
    fade = 0.5 * (1.0 - math.cos(math.pi * t)) 
    return eta * w * fade

def eta_at_custom_sin(sigma_next, eta, mu=0.6):
    y = (math.sin(math.pi * sigma_next - mu) - math.sin(-mu)) / (1.0 - math.sin(-mu))

    return eta * y

@torch.inference_mode()
def steps_sampling(model, x, sigmas, calculate_type, cfg_scale, 
                   cond_emb, cond_mask, uncond_emb, uncond_mask, max_side,
                   gens, dtype, eta, sigma_mean, eta_mode="cos", 
                   after_step=None, cfg_min_sigma=0.0, debug_probe=False, eta_power=1.0):
    model.eval()
    n = len(sigmas) - 1
    kw = dict(model=model, cond_emb=cond_emb, cond_mask=cond_mask,
              uncond_emb=uncond_emb, uncond_mask=uncond_mask,
              cfg_scale=cfg_scale, dtype=dtype, max_side=max_side)

    latents = []
    sigmas_steps = []
    prev_v = None
    hist_v, hist_h = None, None

    try:
        for i in tqdm(range(n), desc="sampling"):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]

            sigmas_steps.append(float(sigma_next))
            cfg_i = cfg_scale if float(sigma) > cfg_min_sigma else 1.0
            kw_i = {**kw, "cfg_scale": cfg_i}
            v = velocity_predict(x=x, sigma=sigma, **kw_i)
            last = (i == n - 1)

            x0_hat = (x - float(sigma) * v).detach().float().cpu()

            progress = i / max(n - 1, 1)

            if eta_mode == "cos":
                eta_i = eta_at_cos(float(sigma_next), eta, mu=sigma_mean)
            elif eta_mode == "ln":
                eta_i = eta_at(progress, eta, power=eta_power)
            elif eta_mode == "custom_sin":
                eta_i = eta_at_custom_sin(sigma_next, eta, sigma_mean)
                
            use_noise = eta_i > 1e-4 and not last
            sigma_lo = sigma_lo_of(sigma_next, eta_i) if use_noise else sigma_next
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
            elif calculate_type == "ab2":
                x = ab2(x, v, hist_v, sigma, sigma_lo, hist_h)
                if use_noise:
                    x = renoise(x, z, sigma_lo, sigma_next)
                    hist_v, hist_h = None, None          # renoise 후엔 외삽 금지
                else:
                    hist_v, hist_h = v.detach(), float(sigma_lo) - float(sigma)
            else:
                raise ValueError(f"Unsupported sampling method: {calculate_type}")

            del z

            if after_step is not None:
                x = after_step(x, sigma, sigma_next)

            if debug_probe:
                if prev_v is not None:
                    H, W = v.shape[-2:]
                    v_tok = v.abs().mean(dim=1)
                    dv_tok = (v - prev_v).abs().mean(dim=1)
                    rel_tok = dv_tok / (v_tok + 1e-8)

                    for b in range(min(v.shape[0], 2)):
                        flat = rel_tok[b].flatten()
                        med = flat.median().item()
                        vals, idxs = flat.topk(5, largest=False)
                        pos = " ".join(
                            f"({idx // W},{idx % W}):{val:.1e}"
                            for val, idx in zip(vals.tolist(), idxs.tolist())
                        )
                        vmin, vidx = v_tok[b].flatten().min(dim=0)
                        tqdm.write(
                            f"[{i:02d}] a={float(sigma):.3f} eta={eta_i:.4f} cfg={cfg_i:.1f} b{b} "
                            f"med={med:.1e} low5(row,col):dv_rel= {pos} | "
                            f"min|v|={vmin.item():.1e}@({vidx.item() // W},{vidx.item() % W})"
                        )
                prev_v = v.detach()

            latents.append(x0_hat)
    finally:
        torch.cuda.empty_cache()

    return x, latents, sigmas_steps