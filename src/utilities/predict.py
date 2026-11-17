import warnings
import torch
from tqdm.auto import tqdm
from .solvers import euler, heun, ddim, ddpm_plus, unipc
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
def steps_sampling(model, x, sigmas, calculate_type, cfg_scale, cond_emb, cond_mask, uncond_emb, uncond_mask, gen, dtype, after_step=None, unipc_order=2, unipc_variant="bh2", unipc_corrector=True):
    model.eval()
    n = len(sigmas) - 1
    kw = dict(model=model, cond_emb=cond_emb, cond_mask=cond_mask,
              uncond_emb=uncond_emb, uncond_mask=uncond_mask,
              cfg_scale=cfg_scale, dtype=dtype)

    if calculate_type == "unipc":
        if after_step is not None:
            warnings.warn("unipc 경로에서는 after_step이 호출되지 않습니다.")
        pbar = tqdm(total=n, desc="sampling")

        def model_fn(xx, s):
            sig_t = torch.tensor(s, device=xx.device, dtype=torch.float32)
            v = velocity_predict(x=xx, sigma=sig_t, **kw)
            pbar.update(1)
            return v

        try:
            x, latents, sigmas_steps = unipc(model_fn, x, [float(s) for s in sigmas],
                                             order=unipc_order, variant=unipc_variant,
                                             use_corrector=unipc_corrector)
        finally:
            pbar.close()
            torch.cuda.empty_cache()
        return x, latents, sigmas_steps
    
    latents = []
    sigmas_steps = []

    try:
        for i in tqdm(range(n), desc="sampling"):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]

            if 0.6 < sigma < 0.9:
                bias = 1e-3
                x = x + bias 

            sigmas_steps.append(float(sigma_next))
            v = velocity_predict(x=x, sigma=sigma, **kw)

            last = (i == n - 1)

            if calculate_type == "euler":
                x = euler(x, v, sigma, sigma_next)
            elif calculate_type == "ddim":
                x = ddim(x, v, sigma, sigma_next)
            elif calculate_type == "heun":
                if last:
                    x = euler(x, v, sigma, sigma_next)
                else:
                    x_e = euler(x, v, sigma, sigma_next)
                    v2 = velocity_predict(x=x_e, sigma=sigma_next, **kw)
                    x = heun(x, v, v2, sigma, sigma_next)
                    del x_e, v2
            elif calculate_type == "ddpm_plus":
                z = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
                x = ddpm_plus(x, v, z, sigma, sigma_next)
                del z
            else:
                raise ValueError(f"Unsupported sampling method: {calculate_type}")

            if after_step is not None:
                x = after_step(x, sigma, sigma_next)

            latents.append(x.detach().float().cpu())
    finally:
        torch.cuda.empty_cache()

    return x, latents, sigmas_steps