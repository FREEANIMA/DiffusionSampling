import torch
from tqdm.auto import tqdm
from .solvers import euler, heun, ddim, ddpm_plus

@torch.inference_mode()
def velocity_predict(model, x, sigma, cond_emb, cond_mask, uncond_emb, uncond_mask, cfg_scale, dtype):
    x_in = x.to(dtype)
    sig = sigma.expand(x.shape[0])
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        cond_out, _ = model.forward(
            x=x_in, sigma=sig, txt_emb=cond_emb, txt_mask=cond_mask,
        )
        if cfg_scale == 1.0:
            return cond_out.float()
        uncond_out, _ = model.forward(
            x=x_in, sigma=sig, txt_emb=uncond_emb, txt_mask=uncond_mask,
        )
    c, u = cond_out.float(), uncond_out.float()
    return u + cfg_scale * (c - u)


@torch.inference_mode()
def steps_sampling(model, x, sigmas, calculate_type, cfg_scale, cond_emb, cond_mask, uncond_emb, uncond_mask, gen, dtype, after_step=None):
    model.eval()
    n = len(sigmas) - 1
    kw = dict(model=model, cond_emb=cond_emb, cond_mask=cond_mask,
              uncond_emb=uncond_emb, uncond_mask=uncond_mask,
              cfg_scale=cfg_scale, dtype=dtype)

    v_prev, sigma_prev = None, None   # AB2 이력 (CFG 적용 후 원본 v)

    try:
        for i in tqdm(range(n), desc="sampling"):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            v = velocity_predict(x=x, sigma=sigma, **kw)

            # 마지막 스텝(sigma_next=0)은 h가 커서 외삽이 과해질 수 있어 AB2 끔
            last = (i == n - 1)
            vp = None if last else v_prev
            sp = None if last else sigma_prev

            if calculate_type == "euler":
                x = euler(x, v, sigma, sigma_next, v_prev=vp, sigma_prev=sp)
            elif calculate_type == "ddim":
                x = ddim(x, v, sigma, sigma_next, v_prev=vp, sigma_prev=sp)
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
                x = ddpm_plus(x, v, z, sigma, sigma_next, v_prev=vp, sigma_prev=sp)
                del z
            else:
                raise ValueError(f"Unsupported sampling method: {calculate_type}")

            v_prev, sigma_prev = v, sigma   # 외삽 전 원본 v 저장
            if after_step is not None:
                x = after_step(x, sigma, sigma_next)
    finally:
        torch.cuda.empty_cache()

    return x