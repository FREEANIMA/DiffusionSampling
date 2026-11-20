import math
from contextlib import nullcontext

import torch
from tqdm.auto import tqdm
from .solvers import euler, heun, ddpm_plus, renoise, sigma_lo_of
import common.globals as g


# ---------------------------------------------------------------------------
# sigma 구간 판정 유틸 (스케줄 최대 sigma 기준 비율로 판정)
# ---------------------------------------------------------------------------
def _sigma_ratio(sigma, sigma_max):
    return float(sigma) / max(float(sigma_max), 1e-8)


@torch.inference_mode()
def velocity_predict(
    model, x, sigma, cond_emb, cond_mask, uncond_emb, uncond_mask,
    cfg_scale, dtype,
    *,
    sigma_max=None,
    cfg_hi_ratio=0.55,      # 이 비율 이상이면 high sigma
    cfg_hi_factor=0.5,      # high sigma에서 CFG를 절반만
):
    x_in = x.to(dtype)
    use_cfg = cfg_scale != 1.0
    B = x_in.shape[0]

    def _rep(t):
        return t.expand(B, *t.shape[1:]) if t.shape[0] == 1 else t

    cond_emb, cond_mask = _rep(cond_emb), _rep(cond_mask)
    uncond_emb, uncond_mask = _rep(uncond_emb), _rep(uncond_mask)

    # ---- high sigma 판정 ----
    is_high = False
    if sigma_max is not None:
        is_high = _sigma_ratio(sigma, sigma_max) >= cfg_hi_ratio

    # ---- high sigma에서 CFG 포화 방지 ----
    cfg_eff = cfg_scale
    if use_cfg and is_high:
        cfg_eff = 1.0 + cfg_hi_factor * (cfg_scale - 1.0)

    def fwd(x_, emb, mask):
        sig = sigma.expand(x_.shape[0])
        fn = model.forward_cpu if g.INIT_DEVICE == "cpu" else model.forward
        out, _ = fn(x=x_, sigma=sig, txt_emb=emb, txt_mask=mask)
        return out.float()

    # autocast는 항상 켬 (bf16). autocast를 끄면 입력/가중치 dtype 불일치 발생.
    ctx = (
        torch.amp.autocast(device_type=x.device.type, dtype=dtype)
        if x.device.type == "cuda"
        else nullcontext()
    )

    with ctx:
        if use_cfg and cfg_eff != 1.0:
            out = fwd(
                torch.cat([x_in, x_in], dim=0),
                torch.cat([cond_emb, uncond_emb], dim=0),
                torch.cat([cond_mask, uncond_mask], dim=0),
            )
            c, u = out.chunk(2, dim=0)
            v = u + cfg_eff * (c - u)
        else:
            v = fwd(x_in, cond_emb, cond_mask)

    return v


@torch.inference_mode()
def steps_sampling(
    model, x, sigmas, calculate_type, cfg_scale,
    cond_emb, cond_mask, uncond_emb, uncond_mask,
    gens, dtype, eta,
    after_step=None,
    eta_sigma_min=0.4,
    cfg_min_sigma=0.0,
    *,
    # ---- 신규 옵션 ----
    sigma_max=None,             # None이면 sigmas[0] 사용
    eta_hi_boost=1.6,           # high sigma에서 eta 증폭
    eta_hi_ratio=0.55,          # high sigma 판정 비율
    renoise_ratio=0.92,         # high sigma에서 renoise를 sigma_next*0.92로
    renoise_hi_ratio=0.5,       # 이 비율 이상에서만 renoise 축소 적용
    cfg_hi_factor=0.5,          # high sigma CFG 감쇠
    ddpm_use_boost=False,       # ddpm_plus 경로에 eta_i(증폭) 적용 여부
    debug_probe=False,          # |Δv|, |Δx| 로그 (기본 OFF)
    debug_every=1,              # 프로브 출력 주기
):
    model.eval()
    n = len(sigmas) - 1
    if sigma_max is None:
        sigma_max = float(sigmas[0])

    kw_base = dict(
        model=model, cond_emb=cond_emb, cond_mask=cond_mask,
        uncond_emb=uncond_emb, uncond_mask=uncond_mask,
        dtype=dtype, sigma_max=sigma_max,
        cfg_hi_factor=cfg_hi_factor,
    )

    latents = []
    sigmas_steps = []

    # 프로브용: GPU 텐서는 저장하지 않고 스칼라만 보관 (VRAM 누수 방지)
    v_mean_prev = None
    x_mean_prev = None

    try:
        for i in tqdm(range(n), desc="sampling"):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            sigmas_steps.append(float(sigma_next))

            r = _sigma_ratio(sigma, sigma_max)
            is_high = r >= eta_hi_ratio

            # ---- CFG: high sigma는 약하게, low sigma는 끔 ----
            if float(sigma) <= cfg_min_sigma:
                cfg_i = 1.0
            elif is_high:
                cfg_i = 1.0 + cfg_hi_factor * (cfg_scale - 1.0)
            else:
                cfg_i = cfg_scale

            v = velocity_predict(x=x, sigma=sigma, cfg_scale=cfg_i, **kw_base)

            last = (i == n - 1)

            # ---- high sigma에서 노이즈 주입 강화 ----
            eta_i = eta
            if is_high:
                eta_i = eta * eta_hi_boost
            if float(sigma) < eta_sigma_min:
                eta_i = 0.0  # low sigma에서는 끔

            use_noise = (eta_i > 0) and (not last)

            if use_noise:
                sigma_lo = sigma_lo_of(sigma_next, eta_i)
            else:
                sigma_lo = sigma_next

            if use_noise and r >= renoise_hi_ratio:
                renoise_target = sigma_next * renoise_ratio
            else:
                renoise_target = sigma_next

            z = (
                torch.cat(
                    [
                        torch.randn(
                            (1, *x.shape[1:]),
                            generator=g_,
                            device=x.device,
                            dtype=x.dtype,
                        )
                        for g_ in gens
                    ],
                    dim=0,
                )
                if use_noise
                else None
            )

            # ---- 솔버 ----
            if calculate_type == "euler":
                x = euler(x, v, sigma, sigma_lo)
                if use_noise:
                    x = renoise(x, z, sigma_lo, renoise_target)

            elif calculate_type == "heun":
                if last:
                    x = euler(x, v, sigma, sigma_next)
                else:
                    x_e = euler(x, v, sigma, sigma_lo)
                    if float(sigma_lo) <= cfg_min_sigma:
                        cfg_i2 = 1.0
                    elif _sigma_ratio(sigma_lo, sigma_max) >= eta_hi_ratio:
                        cfg_i2 = 1.0 + cfg_hi_factor * (cfg_scale - 1.0)
                    else:
                        cfg_i2 = cfg_scale
                    v2 = velocity_predict(
                        x=x_e, sigma=sigma_lo, cfg_scale=cfg_i2, **kw_base
                    )
                    x = heun(x, v, v2, sigma, sigma_lo)
                    del x_e, v2
                if use_noise:
                    x = renoise(x, z, sigma_lo, renoise_target)

            elif calculate_type == "ddpm_plus":
                if z is None:
                    x = euler(x, v, sigma, sigma_next)
                else:
                    eta_ddpm = eta_i if ddpm_use_boost else eta
                    x = ddpm_plus(x, v, z, sigma, sigma_next, eta_ddpm)

            else:
                raise ValueError(f"Unsupported sampling method: {calculate_type}")

            del z

            if after_step is not None:
                x = after_step(x, sigma, sigma_next)

            # ---- 디버그 프로브 (스칼라만, VRAM 누수 없음) ----
            if debug_probe and (i % debug_every == 0):
                v_mean = v.abs().mean().item()
                x_mean = x.abs().mean().item()
                dv = abs(v_mean - v_mean_prev) if v_mean_prev is not None else float("nan")
                dx = abs(x_mean - x_mean_prev) if x_mean_prev is not None else float("nan")
                tqdm.write(
                    f"[{i:02d}] r={r:.2f} amp=on(bf16) "
                    f"σ={float(sigma):.3f}→{float(sigma_next):.3f} "
                    f"cfg={cfg_i:.2f} eta={eta_i:.2f} lo={float(sigma_lo):.3f} "
                    f"rt={float(renoise_target):.3f} |v|={v_mean:.3e} "
                    f"|Δv|={dv:.3e} |Δx|={dx:.3e}"
                )
                v_mean_prev = v_mean
                x_mean_prev = x_mean

            # latents.append(x.detach().float().cpu())
    finally:
        torch.cuda.empty_cache()

    return x, latents, sigmas_steps