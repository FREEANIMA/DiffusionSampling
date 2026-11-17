import math
import torch


def v_to_x0_eps(x, v, sigma):
    x0  = x - sigma * v
    eps = x + (1.0 - sigma) * v
    return x0, eps

def euler(x, v, sigma, sigma_next):
    return x + (sigma_next - sigma) * v

def heun(x, v1, v2, sigma, sigma_next):
    return x + (sigma_next - sigma) * 0.5 * (v1 + v2)

def ddim(x, v, sigma, sigma_next):
    x0, eps = v_to_x0_eps(x, v, sigma)
    return (1.0 - sigma_next) * x0 + sigma_next * eps

def ddpm_plus(x, v, z, sigma, sigma_next, eta=0.3):
    x0 = x - sigma * v
    eps = x + (1.0 - sigma) * v

    sigma_lo = sigma_next * (1.0 - eta)
    x_lo = (1.0 - sigma_lo) * x0 + sigma_lo * eps

    a = (1.0 - sigma_next) / (1.0 - sigma_lo)
    noise_scale = (sigma_next ** 2 - (a * sigma_lo) ** 2).clamp(min=0).sqrt() if torch.is_tensor(sigma_next) \
        else max(sigma_next ** 2 - (a * sigma_lo) ** 2, 0.0) ** 0.5

    return a * x_lo + noise_scale * z

def _unipc_rhos_b(rks, B, hh, phi1):
    """UniPC 선형계 R, b 구성 (rks 마지막 원소는 1.0)."""
    rk_t = torch.tensor(rks, dtype=torch.float64)
    R = torch.stack([rk_t ** j for j in range(len(rks))])
    b, hk, fac = [], phi1 / hh - 1.0, 1
    for j in range(1, len(rks) + 1):
        b.append(hk * fac / B)
        fac *= (j + 1)
        hk = hk / hh - 1.0 / fac
    return R, torch.tensor(b, dtype=torch.float64)


@torch.inference_mode()
def unipc(model_fn, x, sigmas, order=2, variant="bh2", use_corrector=True):
    sigmas = [float(s) for s in sigmas]
    n = len(sigmas) - 1
    # 수식용 sigma: 첫 값 1.0 방지(lambda=-inf), 마지막은 0
    sig = [min(max(s, 1e-4), 1.0 - 1e-4) for s in sigmas[:-1]] + [0.0]
    lam = [math.log((1.0 - s) / s) if s > 0 else float("inf") for s in sig]

    ms, lams = [], []      # x0 예측 / lambda 히스토리 (오래된 것 -> 최신)
    x_prev = None          # 직전 predictor의 입력
    ord_prev = 1

    for i in range(n):
        s, sn = sig[i], sig[i + 1]
        last = (sn == 0.0)

        v = model_fn(x, sigmas[i])
        m = x - sigmas[i] * v                      # x0 예측

        # ---------- corrector: x_prev -> x(s) 를 새 m 포함해 재계산 ----------
        if use_corrector and i > 0 and not last:
            sp = sig[i - 1]
            h = lam[i] - lam[i - 1]
            hh = -h
            phi1 = math.expm1(hh)
            B = hh if variant == "bh1" else math.expm1(hh)

            rks, D1s = [], []
            for k in range(1, ord_prev):
                rk = (lams[-(k + 1)] - lam[i - 1]) / h
                rks.append(rk)
                D1s.append((ms[-(k + 1)] - ms[-1]) / rk)
            rks.append(1.0)

            if len(rks) == 1:
                rho = [0.5]
            else:
                R, b = _unipc_rhos_b(rks, B, hh, phi1)
                rho = torch.linalg.solve(R, b).tolist()

            x_t_ = (s / sp) * x_prev - (1.0 - s) * phi1 * ms[-1]
            res = sum(rho[j] * D1s[j] for j in range(len(D1s))) if D1s else 0.0
            x = x_t_ - (1.0 - s) * B * (res + rho[-1] * (m - ms[-1]))

        ms.append(m)
        lams.append(lam[i])
        ms, lams = ms[-order:], lams[-order:]

        # ---------- predictor ----------
        if last:                                   # 마지막 스텝: x0로 바로
            x = m
            break

        cur = min(order, i + 1, n - i)             # 초반/마지막 근처는 저차
        h = lam[i + 1] - lam[i]
        hh = -h
        phi1 = math.expm1(hh)
        B = hh if variant == "bh1" else math.expm1(hh)

        rks, D1s = [], []
        for k in range(1, cur):
            rk = (lams[-(k + 1)] - lam[i]) / h
            rks.append(rk)
            D1s.append((ms[-(k + 1)] - ms[-1]) / rk)
        rks.append(1.0)

        x_prev = x
        x_t_ = (sn / s) * x - (1.0 - sn) * phi1 * ms[-1]
        if D1s:
            if cur == 2:
                rho = [0.5]
            else:
                R, b = _unipc_rhos_b(rks, B, hh, phi1)
                rho = torch.linalg.solve(R[:-1, :-1], b[:-1]).tolist()
            x = x_t_ - (1.0 - sn) * B * sum(rho[j] * D1s[j] for j in range(len(D1s)))
        else:
            x = x_t_
        ord_prev = cur

    return x, [], [] 