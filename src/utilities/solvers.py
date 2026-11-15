def v_to_x0_eps(x, v, sigma):
    x0  = x - sigma * v
    eps = x + (1.0 - sigma) * v
    return x0, eps

def _ab2(v, v_prev, sigma, sigma_next, sigma_prev):
    """AB2 외삽된 v. 이전 v가 없으면(첫 스텝) 그대로 반환."""
    if v_prev is None or sigma_prev is None:
        return v
    h      = sigma_next - sigma
    h_prev = sigma - sigma_prev
    r = h / h_prev
    return v + 0.5 * r * (v - v_prev)    # (1 + r/2)·v - (r/2)·v_prev

def euler(x, v, sigma, sigma_next, v_prev=None, sigma_prev=None):
    v = _ab2(v, v_prev, sigma, sigma_next, sigma_prev)
    return x + (sigma_next - sigma) * v

def heun(x, v1, v2, sigma, sigma_next, v_prev=None, sigma_prev=None):
    # heun은 이미 v를 2번 평가하므로 AB2 대상이 아님: 기존과 동일
    return x + (sigma_next - sigma) * 0.5 * (v1 + v2)

def ddim(x, v, sigma, sigma_next, v_prev=None, sigma_prev=None):
    v = _ab2(v, v_prev, sigma, sigma_next, sigma_prev)
    x0, eps = v_to_x0_eps(x, v, sigma)
    return (1.0 - sigma_next) * x0 + sigma_next * eps

def ddpm_plus(x, v, z, sigma, sigma_next, eta=1.0, v_prev=None, sigma_prev=None):
    v = _ab2(v, v_prev, sigma, sigma_next, sigma_prev)
    x0, eps = v_to_x0_eps(x, v, sigma)

    sigma_up = eta * sigma_next * (1.0 - (sigma_next / sigma) ** 2).clamp(min=0).sqrt()
    sigma_down = (sigma_next ** 2 - sigma_up ** 2).clamp(min=0).sqrt()

    alpha_next = 1.0 - sigma_next
    alpha_down = 1.0 - sigma_down
    ratio = alpha_next / alpha_down.clamp(min=1e-8)

    x_down = (1.0 - sigma_down) * x0 + sigma_down * eps
    return ratio * x_down + sigma_up * z