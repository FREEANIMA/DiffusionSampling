import math

def euler(x, v, sigma, sigma_next):
    return x + (sigma_next - sigma) * v

def heun_step(model_fn, x, sigma, sigma_next):
    v1 = model_fn(x, sigma)
    x_euler = x + (sigma_next - sigma) * v1
    if sigma_next == 0:
        return x_euler
    v2 = model_fn(x_euler, sigma_next)
    return x + (sigma_next - sigma) * 0.5 * (v1 + v2)

def midpoint_step(model_fn, x, sigma, sigma_next):
    sigma_mid = 0.5 * (sigma + sigma_next)
    v1 = model_fn(x, sigma)
    x_mid = x + (sigma_mid - sigma) * v1
    v2 = model_fn(x_mid, sigma_mid)
    return x + (sigma_next - sigma) * v2

# === DPM++ 2M (x0 기반) ===
def dpmpp_2m(x, x0, x0_prev, sigma, sigma_next, h_prev):
    """
    DPM++ 2M 한 스텝.
    x0     : 현재 스텝의 x0 예측 (x - σ·v)
    x0_prev: 이전 스텝의 x0 예측
    h_prev : 이전 스텝의 h (None이면 1차 Euler와 동일)
    반환   : (x_next, h)
    """
    alpha     = 1.0 - sigma
    alpha_n   = 1.0 - sigma_next
    ratio     = sigma_next / sigma

    D = x0
    h = None
    if 0.0 < sigma < 1.0 and sigma_next > 0.0:
        h = math.log(alpha_n / sigma_next) - math.log(alpha / sigma)
        if x0_prev is not None and h_prev is not None and h_prev > 0 and h != 0:
            r = h_prev / h
            D = (1.0 + 1.0 / (2.0 * r)) * x0 - (1.0 / (2.0 * r)) * x0_prev

    x_next = ratio * x + (alpha_n - alpha * ratio) * D
    return x_next, h