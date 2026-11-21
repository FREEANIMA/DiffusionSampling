import torch

def ab2(x, v, v_prev, sigma, sigma_next, h_prev, r_max=2.0):
    h = sigma_next - sigma
    if v_prev is None or h_prev is None or h_prev == 0:
        return x + h * v
    r = min(max(h / h_prev, 0.0), r_max)
    return x + h * ((1.0 + 0.5 * r) * v - 0.5 * r * v_prev)

def v_to_x0_eps(x, v, sigma):
    x0 = x - sigma * v
    eps = x + (1.0 - sigma) * v
    return x0, eps

def sigma_lo_of(sigma_next, eta):
    return sigma_next * (1.0 - eta)

def renoise(x_lo, z, sigma_lo, sigma_next):
    a = (1.0 - sigma_next) / (1.0 - sigma_lo)
    var = sigma_next ** 2 - (a * sigma_lo) ** 2
    if torch.is_tensor(var):
        noise_scale = var.clamp(min=0).sqrt()
    else:
        noise_scale = max(var, 0.0) ** 0.5
    return a * x_lo + noise_scale * z

def euler(x, v, sigma, sigma_next):
    return x + (sigma_next - sigma) * v

def euler_ancestral(x, v, z, sigma, sigma_next, eta=0.3):
    if z is None or eta <= 0.0:
        return euler(x, v, sigma, sigma_next)

    sigma_lo = sigma_lo_of(sigma_next, eta)
    x_lo = euler(x, v, sigma, sigma_lo)

    return renoise(x_lo, z, sigma_lo, sigma_next)

def heun(x, v1, v2, sigma, sigma_next):
    return x + (sigma_next - sigma) * 0.5 * (v1 + v2)