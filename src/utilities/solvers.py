import torch


def v_to_x0_eps(x, v, sigma):
    x0 = x - sigma * v
    eps = x + (1.0 - sigma) * v
    return x0, eps


def sigma_lo_of(sigma_next, eta):
    return sigma_next * (1.0 - eta)


def renoise(x_lo, z, sigma_lo, sigma_next):
    """x_lo(노이즈 σ_lo)를 노이즈 σ_next 수준으로 맞추고 부족분을 z로 채움."""
    a = (1.0 - sigma_next) / (1.0 - sigma_lo)
    var = sigma_next ** 2 - (a * sigma_lo) ** 2
    if torch.is_tensor(var):
        noise_scale = var.clamp(min=0).sqrt()
    else:
        noise_scale = max(var, 0.0) ** 0.5
    return a * x_lo + noise_scale * z


def euler(x, v, sigma, sigma_next):
    return x + (sigma_next - sigma) * v


def heun(x, v1, v2, sigma, sigma_next):
    return x + (sigma_next - sigma) * 0.5 * (v1 + v2)


def ddim(x, v, sigma, sigma_next):
    x0, eps = v_to_x0_eps(x, v, sigma)
    return (1.0 - sigma_next) * x0 + sigma_next * eps


def ddpm_plus(x, v, z, sigma, sigma_next, eta=0.3):
    """DDIM으로 σ_lo까지 간 뒤 renoise로 σ_next까지 복원.
    z가 None이면(마지막 스텝) renoise 없이 σ_next까지 DDIM."""
    if z is None:
        return ddim(x, v, sigma, sigma_next)
    sigma_lo = sigma_lo_of(sigma_next, eta)
    x_lo = ddim(x, v, sigma, sigma_lo)
    return renoise(x_lo, z, sigma_lo, sigma_next)