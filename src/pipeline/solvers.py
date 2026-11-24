import torch

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

def rk4_step(model_fn, x, sigma, sigma_next):
    dt = sigma_next - sigma
    half_dt = 0.5 * dt
    sigma_mid = sigma + half_dt

    k1 = model_fn(x, sigma)
    x2 = x + half_dt * k1
    k2 = model_fn(x2, sigma_mid)

    x3 = x + half_dt * k2
    k3 = model_fn(x3, sigma_mid)

    x4 = x + dt * k3
    if sigma_next == 0:
        return x + dt * ((k1 + 2 * k2 + 2 * k3 + k4) / 6.0)

    k4 = model_fn(x4, sigma_next)
    return x + dt * ((k1 + 2 * k2 + 2 * k3 + k4) / 6.0)