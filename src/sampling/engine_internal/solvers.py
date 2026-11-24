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