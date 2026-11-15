import torch

def sigma_schedular(step, device, shift):
    t = torch.linspace(0.999, 1e-3, step + 1, device=device)
    s = shift
    return s * t / (1 + (s - 1) * t)