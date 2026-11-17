import torch

def sigma_schedular(step, device, shift=3.0):
    t = torch.linspace(1.0, 0.0, step + 1, device=device)

    s = shift
    sigmas = s * t / (1 + (s - 1) * t)
    
    sigmas = sigmas.clamp(0.001, 0.999)
    
    return sigmas