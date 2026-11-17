import torch

def sigma_schedular(step, device, shift=1.0):
    sigma_max = 1.0
    sigma_min = 0.001
    
    rho = 7.0 if shift == 1.0 else shift 
    
    step_indices = torch.arange(step, dtype=torch.float32, device=device)
    
    t = (sigma_max ** (1 / rho) + 
         step_indices / (step - 1) * 
         (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    
    sigmas = torch.cat([t, torch.zeros_like(t[:1])])
    
    return sigmas