import torch
from contextlib import contextmanager

@contextmanager
def gpu_context(module, device="cuda"):
    original_device = next(module.parameters()).device

    if original_device != torch.device(device):
        module.to(device=device)

    try:
        yield module
    finally:
        if original_device != torch.device(device):
            module.to(device=original_device)