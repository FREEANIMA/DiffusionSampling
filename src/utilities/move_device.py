from contextlib import contextmanager

@contextmanager
def gpu_context(module, device="cuda"):
    module.to(device=device)
    try:
        yield module
    finally:
        module.to(device="cpu")