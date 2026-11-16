import torch

def _mem(tag):
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    p = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{tag:35s}] alloc={a:.2f}GB  reserved={r:.2f}GB  peak_alloc={p:.2f}GB")