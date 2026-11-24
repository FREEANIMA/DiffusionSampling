import os
import time
import traceback
import torch
import contextlib
from src.core import state, shared

def log(console, msg):
    if console is None:
        print(msg)
    else:
        console.put(msg)

def error_report(error=None, name=None):
    tag = f"{name}_" if name else ""
    path = os.path.join(shared.BASE_ROOT, "error_report", f"error_{tag}{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000}.txt")

    body = traceback.format_exc()
    if body.strip() == "NoneType: None":
        body = ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{type(error).__name__ if error else ''}: {error if error else ''}\n\n{body}")
    return path

@contextlib.contextmanager
def vram_use(name, store, vram_check_use):
    if not vram_check_use:
        yield
        return

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        delta = (torch.cuda.memory_allocated() - base) / 1e9
        extra = peak - base / 1e9
        prev = store.get(name)
        if prev is None or peak >= prev["peak_gb"]:
            store[name] = {
                "peak_gb": peak,
                "delta_gb": delta,
                "extra_gb": extra,
            }

@contextlib.contextmanager
def timer(label, store: dict, sync: bool = True):
    if sync:
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if sync:
            torch.cuda.synchronize()
        store[label] = time.perf_counter() - start

def train_profiled(name, enabled):
    def deco(fn):
        return vram_use(name, state.vram, enabled)(timer(name, state.timings)(fn))
    return deco

def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj

def gpu_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: gpu_clone(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(gpu_clone(v) for v in obj)
    return obj