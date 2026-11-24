import os
from dataclasses import fields

import torch
from src.core import state, gpu_clone, to_cpu, ModelSettings
from src.network import Model

def atomic_save(data, path):
    tmp_path = path + ".tmp"

    try:
        with open(tmp_path, "wb") as f:
            torch.save(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def snapshot_checkpoint(model:Model, ema, accum_step, optimizer, on_gpu=False) -> tuple[dict, dict]:
    copy = gpu_clone if on_gpu else to_cpu
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_keys = [f.name for f in fields(ModelSettings)]
    model_cfg = {k: getattr(state.config, k) for k in model_keys}
    raw_ema = ema.shadow._orig_mod if hasattr(ema.shadow, "_orig_mod") else ema.shadow
    
    model_ckpt = {
        "step": accum_step,
        "model_state_dict": copy(raw_model.state_dict()),
        "optimizer_state_dict": copy(optimizer.state_dict()),
        "ema_state_dict" : copy(raw_ema.state_dict()),
        "ema_step" : ema.step,
        "config": model_cfg,
    }
    ema_model_ckpt = {
        "ema_state_dict" : model_ckpt["ema_state_dict"],
        "config": model_ckpt["config"],
    }

    if on_gpu and torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    
    return model_ckpt, ema_model_ckpt

def model_save(model_ckpt, ema_model_ckpt) -> tuple[str, str]:
    model_path = os.path.join(state.config.CKPT_LOCAL_DIR, state.config.MODEL_FILENAME)
    ema_model_path = os.path.join(state.config.CKPT_LOCAL_DIR, state.config.EMA_FILENAME)

    atomic_save(model_ckpt, model_path)
    atomic_save(ema_model_ckpt, ema_model_path)

    return model_path, ema_model_path

