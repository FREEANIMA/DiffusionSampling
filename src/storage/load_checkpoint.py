from typing import Any
import os
import torch

from src.core import log, Config
from .hub import download_model
from src.network import MMDIT, LLM

def load_checkpoint_bin(config:Config, filename:str, path=None, console=None) -> dict[str, Any] | None:
    if path is None or not os.path.isfile(path):
        log(console, "Downloading Model to local from hugging face")
        path = download_model(config=config, filename=filename)

    if path is None:
        log(console, "Model not found")
        return None
    
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)

    return ckpt

def load_ema_only(config: Config, network, console=None) -> MMDIT:
    path = os.path.join(config.CKPT_LOCAL_DIR, config.EMA_FILENAME)
    ckpt = load_checkpoint_bin(config=config, path=path, filename=config.EMA_FILENAME, console=console)
    if ckpt is None:
        msg = "Model not found on Hugging Face. Please contact the administrator."
        log(console, msg)
        raise FileNotFoundError(msg)

    sd = ckpt.get("ema_state_dict")
    if sd is None:
        msg = "'ema_state_dict' not found in checkpoint. Please contact the administrator."
        log(console, msg)
        raise KeyError(msg)

    model: MMDIT = network(**ckpt["config"])
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()})

    return model.to(device=config.device, dtype=config.torch_dtype).eval()

def load_llm_model(config: Config, network:LLM, vocab_size, console=None) -> LLM:
    path = os.path.join(config.CKPT_LOCAL_DIR, config.LLM_FILENAME)
    ckpt = load_checkpoint_bin(config=config, path=path, filename=config.LLM_FILENAME, console=console)
    if ckpt is None:
        msg = "Model not found on Hugging Face. Please contact the administrator."
        log(console, msg)
        raise FileNotFoundError(msg)

    config = ckpt["config"]
    config = {
        **config,
        "vocab_size" : vocab_size
    }

    model: LLM = network(**config)
    model.load_state_dict(ckpt["model"])

    model.eval()

    return model.to(device=config.device, dtype=config.torch_dtype).eval()