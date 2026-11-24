import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class InitInput:
    tags: list[str]
    steps: int
    seed: int
    H: int
    W: int
    B: int

@dataclass
class InitOutput:
    cond_emb: torch.Tensor
    uncond_emb: torch.Tensor
    cond_mask: torch.Tensor
    uncond_mask: torch.Tensor
    noise: torch.Tensor
    gens: list[torch.Generator]
    sigmas: np.ndarray
    seeds: list[int]

@dataclass
class InpaintInput():
    img_bytes: bytes
    start_t: float
    mask: bytes | None = None

@dataclass
class Txt2ImgInput(InitInput):
    final_tags: list[str]
    cfg_scale: float
    calculate_type: str
    eta: float
    eta_mode: str
    video_save: bool
