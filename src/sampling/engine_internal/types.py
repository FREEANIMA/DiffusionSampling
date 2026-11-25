import torch
import numpy as np
from PIL import Image
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
    cfgs: np.ndarray
    etas: np.ndarray

@dataclass
class InpaintInput():
    img: Image.Image
    start_t: float
    mask: Image.Image

@dataclass
class Txt2ImgInput(InitInput):
    final_tags: list[str]
    cfg_scale: float
    calculate_type: str
    eta: float
    video_save: bool

@dataclass
class InpaintCtx:
    z0: torch.Tensor
    mask: torch.Tensor