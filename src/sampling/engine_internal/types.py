import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass

@dataclass
class InitInput:
    prompts: list[str]
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
    seeds: list[int]
    sigmas: np.ndarray
    cfgs: np.ndarray
    etas: np.ndarray

@dataclass
class InpaintInput():
    img: Image.Image
    start_t: float
    mask: Image.Image

@dataclass
class Txt2ImgInput(InitInput):
    cfg_scale: float
    calculate_type: str
    eta: float
    sigma_scheduler: np.ndarray
    cfg_scheduler: np.ndarray
    eta_scheduler: np.ndarray
    video_save: bool

    def __post_init__(self):
        self.sigma_scheduler = np.asarray(self.sigma_scheduler, dtype=np.float32)
        self.cfg_scheduler = np.asarray(self.cfg_scheduler, dtype=np.float32)
        self.eta_scheduler = np.asarray(self.eta_scheduler, dtype=np.float32)
    
@dataclass
class InpaintCtx:
    z0: torch.Tensor
    mask: torch.Tensor