from dataclasses import dataclass
import torch

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

@dataclass(kw_only=True)
class ModelSettings:
    hidden_size: int
    nhead: int
    num_layers: int
    txt_hidden_size: int
    sigma_emb_hz: int
    rope_hz: int
    sigma_scale: int
    pos_scale: float
    in_channels: int
    compression_ratio: int
    allowable_size: int
    patch_size: int
    dropout: float

@dataclass(kw_only=True)
class SigmaSettings:
    low_sigma: float
    high_sigma: float
    alpha: float
    beta: float
    gamma: list
    zeta: list

@dataclass(kw_only=True)
class PathSettins:
    GENERATION_DIR: str
    DETAIL_DIR: float
    ERROR_REPORT: float
    CKPT_LOCAL_DIR: float

@dataclass(kw_only=True)
class SamplingSettings:
    dtype: str
    device: str

@dataclass(kw_only=True)
class FineTuningSettings:
    NAS: list
    init_prompt: str

@dataclass(kw_only=True)
class Config(ModelSettings, PathSettins, SigmaSettings, FineTuningSettings, SamplingSettings):
    ver_number: int

    def __post_init__(self):
        if self.dtype not in DTYPES:
            raise ValueError(f"unknown dtype: {self.dtype}")

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]