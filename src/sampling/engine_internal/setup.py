import random
import torch
import numpy as np

from src.external import TE
from src.core import Config
from .types import Txt2ImgInput, InitOutput
from .trajectory import sigma_cdf, cfg_schedular, eta_schedular

@torch.inference_mode()
def setup_init(model:TE, config:Config, txt2img_input:Txt2ImgInput) -> InitOutput:
    B = len(txt2img_input.tags)
    tags_batch = txt2img_input.tags
    init_prompt_batch = [config.init_prompt] * len(txt2img_input.tags)
    text_batch = tags_batch + init_prompt_batch

    emb, mask = model.encoder(captions=text_batch)
    cond_emb, uncond_emb = emb[:B], emb[B:]
    cond_mask, uncond_mask = mask[:B], mask[B:]

    seed = random.randint(0, 2**32 - 1) if txt2img_input.seed == -1 else txt2img_input.seed
    rng = random.Random(seed)

    seeds = [seed] + [rng.randint(0, 2**32 - 1) for _ in range(B - 1)]
    gens = [torch.Generator(device=config.device).manual_seed(s) for s in seeds]

    f = config.compression_ratio * config.patch_size

    noise = torch.cat([
        torch.randn(
            (1, config.in_channels, txt2img_input.H // f, txt2img_input.W // f),
            generator=g_,
            device=config.device,
            dtype=torch.float32,
        )
        for g_ in gens
    ], dim=0)

    u = np.linspace(0.0, 1.0, txt2img_input.steps + 1)
    sigmas = sigma_cdf(u, config)[::-1].copy()

    c = np.linspace(0.0, 1.0, txt2img_input.steps + 1)
    cfgs = cfg_schedular(c, txt2img_input.cfg_scale, config)[::-1].copy()

    d = np.linspace(0.0, 1.0, txt2img_input.steps + 1)
    etas = eta_schedular(d, txt2img_input.eta, config)[::-1].copy()

    return InitOutput(cond_emb, uncond_emb, cond_mask, uncond_mask, noise, gens, sigmas, seeds, cfgs, etas)