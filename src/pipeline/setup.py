import random
import torch
import numpy as np

from src.external import TE
from src.core import Config
from src.sigma import SigmaScheduler

from .types import SamplingInput, InitOutput

@torch.inference_mode()
def setup_init(model:TE, config:Config, device, sigma_schedular:SigmaScheduler, init_input:SamplingInput) -> InitOutput:
    tags_batch = init_input.tags
    init_prompt_batch = config.init_prompt * len(init_input.tags)
    text_batch = tags_batch + init_prompt_batch

    emb, mask = model.encoder(captions=text_batch)
    cond_emb, uncond_emb = emb[:B], emb[B:]
    cond_mask, uncond_mask = mask[:B], mask[B:]

    seed = random.randint(0, 2**32 - 1) if init_input.seed == -1 else init_input.seed
    rng = random.Random(seed)

    seeds = [seed] + [rng.randint(0, 2**32 - 1) for _ in range(B - 1)]
    gens = [torch.Generator(device=device).manual_seed(s) for s in seeds]

    f = config.compression_ratio * config.patch_size

    noise = torch.cat([
        torch.randn(
            (1, config.in_channels, init_input.H // f, init_input.W // f),
            generator=g_,
            device=device,
            dtype=torch.float32,
        )
        for g_ in gens
    ], dim=0)

    u = np.linspace(0.0, 1.0, init_input.steps + 1)
    sigmas = sigma_schedular.transform(u)[::-1].copy()

    return InitOutput(cond_emb, uncond_emb, cond_mask, uncond_mask, noise, gens, sigmas, seeds)