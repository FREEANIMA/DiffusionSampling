import torch
from PIL import Image

from .types import SamplingInput, InitOutput
from .setup import setup_init

from src.core import log
from src.external import TE, VAE
from src.network import MMDIT, LLM

from .steps import sampling

class Generation:
    def __init__(self, model:MMDIT, llm:LLM, te:TE, vae:VAE, console=None):
        self.console = console
        self.te = te
        self.vae = vae
        self.llm = llm
        self.model = model

    @torch.inference_mode()
    def vae_decode(self, x):
        decoded = self.vae.decoder(x)
        img = self.vae._tensor_to_pils(decoded)
        return img

    @torch.inference_mode()
    def sampling(self, sampling_input:SamplingInput) -> tuple[list[Image.Image], list[Image.Image], list[float]]:
        init_out:InitOutput = setup_init(sampling_input)

        x, latents, sigmas_steps = sampling(
            model=self.model,
            sampling_input=sampling_input,
            init_out=init_out,
            dtype=self.dtype,
        )

        imgs = self.vae_decode(x) 

        frams = []
        if sampling_input.video_save:
            frams.append(self.vae_decode(x=latents))

        return imgs, frams, sigmas_steps