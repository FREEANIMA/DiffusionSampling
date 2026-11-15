from typing import Optional
import torch
from PIL import Image
from src.generation import Sampling, InPaint
from common import T5TextModel, VaeModel, TagGenration
from networks import load_mmdit_dcae

def _parse_seed(seed) -> Optional[int]:
    return None if seed in (None, "", -1, "-1") else int(seed)

class Handler:
    def __init__(self, base_root: str = ".", device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.base_root = base_root
        self.device = device
        self.dtype = dtype

        self.model, self.config = load_mmdit_dcae()
        self.model = self.model.to(device=device, dtype=dtype)
        self.text_encoder = T5TextModel(device=device, dtype=dtype)
        self.vae = VaeModel(device=device, dtype=torch.float32)
        self.tag_generation = TagGenration(device=device, dtype=dtype)

        self.f = self.config["compression_ratio"]
        self.patch_size = self.config["patch_size"]

        self._sampler: Optional[Sampling] = None
        self._inpainter: Optional[InPaint] = None

    @property
    def sampler(self) -> Sampling:
        if self._sampler is None:
            self._sampler = Sampling(
                base_root=self.base_root, 
                model=self.model,
                text_encoder=self.text_encoder, 
                vae=self.vae, 
                config=self.config,
            )
        return self._sampler

    @property
    def inpainter(self) -> InPaint:
        if self._inpainter is None:
            self._inpainter = InPaint(
                base_root=self.base_root, 
                model=self.model,
                text_encoder=self.text_encoder, 
                vae=self.vae, 
                config=self.config,
            )
        return self._inpainter

    def _snap(self, x: float, unit: int) -> int:
        return max(unit, int(round(x / unit)) * unit)

    def _fit_size(self, H: int, W: int) -> tuple[int, int]:
        """긴 변을 1024로 맞추고, 짧은 변은 unit 배수로 스냅."""
        unit = self.f * self.patch_size
        if H >= W:
            return 1024, self._snap(W * 1024 / H, unit)
        return self._snap(H * 1024 / W, unit), 1024

    def generate_txt2img(self, tags_text, nsfw, calc_type, cfg_scale, H, W, steps, seed):
        H, W = self._fit_size(int(H), int(W))

        tags_reshape = self.tag_generation.tags_mapping_worker(
            tags=tags_text, nsfw=bool(nsfw), H=H, W=W,
        )

        return self.sampler.sampling(
            tags=tags_reshape,
            calculate_type=calc_type,
            cfg_scale=float(cfg_scale),
            H=H, W=W,
            steps=int(steps),
            seed=_parse_seed(seed),
        )

    def generate_inpainting(self, tags_text, nsfw, calc_type, cfg_scale, steps, seed, img_path, start_t: float = 0.9):
        with Image.open(img_path) as im:
            W0, H0 = im.size
        H, W = self._fit_size(H0, W0)

        tags_reshape = self.tag_generation.tags_mapping_worker(
            tags=tags_text, nsfw=bool(nsfw), H=H, W=W,
        )

        return self.inpainter.sampling(
            tags=tags_reshape,
            nsfw=bool(nsfw),
            calculate_type=calc_type,
            cfg_scale=float(cfg_scale),
            steps=int(steps),
            seed=_parse_seed(seed),
            path=img_path,
            start_t=float(start_t),
        )