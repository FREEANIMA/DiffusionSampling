from typing import Optional
import torch
from PIL import Image
from src.generation import Sampling, InPainting, Detailering
from common import T5TextModel, VaeModel, TagGenration, console_q
from networks import load_mmdit_dcae
from collections import deque
import os
import io

def _parse_seed(seed) -> Optional[int]:
    return None if seed in (None, "", -1, "-1") else int(seed)

import threading

class Progress:
    def __init__(self):
        self._lock = threading.Lock()
        self._step, self._total = 0, 0

    def reset(self):
        self.update(0, 0)

    def update(self, step: int, total: int):
        with self._lock:
            self._step, self._total = step, total

    def snapshot(self) -> dict:
        with self._lock:
            return {"step": self._step, "total": self._total}
        
class Handler:
    def __init__(self, base_root: str = ".", device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.base_root = base_root
        self.device = device
        self.dtype = dtype

        self.console = console_q()

        self.model, self.config = load_mmdit_dcae()
        self.model = self.model.to(device=device, dtype=dtype)
        self.text_encoder = T5TextModel(device=device, dtype=dtype, console=self.console)
        self.vae = VaeModel(device=device, dtype=torch.float32, console=self.console)
        self.tag_generation = TagGenration(device=device, dtype=dtype, console=self.console)

        self.f = self.config["compression_ratio"]
        self.patch_size = self.config["patch_size"]

        self._sampler: Optional[Sampling] = None
        self._detailer: Optional[Detailering] = None
        self._inpainter: Optional[InPainting] = None
        self.progress = Progress()

        self.tags_infer_cached = deque(maxlen=64)

    @property
    def detailer(self) -> Detailering:
        if self._detailer is None:
            self._detailer = Detailering(
                base_root=self.base_root, 
                model=self.model,
                text_encoder=self.text_encoder, 
                vae=self.vae, 
                config=self.config,
                console=self.console
            )
        return self._detailer
    
    @property
    def sampler(self) -> Sampling:
        if self._sampler is None:
            self._sampler = Sampling(
                base_root=self.base_root, 
                model=self.model,
                text_encoder=self.text_encoder, 
                vae=self.vae, 
                config=self.config,
                console=self.console
            )
        return self._sampler

    @property
    def inpainter(self) -> InPainting:
        if self._inpainter is None:
            self._inpainter = InPainting(
                base_root=self.base_root, 
                model=self.model,
                text_encoder=self.text_encoder, 
                vae=self.vae, 
                config=self.config,
                console=self.console
            )
        return self._inpainter

    def _snap(self, x: float, unit: int) -> int:
        return max(unit, int(round(x / unit)) * unit)

    def _fit_size(self, H: int, W: int) -> tuple[int, int]:
        unit = self.f * self.patch_size
        if H >= W:
            return 1024, self._snap(W * 1024 / H, unit)
        return self._snap(H * 1024 / W, unit), 1024

    def get_tag_list(self) -> list[str]:
        return self.tag_generation.get_ac_tags()

    def get_tag_history(self) -> list[dict]:
        return list(self.tags_infer_cached)

    def _resolve_tags(self, tags_text, nsfw, H, W, fixed):
        tags_reshape, tags_infer = self.tag_generation.tags_mapping_worker(
            tags=tags_text, nsfw=bool(nsfw), H=H, W=W, fixed=bool(fixed),
        )

        if tags_infer:
            self.tags_infer_cached.append({
                "input": tags_text,
                "tags": list(tags_infer),
            })

        self.console.put(f"final tags : {tags_reshape}")
        return tags_reshape

    def generate_txt2img(self, tags_text:str, nsfw, calc_type, cfg_scale, H, W, steps, seed, fixed):
        self.progress.reset()
        tags_text = tags_text.lower()

        H, W = self._fit_size(int(H), int(W))

        tags_reshape = self._resolve_tags(tags_text, nsfw, H, W, fixed)
        
        return self.sampler.sampling(
            tags=tags_reshape,
            calculate_type=calc_type,
            cfg_scale=float(cfg_scale),
            H=H, W=W,
            steps=int(steps),
            seed=_parse_seed(seed),
            on_step=self.progress.update,
        )
    
    def generate_inpainting(self, tags_text: str, nsfw, calc_type, cfg_scale, steps, seed, img_path, fixed, mask=None, start_t: float = 0.8):
        if img_path is None:
            raise ValueError("경로를 입력해 주세요")
        if mask is None:
            raise ValueError("mask가 필요합니다.")

        self.progress.reset()
        tags_text = tags_text.lower()

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        with Image.open(io.BytesIO(img_bytes)) as im:
            W0, H0 = im.size
        H, W = self._fit_size(H0, W0)

        tags_reshape = self._resolve_tags(tags_text, nsfw, H, W, fixed)

        return self.inpainter.sampling(
            tags=tags_reshape,
            calculate_type=calc_type,
            cfg_scale=float(cfg_scale),
            steps=int(steps),
            seed=_parse_seed(seed),
            mask=mask,
            img_bytes=img_bytes,
            start_t=float(start_t),
            on_step=self.progress.update,
            save_dir=os.path.join(self.base_root, "inpainting")
        )

    def _square_box(self, box, W, H):
        left, top, right, bottom = box
        cx, cy = (left + right) / 2, (top + bottom) / 2
        side = min(max(right - left, bottom - top), W, H)

        left = min(max(cx - side / 2, 0), W - side)
        top = min(max(cy - side / 2, 0), H - side)
        left, top = int(left), int(top)
        return left, top, left + int(side), top + int(side)

    def generate_detailering(self, tags_text: str, nsfw, calc_type, cfg_scale, steps, seed, img_path, fixed, box=None, start_t: float = 0.7):
        SIZE = 1024
        if img_path is None:
            raise ValueError("경로를 입력해 주세요")
        if box is None:
            raise ValueError("영역(box)이 필요합니다.")

        self.progress.reset()
        tags_text = tags_text.lower()

        img = Image.open(img_path).convert("RGB")
        left, top, right, bottom = box
        left, top = max(int(left), 0), max(int(top), 0)
        right, bottom = min(int(right), img.width), min(int(bottom), img.height)
        if right - left < 8 or bottom - top < 8:
            raise ValueError("선택한 영역이 너무 작거나 이미지 밖이에요.")

        box = self._square_box((left, top, right, bottom), img.width, img.height)
        crop_img = img.crop(box).resize((SIZE, SIZE), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        crop_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        H, W = SIZE, SIZE
        tags_reshape = [self.tag_generation.INIT_PROMPT] 

        img_out = self.detailer.sampling(
            tags=tags_reshape,
            calculate_type=calc_type,
            cfg_scale=float(cfg_scale),
            steps=int(steps),
            seed=_parse_seed(seed),
            img_bytes=img_bytes,
            start_t=float(start_t),
            on_step=self.progress.update,
        )

        return self.detailer.merge_img(
            origin_img=img,
            img_out=img_out,
            square=box,
            save_dir=os.path.join(self.base_root, "detailer"),
        )