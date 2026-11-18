from typing import Optional
import torch
from PIL import Image
from src.generation import Sampling, InPainting, Detailering
from common import T5TextModel, VaeModel, TagGenration, console_q
from networks import load_mmdit_dcae
from collections import deque
import os
import io
from src.utilities import gpu_context, img_resize
import common.globals as g

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

        quantization = input("use quantization? (y or n): ").strip().lower() == "y"
        position = input("model upload on gpu(y or n): ").strip().lower() == "y"
        self.model, self.config = load_mmdit_dcae(quantization)

        if position:
            init_device = self.device
        else:
            init_device = "cpu"
            num_layers = len(self.model.blocks)
            max_chunk = max(1, num_layers // 4)

            print(f"[info] 1 block ≈ 0.27GB")
            print(f"[info] CHUNK = number of layers loaded onto GPU at once")
            print(f"[info] e.g. CHUNK=4 => 4 layers x {num_layers//4} rounds, "
                f"up to 8 layers (~2.2GB) with prefetch")

            while True:
                raw = input(f"block chunk size (1~{max_chunk}, recommended 4): ").strip()
                if raw.isdigit() and 1 <= int(raw) <= max_chunk:
                    g.BLOCK_CHUNK = int(raw)
                    break
                print(f"Please enter an integer between 1 and {max_chunk}")

        g.INIT_DEVICE = init_device

        self.model = self.model.to(device=init_device, dtype=dtype)
        self.text_encoder = T5TextModel(init_device=init_device, device=self.device, dtype=dtype, console=self.console)
        self.vae = VaeModel(init_device=init_device, device=self.device, dtype=torch.float32, console=self.console)
        self.tag_generation = TagGenration(init_device=init_device, device=self.device, dtype=dtype, console=self.console)

        self.tag_tree = self.tag_generation.tag_tree

        self.f = self.config["compression_ratio"]
        self.patch_size = self.config["patch_size"]
        self.allowable_size = self.config["allowable_size"]

        self._sampler: Optional[Sampling] = None
        self._detailer: Optional[Detailering] = None
        self._inpainter: Optional[InPainting] = None
        self.progress = Progress()

        self.tags_infer_cached = deque(maxlen=64)

    def get_tag_tree(self) -> dict:
        return self.tag_tree

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

    def _resolve_tags(self, tags_text, nsfw, H, W, fixed, record=True, label="final tags"):
        with gpu_context(self.tag_generation.text_model, self.device):
            tags_reshape, final_tags = self.tag_generation.tags_mapping_worker(
                tags=tags_text, nsfw=bool(nsfw), H=H, W=W, fixed=bool(fixed),
            )

        if final_tags and record:
            self.tags_infer_cached.append({
                "input": tags_text,
                "tags": list(final_tags),
            })

        self.console.put(f"{label} : {tags_reshape}")

        return tags_reshape, final_tags

    def generate_txt2img(self, tags_text:str, nsfw, calc_type, cfg_scale, H, W, steps, seed, fixed, shift, eta=0.3):
        self.progress.reset()
        tags_text = tags_text.lower()

        H, W = self._fit_size(int(H), int(W))

        tags_reshape, final_tags = self._resolve_tags(tags_text, nsfw, H, W, fixed)
        
        return self.sampler.sampling(
            tags=tags_reshape,
            final_tags=final_tags,
            calculate_type=calc_type,
            cfg_scale=float(cfg_scale),
            H=H, W=W,
            steps=int(steps),
            seed=_parse_seed(seed),
            on_step=self.progress.update,
            shift=self._clamp_shift(shift),
            eta=eta
        )
    
    def generate_inpainting(self, tags_text: str, nsfw, calc_type, cfg_scale, steps, seed, img_path, fixed, shift, eta, mask=None, start_t: float = 0.8):
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

        tags_reshape, final_tags = self._resolve_tags(tags_text, nsfw, H, W, fixed)

        return self.inpainter.sampling(
            tags=tags_reshape,
            final_tags=final_tags,
            calculate_type=calc_type,
            cfg_scale=float(cfg_scale),
            steps=int(steps),
            seed=_parse_seed(seed),
            mask=mask,
            img_bytes=img_bytes,
            start_t=float(start_t),
            on_step=self.progress.update,
            save_dir=os.path.join(self.base_root, "inpainting"),
            shift=self._clamp_shift(shift),
            eta=eta
        )

    def _clamp_shift(self, shift) -> float:
        try:
            s = float(shift)
        except (TypeError, ValueError):
            s = 3.0
        return min(max(s, 0.1), 10.0)

    def generate_detailering(self, calc_type,  steps, seed, img_path, shift, tags_text : str, cfg_scale, nsfw, eta, mask=None, fixed=True, box=None, start_t: float = 0.3):
        if img_path is None:
            raise ValueError("경로를 입력해 주세요")
        if box is None:
            raise ValueError("영역(box)이 필요합니다.")

        self.progress.reset()
        img = Image.open(img_path).convert("RGB")

        unit = self.f * self.patch_size
        left, top, right, bottom = box
        left, top = max(int(left), 0), max(int(top), 0)
        right, bottom = min(int(right), img.width), min(int(bottom), img.height)

        w = (right - left) // unit * unit
        h = (bottom - top) // unit * unit

        if w < unit or h < unit:
            raise ValueError(f"선택한 영역이 너무 작거나 이미지 밖이에요. (최소 {unit}px)")

        box = (left, top, left + w, top + h)
        crop_img = img.crop(box)

        buf = io.BytesIO()
        crop_img.save(buf, format="PNG")

        w, h = crop_img.size

        scale = self.allowable_size / max(w, h)

        unit = (self.patch_size * self.f)

        W = max(unit, round(w * scale / unit) * unit)
        H = max(unit, round(h * scale / unit) * unit)

        if w >= h:
            W = self.allowable_size
        else:
            H = self.allowable_size

        crop_img = crop_img.resize((W, H), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        crop_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        tags_text = tags_text.strip().lower()

        arr, new_H, new_W = img_resize(img_bytes, self.allowable_size, self.patch_size, self.f)

        assert H == new_H, f"Height mismatch: expected={H}, actual={new_H}"
        assert W == new_W, f"Width mismatch: expected={W}, actual={new_W}"

        tags_reshape, final_tags = self._resolve_tags(tags_text, nsfw, new_H, new_W, fixed)
        img_out, imgs, sigmas_steps = self.detailer.sampling(
            tags=tags_reshape,
            calculate_type=calc_type,
            cfg_scale=float(cfg_scale),
            steps=int(steps),
            seed=_parse_seed(seed),
            arr=arr,
            start_t=float(start_t),
            on_step=self.progress.update,
            shift=self._clamp_shift(shift),
            mask=mask,
            H=new_H,
            W=new_W,
            eta=eta,
        )

        if img_out.size != (w, h):
            print(f"[fix] resize {img_out.size} -> ({w}, {h})")
            img_out = img_out.resize((w, h), Image.Resampling.LANCZOS)

        return self.detailer.merge_img(
            origin_img=img,
            img_out=img_out,
            square=box,
            save_dir=os.path.join(self.base_root, "detailer"),
            img_in=crop_img,
            imgs=imgs,
            sigmas_steps=sigmas_steps,
            final_tags=final_tags,
        )