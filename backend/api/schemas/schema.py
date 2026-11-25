from dataclasses import dataclass


@dataclass
class SamplingRequest:
    prompts: list[str]
    sigma_curve: list[float]
    cfg_curve: list[float]
    eta_curve: list[float]
    calculate_type:str
    seed: int | None = -1
    steps: int = 28
    width: int = 1024
    height: int = 1024
    prompt_static:bool = False
    frame_save: bool = False

@dataclass
class InPaintingRequest:
    image_b64: str | None = None       # 원본 전체 (PNG, 접두사 없는 base64)
    mask_b64: str | None = None        # 원본과 같은 크기, 수정 = 흰색(255) / 그 외 = 검정(0)
    crop_b64: str | None = None        # 크롭 영역 이미지 (크롭 없으면 None)
    crop_box: list[int] | None = None  # [x, y, w, h] 원본 좌표 (크롭 없으면 None)


@dataclass
class SamplingResponse:                # 아직 사용 안 함 (지금은 요청만 받음)
    image_b64: str
    seed: int