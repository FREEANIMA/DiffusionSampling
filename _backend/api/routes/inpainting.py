from fastapi import APIRouter, Response

from api.schemas import InPaintingRequest, SamplingRequest
from services.debug import print_inpainting, print_sampling, save_debug_images

router = APIRouter(prefix="/api", tags=["inpainting"])

# body가 둘이라 JSON의 키가 파라미터 이름: { "sampling": {...}, "inpainting": {...} }
# 지금은 요청을 받아서 출력만 함 (반환 없음 → 204 No Content)
@router.post("/inpainting", status_code=204)
def run_inpainting(sampling: SamplingRequest, inpainting: InPaintingRequest) -> Response:
    print("\n===== POST /api/inpainting =====")
    print_sampling(sampling)
    print_inpainting(inpainting)
    save_debug_images(
        image=inpainting.image_b64,
        mask=inpainting.mask_b64,
        crop=inpainting.crop_b64,
    )
    return Response(status_code=204)