from fastapi import APIRouter, Response

from api.schemas import SamplingRequest
from services.debug import print_sampling

router = APIRouter(prefix="/api", tags=["sampling"])


# 지금은 요청을 받아서 출력만 함 (반환 없음 → 204 No Content)
@router.post("/sampling", status_code=204)
def run_sampling(req: SamplingRequest) -> Response:
    print("\n===== POST /api/sampling =====")
    print_sampling(req)
    return Response(status_code=204)