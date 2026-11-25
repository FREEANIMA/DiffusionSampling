from fastapi import APIRouter, Response

from api.schemas import SamplingRequest
from src import Engine, Txt2ImgInput

router = APIRouter(prefix="/api", tags=["sampling"])

@router.post("/sampling", status_code=204)
def run_sampling(sampling: SamplingRequest, engine:Engine) -> Response:
    fmt = lambda xs: "[" + ", ".join(f"{x:.4f}" for x in xs) + "]"


    return Response(status_code=204)