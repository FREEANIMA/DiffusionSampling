from fastapi import APIRouter, Depends, Request, Response

from api.schemas import SamplingRequest
from src.sampling import Engine

router = APIRouter(prefix="/api", tags=["sampling"])


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


@router.post("/sampling", status_code=204)
def run_sampling(sampling: SamplingRequest, engine: Engine = Depends(get_engine)) -> Response:
    print(sampling.prompts)
    return Response(status_code=204)