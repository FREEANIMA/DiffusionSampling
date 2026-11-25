from fastapi import APIRouter, Depends, Request, Response

from api.schemas import SamplingRequest
from src.sampling import Engine, Txt2ImgInput

router = APIRouter(prefix="/api", tags=["sampling"])

def get_engine(request: Request) -> Engine:
    return request.app.state.engine

@router.post("/sampling", status_code=204)
def run_sampling(sampling: SamplingRequest, engine: Engine = Depends(get_engine)) -> Response:
    txt2img_input = Txt2ImgInput(
        prompts=sampling.prompts,
        steps=sampling.steps,
        seed=sampling.seed,
        H=sampling.height,
        W=sampling.width,
        calculate_type=sampling.calculate_type,
        sigma_curve=sampling.sigma_curve,
        cfg_curve=sampling.cfg_curve,
        eta_curve=sampling.eta_curve,
        frame_save=sampling.frame_save,
        prompt_static=sampling.prompt_static,
    )

    engine.txt2img(txt2img_input=txt2img_input, start_t=0, inpaint_ctx=None)

    return Response(status_code=204)