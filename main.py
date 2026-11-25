import gc
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.sampling import Engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = Engine()
    try:
        yield
    finally:
        engine = app.state.engine
        # 모델/VAE/TE를 CPU로 내리거나 참조 제거
        for name in ("model", "te", "vae", "llm"):
            if hasattr(engine, name):
                setattr(engine, name, None)
        del engine
        app.state.engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)