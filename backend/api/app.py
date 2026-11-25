from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from src.sampling import Engine

from services import load_tags
from api.routes import inpainting, sampling, tag_browser

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = Engine(console=None)
    app.state.tags = load_tags()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sampling.router)
app.include_router(inpainting.router)
app.include_router(tag_browser.router)