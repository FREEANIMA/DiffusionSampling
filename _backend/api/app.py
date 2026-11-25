from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import inpainting, sampling

app = FastAPI(title="Sampling API")

# 프론트(dev_server)와 포트가 달라도 호출되도록 localhost 계열 origin 허용
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sampling.router)
app.include_router(inpainting.router)