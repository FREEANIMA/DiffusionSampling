from __future__ import annotations

import base64
import io
import json
import sys
import threading
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, Response
from PIL import Image

from .handler import Handler


class UI:
    ASSET_DIR = Path(__file__).with_name("ui_assets")

    def __init__(self, handler: Handler):
        self.handler = handler
        self.app = Flask(__name__)
        self._lock = threading.Lock()  # GPU 생성은 한 번에 하나만

        self._register_routes()

    def _read_asset(self, name: str) -> str:
        return (self.ASSET_DIR / name).read_text(encoding="utf-8")

    def _render_page(self) -> str:
        css = self._read_asset("style.css")
        layout = self._read_asset("layout.html")
        js = self._read_asset("script.js")

        tags = json.dumps(
            self.handler.get_tag_list(),
            ensure_ascii=False,
        ).replace("</", "<\\/")

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MMDIT</title>
            <style>{css}</style>
        </head>
        <body>
            {layout}

            <script>
                window.TAG_LIST = {tags};
            </script>

            <script>{js}</script>
        </body>
        </html>
        """

    @staticmethod
    def _pil_to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _b64_to_pil(data: str) -> Image.Image:
        if not data:
            raise ValueError("Mask data is empty.")

        if "," in data:
            data = data.split(",", 1)[1]

        raw = base64.b64decode(data, validate=True)

        with Image.open(io.BytesIO(raw)) as img:
            return img.convert("L")

    def _generate(self, p: dict):
        seed_text = str(p.get("seed", "-1")).strip()

        common = dict(
            tags_text=p["tags"],
            nsfw=bool(p["nsfw"]),
            calc_type=p["sampler"],
            cfg_scale=float(p["cfg"]),
            steps=int(p["steps"]),
            seed=int(seed_text) if seed_text else -1,
            fixed=bool(p.get("fixed", True)),
        )

        if p["mode"] == "txt2img":
            return self.handler.generate_txt2img(
                H=int(p["H"]),
                W=int(p["W"]),
                **common,
            )

        if p["mode"] in ("inpainting", "detailer"):
            img_path = str(p["img_path"]).strip()

            if not img_path or not Path(img_path).is_file():
                raise FileNotFoundError(f"Image not found: {img_path}")

            if p["mode"] == "inpainting":
                mask = self._b64_to_pil(p.get("mask") or "")
                return self.handler.generate_inpainting(
                    img_path=img_path,
                    mask=mask,
                    start_t=float(p.get("start_t", 0.9)),
                    **common,
                )

            r = p.get("rect")
            if not r:
                raise ValueError("사각형 영역이 필요합니다.")

            x, y = int(r["x"]), int(r["y"])
            box = (x, y, x + int(r["w"]), y + int(r["h"]))

            return self.handler.generate_detailering(
                img_path=img_path,
                box=box,
                start_t=float(p.get("start_t", 0.85)),
                neg_tags_text=str(p.get("neg_tags", "") or ""),
                n_cfg_scale=float(p.get("n_cfg", 0) or 0),
                **common,
            )

        raise ValueError(f"Unknown mode: {p['mode']}")

    def _register_routes(self):

        @self.app.get("/")
        def index():
            return Response(self._render_page(), mimetype="text/html")

        @self.app.post("/api/load_image")
        def load_image():
            try:
                params = request.get_json(force=True)
                img_path = str(params.get("path", "")).strip()

                if not img_path or not Path(img_path).is_file():
                    raise FileNotFoundError(f"Image not found: {img_path}")

                with Image.open(img_path) as img:
                    img = img.convert("RGB")

                return jsonify({
                    "success": True,
                    "image": self._pil_to_b64(img),
                })

            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                }), 400

        @self.app.post("/api/generate")
        def generate():
            if not self._lock.acquire(blocking=False):
                return jsonify({
                    "success": False,
                    "error": "이미 생성 중입니다.",
                }), 409

            try:
                params = request.get_json(force=True)
                img, path = self._generate(params)

                return jsonify({
                    "success": True,
                    "path": str(path),
                    "image": self._pil_to_b64(img),
                })

            except Exception as e:
                traceback.print_exc()

                return jsonify({
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                }), 500

            finally:
                self._lock.release()

        @self.app.get("/api/progress")
        def progress():
            return jsonify(self.handler.progress.snapshot())

        @self.app.get("/api/logs")
        def logs():
            return jsonify({"lines": self.handler.console.drain()})

        @self.app.get("/api/tag_history")
        def tag_history():
            return jsonify({"items": self.handler.get_tag_history()})

    def run(self, host: str = "127.0.0.1", port: int = 7860,
            colab: bool | None = None, height: int = 1000):
        if colab is None:
            colab = "google.colab" in sys.modules  # 자동 감지

        def serve():
            self.app.run(
                host=host,
                port=port,
                debug=False,
                threaded=True,
                use_reloader=False,
            )

        if colab:
            from google.colab import output

            # 셀이 막히지 않게 백그라운드 스레드로 실행
            threading.Thread(target=serve, daemon=True).start()
            output.serve_kernel_port_as_iframe(port, height=height)
        else:
            print(f"MMDIT Local UI: http://{host}:{port}")
            serve()