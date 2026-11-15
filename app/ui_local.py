
from __future__ import annotations

import base64
import io
import json
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, Response
from PIL import Image

from .handler import Handler


class LocalUI:
    ASSET_DIR = Path(__file__).with_name("ui_assets")

    def __init__(self, handler: Handler):
        self.handler = handler
        self.app = Flask(__name__)

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
                window.LOCAL_MODE = true;
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

    def _generate(self, p: dict):
        seed_text = str(p.get("seed", "-1")).strip()

        common = dict(
            tags_text=p["tags"],
            nsfw=bool(p["nsfw"]),
            calc_type=p["sampler"],
            cfg_scale=float(p["cfg"]),
            steps=int(p["steps"]),
            seed=int(seed_text) if seed_text else -1,
        )

        if p["mode"] == "txt2img":
            return self.handler.generate_txt2img(
                H=int(p["H"]),
                W=int(p["W"]),
                **common,
            )

        if p["mode"] == "inpainting":
            img_path = str(p["img_path"]).strip()

            if not img_path or not Path(img_path).is_file():
                raise FileNotFoundError(
                    f"Image not found: {img_path}"
                )

            return self.handler.generate_inpainting(
                img_path=img_path,
                start_t=float(p.get("start_t", 0.9)),
                **common,
            )

        raise ValueError(f"Unknown mode: {p['mode']}")

    def _register_routes(self):

        @self.app.get("/")
        def index():
            return Response(
                self._render_page(),
                mimetype="text/html",
            )

        @self.app.post("/api/generate")
        def generate():
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

    def run(
        self,
        host: str = "127.0.0.1",
        port: int = 7860,
    ):
        print(f"MMDIT Local UI: http://{host}:{port}")

        self.app.run(
            host=host,
            port=port,
            debug=False,
            threaded=False,
            use_reloader=False,
        )
