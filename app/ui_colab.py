# app/ui_colab.py
from __future__ import annotations

import base64
import io
import json
import time
import traceback
from pathlib import Path

from google.colab.output import eval_js
from IPython.display import HTML, display
from PIL import Image

from common.utils import free_cuda

from .handler import Handler

class ColabUI:
    ASSET_DIR = Path(__file__).with_name("ui_assets")
    POLL_INTERVAL = 0.2  # seconds

    def __init__(self, handler: Handler):
        self.handler = handler
        self._result_area = None

    # ---------- 폼 ----------
    def _read_asset(self, name: str) -> str:
        return (self.ASSET_DIR / name).read_text(encoding="utf-8")
    
    def _render_form(self) -> None:
        css = self._read_asset("style.css")
        layout = self._read_asset("layout.html")
        js = self._read_asset("script.js")
        display(HTML(
            f"<style>{css}</style>\n"
            f"{layout}\n"
            f"<script>{js}</script>"
        ))
        self._result_area = display(HTML("<div></div>"), display_id=True)
        eval_js("window.ui_params = null")

    def _set_status(self, text: str, busy: bool) -> None:
        eval_js(
            f"document.getElementById('out').textContent = {json.dumps(text)};"
            f"document.getElementById('run').disabled = {str(busy).lower()};"
        )

    def _wait_for_params(self) -> dict:
        """Generate 버튼이 눌릴 때까지 대기. 읽자마자 비워서 다음 클릭에 대비."""
        while True:
            params = eval_js("window.ui_params")
            if params:
                eval_js("window.ui_params = null")
                return params
            time.sleep(self.POLL_INTERVAL)

    # ---------- 생성 ----------
    def _generate(self, p: dict):
        seed_text = str(p["seed"]).strip()
        common = dict(
            tags_text=p["tags"],
            nsfw=bool(p["nsfw"]),
            calc_type=p["sampler"],
            cfg_scale=float(p["cfg"]),
            steps=int(p["steps"]),
            seed=int(seed_text) if seed_text else -1,
        )

        if p["mode"] == "txt2img":
            return self.handler.generate_txt2img(H=int(p["H"]), W=int(p["W"]), **common)

        img_path = p["img_path"].strip()
        if not img_path or not Path(img_path).exists():
            raise FileNotFoundError(f"이미지 경로 없음: {img_path}")
        return self.handler.generate_inpainting(
            img_path=img_path, start_t=float(p.get("start_t", 0.9)), **common
        )

    @staticmethod
    def _pil_to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _show_result(self, img: Image.Image, path: str) -> None:
        self._result_area.update(HTML(f"""
        <div style="font-family:sans-serif;">
          <div><b>saved:</b> {path}</div>
          <img src="data:image/png;base64,{self._pil_to_b64(img)}"
               style="max-width:512px; border:1px solid #ccc; margin-top:8px;">
        </div>
        """))

    def _show_error(self, e: Exception) -> None:
        tb = traceback.format_exc()
        self._result_area.update(HTML(
            f"<pre style='color:#b00; white-space:pre-wrap;'>"
            f"실패: {type(e).__name__}: {e}\n\n{tb}</pre>"
        ))

    def run(self) -> None:
        self._render_form()
        self._set_status("파라미터를 입력하고 Generate 를 누르세요", busy=False)

        try:
            while True:
                params = self._wait_for_params()
                self._set_status("running...", busy=True)
                try:
                    img, path = self._generate(params)
                    self._show_result(img, path)
                    status = f"done: {path}"
                except Exception as e:
                    self._show_error(e)
                    status = "failed"

                free_cuda(verbose=True)
                self._set_status(status, busy=False)
        except KeyboardInterrupt:
            self._set_status("중지됨 (run() 을 다시 실행하세요)", busy=True)