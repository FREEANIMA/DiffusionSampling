from datetime import datetime
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .sigma import sigma_schedular


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)   # Pillow 10.1+
    except TypeError:
        return ImageFont.load_default()


def _with_label(frame, text):
    """프레임 위에 검은 띠를 붙이고 가운데에 글자를 넣음."""
    frame = frame.convert("RGB")
    w, h = frame.size
    size = max(14, w // 20)
    bar = size * 2                                  # 짝수 유지 (libx264 요구)

    canvas = Image.new("RGB", (w, h + bar), (16, 16, 16))
    canvas.paste(frame, (0, bar))

    d = ImageDraw.Draw(canvas)
    font = _font(size)
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    d.text(((w - (r - l)) / 2 - l, (bar - (b - t)) / 2 - t),
           text, font=font, fill=(235, 235, 235))
    return canvas


def _save_img(save_dir, filename, img=None, imgs=None, sigmas=None, shift=3.0):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, stamp)
    os.makedirs(save_dir, exist_ok=True)
    save_img_name = os.path.join(save_dir, f"{filename}.png")

    if img is not None:
        img.save(save_img_name)

    if imgs:
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("[warn] imageio 미설치: mp4 저장 생략 (pip install imageio imageio-ffmpeg)")
            return save_img_name

        n = len(imgs)

        # sigma가 안 넘어오면 전체 스케줄에서 계산 (마지막 값은 모델 호출에 안 쓰이므로 제외)
        if sigmas is None:
            sigmas = sigma_schedular(step=n, device="cpu", shift=shift)[:-1]
        sigmas = [float(s) for s in sigmas]

        def label(i):
            if i < len(sigmas):
                return f"step {i + 1}/{n}   sigma {sigmas[i]:.3f}"
            return f"step {i + 1}/{n}"

        half = lambda f: f.resize((f.width // 2, f.height // 2))

        frames = [_with_label(half(f), label(i)) for i, f in enumerate(imgs)]
        final = img if img is not None else imgs[-1]
        frames += [_with_label(half(final), "final")] * 8

        with imageio.get_writer(
            os.path.join(save_dir, "process.mp4"),
            fps=6, codec="libx264", quality=8, macro_block_size=1,
        ) as w:
            for f in frames:
                w.append_data(np.asarray(f))

    return save_img_name