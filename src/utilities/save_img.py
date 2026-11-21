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
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def _with_label(frame, text):
    frame = frame.convert("RGB")
    w, h = frame.size
    size = max(14, w // 20)
    bar = size * 2

    canvas = Image.new("RGB", (w, h + bar), (16, 16, 16))
    canvas.paste(frame, (0, bar))

    d = ImageDraw.Draw(canvas)
    font = _font(size)
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    d.text(((w - (r - l)) / 2 - l, (bar - (b - t)) / 2 - t),
           text, font=font, fill=(235, 235, 235))
    return canvas


def _as_frame_list(x):
    if isinstance(x, Image.Image):
        return [x]
    return list(x)


def _save_img(save_dir, filename, imgs=None, video_imgs=None, sigmas=None, shift=3.0):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, stamp)
    os.makedirs(save_dir, exist_ok=True)

    if imgs is not None and not isinstance(imgs, (list, tuple)):
        imgs = [imgs]          # PIL 한 장이면 리스트로

    paths = []
    for i, im in enumerate(imgs or []):
        p = os.path.join(save_dir, f"{filename}_{i}.png")
        im.save(p)
        paths.append(p)

    if video_imgs:
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("[warn] imageio 미설치: mp4 저장 생략 (pip install imageio imageio-ffmpeg)")
            return paths

        # [step] 각 원소를 PIL 리스트로 통일 → [step][b]
        per_step = [_as_frame_list(s) for s in video_imgs]
        B = len(per_step[0])

        # [step][b] → [b][step]
        per_sample = [[s[b] for s in per_step] for b in range(B)]

        n = len(per_step)
        if sigmas is None:
            sigmas = sigma_schedular(step=n, device="cpu", shift=shift)[:-1]
        sigmas = [float(s) for s in sigmas]

        def label(i):
            if i < len(sigmas):
                return f"step {i + 1}/{n}   sigma {sigmas[i]:.3f}"
            return f"step {i + 1}/{n}"

        half = lambda f: f.resize((f.width // 2, f.height // 2))

        for b, frames in enumerate(per_sample):
            name = "process.mp4" if B == 1 else f"process_{b}.mp4"
            final = imgs[b] if imgs and b < len(imgs) else frames[-1]

            size = half(frames[0]).size
            final_half = half(final)
            if final_half.size != size:
                final_half = final_half.resize(size, Image.Resampling.LANCZOS)

            with imageio.get_writer(
                os.path.join(save_dir, name),
                fps=6, codec="libx264", quality=8, macro_block_size=1,
            ) as w:
                for i, f in enumerate(frames):
                    w.append_data(np.asarray(_with_label(half(f), label(i))))

                last = np.asarray(_with_label(final_half, "final"))
                for _ in range(8):
                    w.append_data(last)

    return paths