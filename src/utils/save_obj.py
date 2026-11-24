import os
import json
import imageio.v2 as imageio

def save_video(path, frames, fps=16):
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    try:
        with imageio.get_writer(tmp, fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

def save_img(path, img, format="PNG"):
    tmp = path + ".tmp"
    try:
        img.save(tmp, format=format)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

def save_json(path, data):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise