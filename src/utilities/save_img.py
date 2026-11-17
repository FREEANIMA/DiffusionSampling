from datetime import datetime
import os
import numpy as np
import imageio.v2 as imageio

def _save_img(save_dir, filename, img=None, imgs=None):  
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, stamp)
    os.makedirs(save_dir, exist_ok=True)
    save_img_name = os.path.join(save_dir, f"{filename}.png")

    if img:
        img.save(save_img_name)

    if imgs:
        frames = [f.resize((f.width // 2, f.height // 2)) for f in imgs]
        last = img.resize((img.width // 2, img.height // 2))
        frames += [last] * 8

        with imageio.get_writer(
            os.path.join(save_dir, "process.mp4"),
            fps=6, codec="libx264", quality=8, macro_block_size=1,
        ) as w:
            for f in frames:
                w.append_data(np.asarray(f.convert("RGB")))

    return save_img_name
    
