import os
import json

def _save_text(save_dir, tags, tags_list:list=None, filename="tags", cfg_scale=None, steps=None, calc_type=None, shift=None, seed=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{filename}.json")

    param = {
        "tags" : tags[0],
        "tags_list": tags_list,        
        "calc_type": calc_type,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "shift": shift,
        "seed": seed,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(param, f, indent=2)

    return path