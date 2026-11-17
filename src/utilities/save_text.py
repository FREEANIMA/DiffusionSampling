import os

def _save_text(save_dir, tags, tags_list=None, filename="tags",
               cfg_scale=None, steps=None, calc_type=None, shift=None, seed=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{filename}.txt")

    lines = ["[tags]", tags[0]]

    if tags_list is not None:
        lines += ["", "[tags_list]"]
        lines += [str(t) for t in tags_list]

    params = {
        "calc_type": calc_type,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "shift": shift,
        "seed": seed,
    }
    params = {k: v for k, v in params.items() if v is not None}

    if params:
        lines += ["", "[params]"]
        lines += [f"{k}: {v}" for k, v in params.items()]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return path