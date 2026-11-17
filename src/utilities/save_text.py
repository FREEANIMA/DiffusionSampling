import os

def _save_text(save_dir, tags, tags_list=None, filename="tags", extra=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{filename}.txt")

    lines = ["[tags]", tags[0]]

    if tags_list is not None:
        lines += ["", "[tags_list]"]
        lines += [str(t) for t in tags_list]

    if extra:
        lines += ["", "[extra]"]
        lines += [f"{k}: {v}" for k, v in extra.items()]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return path