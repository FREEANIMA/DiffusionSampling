import os
import time
import traceback
from src.core import shared

def log(console, msg):
    if console is None:
        print(msg)
    else:
        console.put(msg)

def error_report(error=None, name=None):
    tag = f"{name}_" if name else ""
    path = os.path.join(shared.BASE_ROOT, "error_report", f"error_{tag}{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000}.txt")

    body = traceback.format_exc()
    if body.strip() == "NoneType: None":
        body = ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{type(error).__name__ if error else ''}: {error if error else ''}\n\n{body}")
    return path