import gc
import os
from pathlib import Path
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import sys
import signal

COLAB_ROOT = "/content/drive/MyDrive/model/mmdit_sampling"

_sigint_count = 0

def _sigint_handler(signum, frame):
    global _sigint_count
    _sigint_count += 1
    if _sigint_count >= 2:
        print("\n[MAIN] force exit")
        os._exit(130)
    print("\n[MAIN] Ctrl+C: shutting down... (한 번 더 누르면 강제 종료)")
    raise KeyboardInterrupt

def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

def get_root() -> str:
    if is_colab():
        from google.colab import drive

        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")

        os.makedirs(COLAB_ROOT, exist_ok=True)
        return COLAB_ROOT

    return os.path.dirname(os.path.abspath(__file__))


def load_env(root: str) -> None:
    env_path = Path(root) / ".env"

    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)
        print(f"[ENV] loaded: {env_path}")
    else:
        print(f"[ENV] .env not found at {env_path}")


def hf_login() -> None:
    token = os.environ.get("HF_TOKEN")

    if token:
        login(token=token)
        print("[HF] logged in via HF_TOKEN")
        return

    try:
        from huggingface_hub import whoami

        info = whoami()
        print(f"[HF] already logged in as {info.get('name')}")
    except Exception:
        print("[HF] not logged in. Set HF_TOKEN in .env or run `hf auth login`.")


def cleanup_memory(*objs) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass

    for _ in range(3):
        gc.collect()

    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved() / 1024**3
        print(f"[MEM] cuda allocated={alloc:.2f}GB reserved={reserv:.2f}GB")
    except Exception:
        pass


if __name__ == "__main__":
    handler = None
    ui = None
    
    if not is_colab():
        signal.signal(signal.SIGINT, _sigint_handler)
        signal.signal(signal.SIGTERM, _sigint_handler)
        
    try:
        ROOT = get_root()
        load_env(ROOT)
        hf_login()

        from app.handler import Handler
        from app.ui import UI

        handler = Handler(base_root=ROOT)
        ui = UI(handler)
        ui.run(colab=is_colab())
    except KeyboardInterrupt:
        print("[MAIN] interrupted by user")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[MAIN] error: {e}")
    finally:
        try:
            if ui is not None and hasattr(ui, "close"):
                ui.close()
        except Exception:
            pass

        try:
            if handler is not None and hasattr(handler, "close"):
                handler.close()
        except Exception:
            pass

        cleanup_memory(ui, handler)
        print("[MAIN] shutdown complete")

        if not is_colab():
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)