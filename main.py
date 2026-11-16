import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login

COLAB_ROOT = "/content/drive/MyDrive/model/mmdit_sampling"

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


if __name__ == "__main__":
    ROOT = get_root()
    load_env(ROOT)
    hf_login()

    from app.handler import Handler
    from app.ui import UI

    handler = Handler(base_root=ROOT)
    UI(handler).run(colab=is_colab())