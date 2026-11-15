import os
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


def hf_login():
    token = os.environ.get("HF_TOKEN")

    if token:
        login(token=token)


if __name__ == "__main__":
    ROOT = get_root()
    hf_login()

    from app.handler import Handler
    from app.ui import UI

    handler = Handler(base_root=ROOT)
    UI(handler).run(colab=is_colab())