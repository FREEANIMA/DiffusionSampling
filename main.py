
import os
from huggingface_hub import login

COLAB_ROOT = "/content/drive/MyDrive/model/mmdit_sampling"


def is_colab():
    return os.path.exists("/content") and os.path.exists("/usr/local/lib/python3.10/dist-packages/google/colab")


def get_root():
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

    handler = Handler(base_root=ROOT)

    if is_colab():
        from app.ui_colab import ColabUI
        ColabUI(handler).run()
    else:
        from app.ui_local import LocalUI
        LocalUI(handler).run()
