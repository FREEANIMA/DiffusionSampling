import os
from huggingface_hub import hf_hub_download, HfApi, CommitOperationAdd
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from src.core import Config, error_report

def download_model(config:Config, filename) -> str | None:
    try:
        path = hf_hub_download(
            repo_id=f"{config.CKPT_REPO_ID}/VER{config.ver_number}",
            filename=filename,
            repo_type="model",
            local_dir=config.CKPT_LOCAL_DIR
        )
        return path
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        error_report(e, "download_failed")
        return None

def download_meta(config:Config, filename) -> str | None:
    try:
        path = hf_hub_download(
            repo_id=config.META_REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=config.DATA_LOCAL_DIR
        )
        return path
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        return None