import os
from huggingface_hub import hf_hub_download, HfApi, CommitOperationAdd
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from src.core import Config, error_report

def download_model(config:Config, filename) -> str | None:
    try:
        path = hf_hub_download(
            repo_id=config.CKPT_REPO_ID,
            filename=filename,
            repo_type="model",
            local_dir=config.CKPT_LOCAL_DIR
        )
        return path
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
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

def upload_model(accum_step, config:Config) -> bool:
    model_path = os.path.join(config.CKPT_LOCAL_DIR, config.MODEL_FILENAME)
    ema_model_path = os.path.join(config.CKPT_LOCAL_DIR, config.EMA_FILENAME)
    try:
        api = HfApi()
        api.create_commit(
            repo_id=config.CKPT_REPO_ID,
            repo_type="model",
            operations=[
                CommitOperationAdd(path_in_repo=config.MODEL_FILENAME, path_or_fileobj=model_path),
                CommitOperationAdd(path_in_repo=config.EMA_FILENAME, path_or_fileobj=ema_model_path),
            ],
            commit_message=f"step {accum_step}",
        )
        return True
    except Exception as e:
        error_report(error=e, name="uploader_model")
        return False

def upload_meta(config: Config, path: str):
    file_name = os.path.basename(path)
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=path,  
            path_in_repo=file_name,
            repo_id=config.META_REPO_ID,
            repo_type="dataset",
        )

        return True
    except Exception as e:
        error_report(error=e, name="uploader_meta")
        return False

def upload_validation(config: Config, folder_path: str, dir_name: str) -> bool:
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=folder_path,
            path_in_repo=f"validation/{dir_name}",
            repo_id=config.META_REPO_ID,
            repo_type="dataset",
            commit_message=f"validation {dir_name}",
        )
        return True
    except Exception as e:
        error_report(error=e, name="uploader_meta")
        return False