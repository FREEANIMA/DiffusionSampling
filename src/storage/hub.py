import os
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from src.core import Config, error_report

def download_model(config: Config,  filename: str, revision: str | None = None) -> str | None:
    repo_id = config.CKPT_REPO_ID
    target_rev = revision or (f"VER{config.ver_number}" if config.ver_number else None)

    def _try(rev: str | None) -> str:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=rev,
            local_dir=config.CKPT_LOCAL_DIR,
        )

    try:
        return _try(target_rev)
    except RevisionNotFoundError as e:
        error_report(e, "revision_not_found_fallback_main")
        try:
            return _try("main")
        except (EntryNotFoundError, RepositoryNotFoundError) as e2:
            error_report(e2, "download_failed")
            return None
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