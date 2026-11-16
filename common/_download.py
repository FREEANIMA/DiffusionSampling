import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, try_to_load_from_cache, HfApi
from huggingface_hub.constants import HF_HUB_CACHE

def _purge_cached_file(repo_id: str, filename: str) -> None:
    repo_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    if not repo_dir.exists():
        return
    
    for snap in (repo_dir / "snapshots").glob("*/" + filename):
        try:
            snap.unlink()
        except FileNotFoundError:
            pass

    for snapdir in (repo_dir / "snapshots").iterdir():
        try:
            snapdir.rmdir()
        except OSError:
            pass

    refs_file = repo_dir / "refs" / "main"
    live_shas = set()
    if refs_file.exists():
        live_shas.add(refs_file.read_text().strip())

    for snapdir in (repo_dir / "snapshots").iterdir():
        live_shas.add(snapdir.name)

    blobs_dir = repo_dir / "blobs"
    if blobs_dir.exists():
        for blob in blobs_dir.iterdir():
            sha = blob.name.split(".", 1)[0]
            if sha not in live_shas:
                try:
                    blob.unlink()
                except FileNotFoundError:
                    pass


def fetch_ckpt_from_hf(repo_id: str, filename: str) -> str:
    cached = try_to_load_from_cache(repo_id=repo_id, filename=filename, repo_type="model")

    if not isinstance(cached, str):
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")

    try:
        remote_sha = HfApi().model_info(repo_id).sha
    except Exception as e:
        print(f"[{repo_id}] 원격 확인 실패, 캐시 사용: {type(e).__name__}")
        return cached

    local_sha = Path(cached).parent.name
    if local_sha == remote_sha:
        return cached

    ans = input(
        f"[{repo_id}/{filename}] 새 버전이 있어요.\n"
        f"  local : {local_sha[:8]}\n"
        f"  remote: {remote_sha[:8]}\n"
        f"업데이트할까요? [y/N]: "
    )
    if ans.strip().lower() in ("y", "yes"):
        _purge_cached_file(repo_id, filename)
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")

    return cached