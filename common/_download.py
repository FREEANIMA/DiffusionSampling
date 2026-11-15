from pathlib import Path
from huggingface_hub import hf_hub_download, try_to_load_from_cache, HfApi


def fetch_ckpt_from_hf(repo_id, filename):
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

    # 3) 다를 때만 질문
    ans = input(
        f"[{repo_id}/{filename}] 새 버전이 있어요.\n"
        f"  local : {local_sha[:8]}\n"
        f"  remote: {remote_sha[:8]}\n"
        f"업데이트할까요? [y/N]: "
    )
    if ans.strip().lower() in ("y", "yes"):
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    return cached