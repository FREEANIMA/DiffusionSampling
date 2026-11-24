import argparse
from dataclasses import fields
import json, os, glob, sys
from .config import Config
from .shared import BASE_ROOT

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=int, default=None)
    parser.add_argument("--deadline", type=int, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

def prepared_dir(config:Config):
    os.makedirs(config.GENERATION_DIR, exist_ok=True)
    os.makedirs(config.INPAINTING_DIR, exist_ok=True)
    os.makedirs(config.DETAIL_DIR, exist_ok=True)
    os.makedirs(config.ERROR_REPORT, exist_ok=True)
    os.makedirs(config.CKPT_LOCAL_DIR, exist_ok=True)

def config_settings(ver_number: int | None = None):
    args = _parse_args()

    if ver_number is None:
        ver_number = args.ver

    if ver_number is None:
        ver_number = int(input("ver number : "))

    pattern = os.path.join(BASE_ROOT, "configs", f"version_{ver_number}", "*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"json 파일이 없습니다: {pattern}")

    config_dict = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            config_dict.update(json.load(f))

    def pick(cls, params):
        names = {f.name for f in fields(cls)}
        return {k: v for k, v in params.items() if k in names}

    config = Config(**pick(Config, config_dict))

    return config