import os
import numpy as np
import json
from src.core import Config
from src.storage import download_meta, upload_meta

def download_plot_data(config: Config, filename):
    return download_meta(config=config, filename=filename) or os.path.join(config.DATA_LOCAL_DIR, filename)

def read_meta_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError): 
        return None

def read_meta_np(path):
    if not os.path.isfile(path):
        return None
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError): 
        return None

def write_meta_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)

def write_meta_np(path, data):
    tmp = path + ".tmp.npy"
    np.save(tmp, np.asarray(data))
    os.replace(tmp, path)

def upload_plot_data(config, path):
    upload_meta(config=config, path=path)