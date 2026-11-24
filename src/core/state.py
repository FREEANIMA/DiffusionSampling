import queue
from .runtime import config_settings

config = config_settings()
timings = {}
vram = {}
MONITOR = {}

console = queue.Queue(maxsize=64)

_short = set(config.NAS_SHORT)
NAS_CANDIDATE_TABLE = [k for k in config.NAS if k not in _short]