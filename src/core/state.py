import queue
from .runtime import config_settings

config = config_settings()
console = queue.Queue(maxsize=64)