import threading
from pathlib import Path

BASE_ROOT = Path(__file__).resolve().parents[2]

IDLE_LIMIT = 600

STOP_EVENT = threading.Event()
LIMIT_TIME = threading.Event()
UPLOAD_START = threading.Event()
SAVE_WATCH = threading.Event()

