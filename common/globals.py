import queue

class console_q:
    def __init__(self, maxsize: int = 256):
        self._q = queue.Queue(maxsize=maxsize)

    def put(self, msg) -> None:
        msg = str(msg)
        print(msg, flush=True)           # 서버 터미널에도 출력
        while True:
            try:
                self._q.put_nowait(msg)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()  # 가장 오래된 것 버림
                except queue.Empty:
                    pass

    def drain(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

INIT_DEVICE = None
BLOCK_CHUNK = None