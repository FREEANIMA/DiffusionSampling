import threading
from src.core import state, log, error_report, shared
from .meta_save import read_meta_np, write_meta_json, write_meta_np, read_meta_json, download_plot_data, upload_plot_data
import numpy as np
from dataclasses import dataclass, field
import queue
import time

NB = 10000

@dataclass
class SigmaSchedual_Info:
    Y_COUNT: np.ndarray = field(default_factory=lambda: np.zeros(NB, dtype=np.int64))

class SigmaScheduler:
    def __init__(self, stop_event: threading.Event, console=None):
        self.console = console
        self.stop_event = stop_event

        self.lock = threading.Lock()
        self.batchinfo = SigmaSchedual_Info()
        self.rng = np.random.default_rng()
        
        self.sigma_data_path = download_plot_data(state.config, f"{state.config.SIGMA_HISTROY_FILE_NAME}_{state.config.ver_number}.npy")
        hist = read_meta_np(self.sigma_data_path) if self.sigma_data_path is not None else None
        if hist is not None:
            hist = np.asarray(hist, dtype=np.int64).ravel()
            if hist.shape[0] == NB:             
                self.batchinfo.Y_COUNT += hist
        log(console, f"download completed : {self.sigma_data_path}")

        self.loss_log_path = download_plot_data(state.config, f"{state.config.LOSS_HISTORY_FILE_NAME}_{state.config.ver_number}.json")
        loaded = read_meta_json(self.loss_log_path) if self.loss_log_path is not None else None

        log(console, f"download completed : {self.loss_log_path}")

        self.loss_log:dict = loaded if isinstance(loaded, dict) else {}
        self.loss_q = queue.Queue(maxsize=1024)
        self.upload_event = threading.Event()

    def get_loss_per_sigma(self, sigma_1d, loss_per_sample, step):
        try:
            if hasattr(sigma_1d, "detach"):
                sigma_1d = sigma_1d.detach().clone()
            self.loss_q.put_nowait((int(step), sigma_1d, loss_per_sample.detach().clone()))
        except queue.Full:
            pass

    def save_loss_per_sigma(self):
        while not self.stop_event.is_set():
            try:
                step, sigma_1d, loss = self.loss_q.get(timeout=shared.IDLE_LIMIT)
            except queue.Empty:
                continue
            try:
                sigma_np = sigma_1d.float().cpu().numpy() if hasattr(sigma_1d, "float") else np.asarray(sigma_1d)
                self.loss_log[str(step)] = {
                    "loss": np.round(loss.float().cpu().numpy(), 5).tolist(),
                    "sigma": np.round(sigma_np, 5).tolist(),
                }
                write_meta_json(self.loss_log_path, self.loss_log)

                if self.upload_event.is_set():
                    upload_plot_data(state.config, self.loss_log_path)
                    self.upload_event.clear()

            except Exception as e:
                error_report(e, "sigma_save")
                self.stop_event.set()
                raise

        upload_plot_data(state.config, self.loss_log_path)
        
    def distribution(self, B):
        x = self.rng.uniform(0.0, 1.0, size=B).astype(np.float32)
        return self.caculate(x)

    def to_idx(self, v):
        return np.clip(np.rint(v * (NB - 1)).astype(np.int64), 0, NB - 1)

    def transform(self, x):
        z = np.clip(state.config.beta - state.config.gamma * x, -1.0, 1.0)
        y = state.config.alpha * np.arccos(z) - state.config.zeta
        y = np.clip(y, state.config.low_sigma, state.config.high_sigma)
        return y

    def caculate(self, x):
        y = self.transform(x)
        cy = np.bincount(self.to_idx(y), minlength=NB)
        with self.lock:
            self.batchinfo.Y_COUNT += cy
        return y

    def snapshot(self):
        with self.lock:
            return self.batchinfo.Y_COUNT.copy()

    def save_plot_data(self):
        while not self.stop_event.is_set():
            try:
                deadline = time.monotonic() + state.config.sigma_distribution_save_time
                while not self.stop_event.is_set():
                    remain = deadline - time.monotonic()
                    if remain <= 0:
                        break
                    state.MONITOR["sigma_save_timer"] = f"{int(remain)}s"
                    write_meta_np(path=self.sigma_data_path, data=self.snapshot())
                    self.stop_event.wait(1)

                state.MONITOR["sigma_save_timer"] = "saving..."
                self.upload_event.set()
                self._save()
            except Exception as e:
                error_report(e, "sigma_save")
                self.stop_event.set()
                raise

        state.MONITOR.pop("sigma_save_timer", None)

    def _save(self):
        write_meta_np(path=self.sigma_data_path, data=self.snapshot())
        upload_plot_data(state.config, self.sigma_data_path)

    def start(self):
        t1 = threading.Thread(
            target=self.save_plot_data,
            name="sigma_scheduler",
            daemon=True,
        )
        t1.start()

        t2 = threading.Thread(
            target=self.save_loss_per_sigma,
            name="loss_report",
            daemon=True
        )

        t2.start()
        return [t1, t2]