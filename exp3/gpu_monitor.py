"""Background GPU memory and utilization monitor.

Spawns a daemon thread that samples PyTorch CUDA memory stats and (if
``pynvml`` is available) NVIDIA Management Library stats every ``interval``
seconds, writing one CSV row per GPU per sample. Designed for OOM-debugging
during GRPO training: tag the current phase from the training loop
(``set_phase("backward", step=42)``) and the rows are labeled accordingly so
you can later answer "which phase blew up the memory?".

Usage from train.py:

    monitor = GPUMonitor(config["monitor_config"])
    monitor.set_phase("init", step=0)
    monitor.start()
    ...
    monitor.set_phase("rollout", step=self.global_step)
    ...
    monitor.stop()

Threading model:
    The monitor runs in a daemon thread inside the training process. If the
    trainer hangs in CUDA, the monitor hangs too — that is acceptable for an
    OOM-debugging tool because the training log will already show the hang.
    For a more resilient setup, a separate subprocess would be needed.

CSV behavior:
    - Truncated on every fresh start (mode "w") -> rerunning train.py wipes
      the previous run's data, as requested.
    - Header written once at start.
    - Each row is flushed + fsync'd so partial data survives a crash.

Pynvml is optional:
    - Imported lazily; absence does not crash the trainer.
    - When unavailable, NVML columns are written as empty cells.
"""

import csv
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import torch


class GPUMonitor:
    """Daemon-thread GPU memory and utilization sampler -> CSV."""

    CSV_HEADER = [
        "timestamp",
        "elapsed_s",
        "gpu_id",
        "gpu_name",
        "mem_allocated_gb",
        "mem_reserved_gb",
        "mem_max_allocated_gb",
        "mem_total_gb",
        "util_compute_pct",
        "util_mem_pct",
        "temperature_c",
        "power_w",
        "phase",
        "step",
    ]

    BYTES_PER_GB = 1024 ** 3  # binary GB (GiB), matches `nvidia-smi` reporting

    def __init__(self, config: dict):
        self.csv_path = Path(config["csv_path"])
        self.log_dir = Path(config["log_dir"])
        self.interval_s = float(config.get("interval_s", 2.0))
        self.gpu_ids = config.get("gpu_ids", None)  # None = all visible GPUs

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        if not torch.cuda.is_available():
            raise RuntimeError("GPUMonitor requires CUDA. No GPU detected.")

        # Resolve which GPUs to watch.
        n_devices = torch.cuda.device_count()
        if self.gpu_ids is None:
            self.gpu_ids = list(range(n_devices))
        else:
            for gid in self.gpu_ids:
                if gid < 0 or gid >= n_devices:
                    raise ValueError(
                        f"gpu_id {gid} out of range [0, {n_devices})"
                    )

        # Cache device names + total memory once (these don't change).
        self._gpu_names: dict[int, str] = {}
        self._gpu_total_bytes: dict[int, int] = {}
        for gid in self.gpu_ids:
            props = torch.cuda.get_device_properties(gid)
            self._gpu_names[gid] = props.name
            self._gpu_total_bytes[gid] = props.total_memory

        # Optional NVML setup (graceful if pynvml missing).
        self._nvml_handles: dict[int, object] = {}
        self._nvml_available = self._init_nvml()

        # Phase tagging — written by the trainer, read by the sampler thread.
        # Single-attribute writes are atomic under CPython's GIL, so no lock
        # is needed for these reads.
        self._current_phase: str = "init"
        self._current_step: int = 0

        # Thread machinery
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float | None = None

        # Open CSV in "w" mode -> truncates any existing file from prior runs.
        self._csv_file = self.csv_path.open("w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADER)
        self._csv_file.flush()

        self.logger.info(
            f"GPUMonitor initialised: csv={self.csv_path}, "
            f"interval={self.interval_s}s, "
            f"gpus={self.gpu_ids} ({[self._gpu_names[g] for g in self.gpu_ids]}), "
            f"nvml={'available' if self._nvml_available else 'unavailable'}"
        )

    def _setup_logging(self):
        log_file = self.log_dir / "gpu_monitor.log"
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh = logging.FileHandler(log_file, mode="w")
            fh.setFormatter(fmt)
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)

    def _init_nvml(self) -> bool:
        """Initialise NVML lazily. Return True on success, False otherwise."""
        try:
            import pynvml  # type: ignore
        except ImportError:
            self.logger.warning(
                "pynvml not installed; util/temp/power columns will be empty. "
                "Install with: pip install nvidia-ml-py"
            )
            self._pynvml = None
            return False

        try:
            pynvml.nvmlInit()
        except Exception as e:
            self.logger.warning(
                f"pynvml.nvmlInit() failed: {e}. NVML columns will be empty."
            )
            self._pynvml = None
            return False

        self._pynvml = pynvml
        # Map torch device id -> NVML handle. CUDA_VISIBLE_DEVICES can remap
        # indices, but for the common case (no remapping) torch id == NVML id.
        # If a user has set CUDA_VISIBLE_DEVICES, NVML still sees the physical
        # GPU index; we resolve via UUID for safety.
        try:
            for gid in self.gpu_ids:
                # torch's device UUID matches NVML's UUID exactly.
                torch_uuid = str(torch.cuda.get_device_properties(gid).uuid)
                handle = None
                for nvml_idx in range(pynvml.nvmlDeviceGetCount()):
                    h = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
                    nvml_uuid = pynvml.nvmlDeviceGetUUID(h)
                    if isinstance(nvml_uuid, bytes):
                        nvml_uuid = nvml_uuid.decode()
                    if nvml_uuid.replace("GPU-", "") == torch_uuid.replace("GPU-", ""):
                        handle = h
                        break
                if handle is None:
                    self.logger.warning(
                        f"Could not match torch GPU {gid} to an NVML handle; "
                        f"NVML data for this GPU will be empty."
                    )
                else:
                    self._nvml_handles[gid] = handle
        except Exception as e:
            self.logger.warning(f"NVML handle resolution failed: {e}")
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Phase tagging (called from training loop)
    # ------------------------------------------------------------------ #

    def set_phase(self, phase: str, step: int):
        """Update the phase label written into subsequent CSV rows.

        Optionally resets ``torch.cuda.max_memory_allocated`` so the next
        phase's peak is measured fresh. We keep that off by default to avoid
        surprising the rest of training; call ``reset_peak()`` explicitly if
        you want per-phase peaks.
        """
        self._current_phase = phase
        self._current_step = step

    def reset_peak(self):
        """Reset PyTorch's peak-memory counters on all watched GPUs."""
        for gid in self.gpu_ids:
            torch.cuda.reset_peak_memory_stats(gid)

    # ------------------------------------------------------------------ #
    #  Sampling
    # ------------------------------------------------------------------ #

    def _sample_one_gpu(self, gid: int, elapsed_s: float) -> list:
        """Build one CSV row for GPU ``gid``."""
        allocated = torch.cuda.memory_allocated(gid)
        reserved = torch.cuda.memory_reserved(gid)
        max_allocated = torch.cuda.max_memory_allocated(gid)
        total = self._gpu_total_bytes[gid]

        # NVML stats (graceful empty-string fallback)
        util_compute = ""
        util_mem = ""
        temperature = ""
        power_w = ""

        if self._nvml_available and gid in self._nvml_handles:
            try:
                pynvml = self._pynvml
                handle = self._nvml_handles[gid]
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_compute = util.gpu
                util_mem = util.memory
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                # Power is in milliwatts; convert to watts.
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = round(power_mw / 1000.0, 2)
            except Exception as e:
                # Single-sample NVML hiccups are noisy; log once at debug
                # level and keep going.
                self.logger.debug(f"NVML sample failed for GPU {gid}: {e}")

        return [
            datetime.now().isoformat(timespec="seconds"),
            round(elapsed_s, 2),
            gid,
            self._gpu_names[gid],
            round(allocated / self.BYTES_PER_GB, 3),
            round(reserved / self.BYTES_PER_GB, 3),
            round(max_allocated / self.BYTES_PER_GB, 3),
            round(total / self.BYTES_PER_GB, 3),
            util_compute,
            util_mem,
            temperature,
            power_w,
            self._current_phase,
            self._current_step,
        ]

    def _run(self):
        """Main loop of the sampler thread."""
        self.logger.info(
            f"GPUMonitor thread started; sampling every {self.interval_s}s."
        )
        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time
            try:
                for gid in self.gpu_ids:
                    row = self._sample_one_gpu(gid, elapsed)
                    self._csv_writer.writerow(row)
                self._csv_file.flush()
                # fsync is heavier than flush but ensures the row hits disk
                # even if the process is killed -9 a moment later. For OOM
                # debugging, that durability matters.
                import os
                os.fsync(self._csv_file.fileno())
            except Exception as e:
                # Never let monitor errors bring down training.
                self.logger.error(f"Sampling failed: {e}")

            # Wait with timeout so stop() unblocks promptly.
            self._stop_event.wait(self.interval_s)

        self.logger.info("GPUMonitor thread stopping.")

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            self.logger.warning("GPUMonitor.start() called but already running.")
            return
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="GPUMonitor", daemon=True
        )
        self._thread.start()

    def stop(self, join_timeout_s: float = 5.0):
        """Signal the thread to stop, join it, then close the CSV."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=join_timeout_s)
        if self._thread.is_alive():
            self.logger.warning(
                f"GPUMonitor thread did not exit within {join_timeout_s}s."
            )

        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception as e:
            self.logger.error(f"Failed to close CSV: {e}")

        if self._nvml_available and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        self.logger.info(f"GPUMonitor stopped. CSV: {self.csv_path}")


def main():
    """Standalone smoke test: monitor for 10 seconds and exit."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected.")
    main_logger.info(
        f"GPUs available: {torch.cuda.device_count()} "
        f"({torch.cuda.get_device_name(0)})"
    )

    config = {
        "csv_path": Path("./exp2/logs/gpu_monitor.csv"),
        "log_dir":  Path("./exp2/logs"),
        "interval_s": 2.0,
    }

    monitor = GPUMonitor(config)
    monitor.set_phase("smoke_test", step=0)
    monitor.start()

    # Allocate and free some memory to exercise the sampler.
    main_logger.info("Allocating 1 GiB on GPU 0...")
    x = torch.empty((1024, 1024, 256), dtype=torch.float32, device="cuda:0")
    monitor.set_phase("allocated_1gb", step=1)
    time.sleep(5)

    main_logger.info("Freeing...")
    del x
    torch.cuda.empty_cache()
    monitor.set_phase("freed", step=2)
    time.sleep(5)

    monitor.stop()
    main_logger.info(f"Done. Inspect {config['csv_path']}")


if __name__ == "__main__":
    main()