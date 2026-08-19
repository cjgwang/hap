"""
NVML telemetry logger.

Samples GPU telemetry at ~1 Hz from a background thread and writes one CSV
row per sample to `nvml.csv` inside an episode directory. This is the
content-agnostic signal used by experiments/train_nvml_classifier.py, modeled
on the sampling approach in the referenced paper (1 Hz NVML sampling +
engineered temporal features + Random Forest).

Design notes:
  - We identify the GPU by UUID (stable across reboots / index reassignment),
    not just by index, per the experiment spec. The index is still needed to
    open the NVML handle, so we resolve index -> handle -> UUID once at
    start() and record both.
  - NVML calls can transiently fail (e.g. driver hiccup, permission blip,
    a metric unsupported on a given GPU/driver combo). We do NOT let a single
    failed metric drop the whole sample, and we do NOT silently swallow the
    error: each row has an `error` column that lists which fields failed and
    why. Only if pynvml itself cannot be queried at all for a given tick do we
    write a row with all metrics empty and an error message.
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import pynvml


# NVML fields written per sample. Keep in one place so the CSV header and the
# per-tick sampling logic can't drift apart.
NVML_FIELDS = [
    "timestamp",
    "gpu_index",
    "gpu_uuid",
    "gpu_utilization",  # percent, 0-100
    "memory_utilization",  # percent, 0-100
    "memory_used_mb",
    "power_w",
    "temperature_c",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "pcie_tx_mb",  # cumulative-rate sample (KB/s from NVML, converted to MB/s)
    "pcie_rx_mb",
    "error",  # semicolon-separated list of "<field>: <exception>", or empty
]


@dataclass
class GPUIdentity:
    index: int
    uuid: str
    name: str
    driver_version: str


def resolve_gpu_identity(gpu_index: int) -> GPUIdentity:
    """Open an NVML handle for `gpu_index` and read its stable identity fields.

    Call this once up front (e.g. from episode_runner) so metadata.json and
    the NVML logger agree on exactly which physical GPU was used, even if
    something later remaps device indices.
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    name = pynvml.nvmlDeviceGetName(handle)
    driver_version = pynvml.nvmlSystemGetDriverVersion()
    # Older pynvml versions return bytes; normalize to str.
    if isinstance(uuid, bytes):
        uuid = uuid.decode()
    if isinstance(name, bytes):
        name = name.decode()
    if isinstance(driver_version, bytes):
        driver_version = driver_version.decode()
    return GPUIdentity(index=gpu_index, uuid=uuid, name=name, driver_version=driver_version)


class NVMLLogger:
    """Background-thread NVML sampler.

    Usage:
        logger = NVMLLogger(gpu_index=0, out_path=episode_dir / "nvml.csv")
        logger.start()
        ... run workload ...
        logger.stop()
    """

    def __init__(self, gpu_index: int, out_path: Path, interval_s: float = 1.0):
        self.gpu_index = gpu_index
        self.out_path = Path(out_path)
        self.interval_s = interval_s

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml_owned = False  # whether this instance called nvmlInit()
        self._handle = None
        self.error_count = 0
        self.sample_count = 0

    def start(self) -> None:
        # pynvml.nvmlInit() is reference-counted internally by NVML, so it is
        # safe to call even if something else in-process already initialized
        # it; we always pair our own init with our own shutdown.
        pynvml.nvmlInit()
        self._nvml_owned = True
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.out_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=NVML_FIELDS)
        self._writer.writeheader()
        self._file.flush()

        self._thread = threading.Thread(target=self._run, name="nvml-logger", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=self.interval_s * 5 + 5)
        if self._nvml_owned:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                # Shutdown failing is not something we can recover from or
                # meaningfully log at this point (the CSV is already closed
                # below); it does not affect data already collected.
                pass
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            row = self._sample_once()
            self._writer.writerow(row)
            self._file.flush()  # flush every sample: episodes are short and
            # we would rather pay a small perf cost than lose a partially
            # written file if the workload process is killed.
            self.sample_count += 1
            if row["error"]:
                self.error_count += 1

            next_tick += self.interval_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                self._stop_event.wait(sleep_s)
            else:
                # We fell behind (e.g. a slow NVML call); resync instead of
                # busy-looping to catch up.
                next_tick = time.monotonic()

    def _sample_once(self) -> dict:
        row = {k: "" for k in NVML_FIELDS}
        row["timestamp"] = time.time()
        row["gpu_index"] = self.gpu_index
        errors = []

        def safe(field_name, fn):
            try:
                row[field_name] = fn()
            except pynvml.NVMLError as e:
                errors.append(f"{field_name}: {e}")
            except Exception as e:  # defensive: never let one metric crash sampling
                errors.append(f"{field_name}: unexpected {type(e).__name__}: {e}")

        h = self._handle
        safe("gpu_uuid", lambda: _decode(pynvml.nvmlDeviceGetUUID(h)))

        def util():
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            return u.gpu
        safe("gpu_utilization", util)

        def mem_util():
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            return u.memory
        safe("memory_utilization", mem_util)

        def mem_used():
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            return round(m.used / (1024 ** 2), 3)
        safe("memory_used_mb", mem_used)

        def power():
            return round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 3)  # mW -> W
        safe("power_w", power)

        def temp():
            return pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        safe("temperature_c", temp)

        def sm_clock():
            return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        safe("sm_clock_mhz", sm_clock)

        def mem_clock():
            return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
        safe("memory_clock_mhz", mem_clock)

        def pcie_tx():
            # NVML reports PCIe throughput counters in KB/s.
            return round(pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0, 3)
        safe("pcie_tx_mb", pcie_tx)

        def pcie_rx():
            return round(pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0, 3)
        safe("pcie_rx_mb", pcie_rx)

        row["error"] = "; ".join(errors)
        return row


def _decode(x):
    return x.decode() if isinstance(x, bytes) else x
