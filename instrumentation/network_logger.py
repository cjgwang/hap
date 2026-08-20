"""
Network telemetry logger.

Samples, at ~1 Hz, network-level metadata for this episode:
  - bytes_sent_delta / bytes_recv_delta: system-wide network throughput
    since the previous tick (psutil has no per-process byte counter, so
    this is the one field in this file that is NOT scoped to just the
    episode's process tree -- see CAVEAT below).
  - remote_addresses: the set of "ip:port" pairs with an ESTABLISHED
    connection whose owning pid is in this episode's own process tree
    (psutil.net_connections() carries a pid per connection, so this part
    IS correctly scoped, unlike the byte counts).

Written to network.csv: timestamp, bytes_sent_delta, bytes_recv_delta,
remote_addresses (semicolon-joined), error.

CAVEAT: byte counts are system-wide. On a GPU pod running one job at a
time (this experiment's target environment) that's a reasonable proxy for
"this episode's network activity" -- it is not a correct measurement on a
shared/multi-tenant host. This is recorded explicitly here rather than
silently treated as per-process.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Optional

import psutil

NETWORK_FIELDS = ["timestamp", "bytes_sent_delta", "bytes_recv_delta", "remote_addresses", "error"]


class NetworkLogger:
    def __init__(self, root_pid: int, out_path: Path, interval_s: float = 1.0):
        self.root_pid = root_pid
        self.out_path = Path(out_path)
        self.interval_s = interval_s

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_counters: Optional[tuple[int, int]] = None
        self.error_count = 0
        self.sample_count = 0

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.out_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=NETWORK_FIELDS)
        self._writer.writeheader()
        self._file.flush()

        counters = psutil.net_io_counters()
        self._last_counters = (counters.bytes_sent, counters.bytes_recv)

        self._thread = threading.Thread(target=self._run, name="network-logger", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=self.interval_s * 5 + 5)
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()

    def _tracked_pids(self) -> set[int]:
        try:
            root = psutil.Process(self.root_pid)
        except psutil.NoSuchProcess:
            return set()
        pids = {self.root_pid}
        try:
            pids.update(p.pid for p in root.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return pids

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            self._sample_once()
            next_tick += self.interval_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                self._stop_event.wait(sleep_s)
            else:
                next_tick = time.monotonic()

    def _sample_once(self) -> None:
        row = {k: "" for k in NETWORK_FIELDS}
        row["timestamp"] = time.time()
        errors = []

        try:
            counters = psutil.net_io_counters()
            sent, recv = counters.bytes_sent, counters.bytes_recv
            last_sent, last_recv = self._last_counters
            row["bytes_sent_delta"] = sent - last_sent
            row["bytes_recv_delta"] = recv - last_recv
            self._last_counters = (sent, recv)
        except Exception as e:
            errors.append(f"net_io_counters: {type(e).__name__}: {e}")

        try:
            pids = self._tracked_pids()
            conns = psutil.net_connections(kind="inet")
            addrs = {
                f"{c.raddr.ip}:{c.raddr.port}"
                for c in conns
                if c.pid in pids and c.status == psutil.CONN_ESTABLISHED and c.raddr
            }
            row["remote_addresses"] = ";".join(sorted(addrs))
        except Exception as e:
            errors.append(f"net_connections: {type(e).__name__}: {e}")

        row["error"] = "; ".join(errors)
        if errors:
            self.error_count += 1

        self._writer.writerow(row)
        self._file.flush()
        self.sample_count += 1
