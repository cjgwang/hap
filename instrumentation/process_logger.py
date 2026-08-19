"""
Process-tree logger.

Samples, at ~1 Hz, the process metadata for a single root PID (the workload
subprocess launched by episode_runner) and all of its current descendants.
This intentionally does NOT scan the whole system process table for
unrelated users/processes -- it only ever walks psutil.Process(root_pid)'s
own subtree, so a run on a shared or multi-tenant box cannot leak other
sessions' process metadata into episode data.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Optional

import psutil


PROCESS_FIELDS = [
    "timestamp",
    "pid",
    "process_name",
    "command_line",
    "cpu_percent",
    "memory_mb",
]


class ProcessLogger:
    """Background-thread process-tree sampler scoped to one root PID.

    Usage:
        logger = ProcessLogger(root_pid=proc.pid, out_path=episode_dir / "processes.csv")
        logger.start()
        ... wait for workload subprocess to finish ...
        logger.stop()
    """

    def __init__(self, root_pid: int, out_path: Path, interval_s: float = 1.0):
        self.root_pid = root_pid
        self.out_path = Path(out_path)
        self.interval_s = interval_s

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error_count = 0
        self.sample_count = 0

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.out_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=PROCESS_FIELDS)
        self._writer.writeheader()
        self._file.flush()

        # cpu_percent() is meaningless on its own first call (it measures
        # delta since the *previous* call), so prime it once here for every
        # process we can already see. Subsequent per-tick calls then report
        # real interval averages instead of a meaningless 0.0 on tick 1.
        for p in self._current_tree():
            try:
                p.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        self._thread = threading.Thread(target=self._run, name="process-logger", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=self.interval_s * 5 + 5)
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()

    def _current_tree(self):
        """Root process + all live descendants. Processes that exit between
        listing and sampling are skipped (not errors -- just a race with a
        short-lived child), which we log at the row level via `error`.
        """
        try:
            root = psutil.Process(self.root_pid)
        except psutil.NoSuchProcess:
            return []
        procs = [root]
        try:
            procs.extend(root.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return procs

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
        ts = time.time()
        rows_written = 0
        for p in self._current_tree():
            row = {k: "" for k in PROCESS_FIELDS}
            row["timestamp"] = ts
            row["pid"] = p.pid
            try:
                row["process_name"] = p.name()
                row["command_line"] = " ".join(p.cmdline())
                row["cpu_percent"] = p.cpu_percent(None)
                row["memory_mb"] = round(p.memory_info().rss / (1024 ** 2), 3)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                # Process exited mid-sample or we lost permission to read it.
                # Record what we can (pid, timestamp) and note the failure
                # explicitly rather than dropping the row.
                row["process_name"] = f"<error: {type(e).__name__}>"
                self.error_count += 1
            self._writer.writerow(row)
            rows_written += 1
        if rows_written == 0:
            # Root process (and therefore the whole tree) is already gone.
            # Still emit one row so gaps in the CSV are visible/explainable
            # rather than silently absent.
            self._writer.writerow({
                "timestamp": ts,
                "pid": self.root_pid,
                "process_name": "<error: root process not found>",
                "command_line": "",
                "cpu_percent": "",
                "memory_mb": "",
            })
            self.error_count += 1
        self._file.flush()
        self.sample_count += 1
