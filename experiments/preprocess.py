"""
Raw -> processed preprocessing.

    python experiments/preprocess.py

Reads every episode under data/raw/episodes/, and writes THREE new files
under data/processed/ (never touching data/raw/):

  episodes_index.csv    one row per episode: id, label, family, status, ...
  text_features.csv     one row per episode: a text blob built ONLY from
                         job/container, network, and storage I/O metadata
                         -- see "TEXT FEATURE SCOPE" below
  nvml_features.csv     one row per episode: engineered temporal features
                         over the 1 Hz NVML trace, for
                         train_nvml_classifier.py (mirrors the feature-
                         engineering approach in the referenced paper)

Episodes whose metadata.json status != "success" are excluded entirely
(never silently coerced into a valid example) and are listed in the
printed summary.

TEXT FEATURE SCOPE
-------------------
text_features.csv deliberately contains NONE of: shell.log commands,
process names, or process command lines. An earlier version of this
script included them, and it was a real label leak -- the workload
script's filename (e.g. "workloads/adversarial_inference.py") appears in
the launch command regardless of invocation style, and scenario_family
determines label, so a classifier trained on that text was largely just
reading the answer off the command line rather than learning anything
about infrastructure-level metadata.

The text is now built ONLY from three categories a real compute provider's
control plane could plausibly see without inspecting what's running
inside a job:
  1. job/container metadata -- GPU name/index, CPU count, system RAM,
     GPU memory capacity, runtime versions, duration (see
     build_job_metadata_text, sourced from metadata.json)
  2. network metadata -- bytes sent/received, count of distinct remote
     addresses contacted (see build_network_text, sourced from
     network.csv)
  3. storage I/O metadata -- total bytes read/written across the
     episode's process tree (see build_storage_text, sourced from
     processes.csv's read_bytes/write_bytes columns)

metadata.json's `params` field (model choice, dataset name, hyperparameters)
is intentionally EXCLUDED even though it isn't a raw command -- some of
its keys (is_safe_filter, prompt_harm_label_filter) directly encode the
label, and including it would just relocate the same leak.

Text normalization (normalize_text) still strips run-specific artifacts
(timestamps, UUIDs, seeds, absolute paths, episode IDs) from whatever text
does get built, for any future text source that needs it.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_ISO_TIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+\d{2}:\d{2}|Z)?")
_EPOCH_TIME_RE = re.compile(r"\b\d{10}\.\d{2,9}\b")  # shell.log / nvml.csv style unix timestamps
_UUID_RE = re.compile(r"\b(GPU-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_ABS_PATH_RE = re.compile(r"(?:/[\w.\-]+){2,}/?")  # 2+ leading "/segment" path components
_SEED_FLAG_RE = re.compile(r"(--seed[= ])(\d+)")
_EPISODE_ID_FLAG_RE = re.compile(r"(--episode-id[= ])(\S+)")
_ID_TOKEN_RE = re.compile(r"\b(run_|cc_|episode_)(\d{3,})\b")


def normalize_text(text: str) -> str:
    text = _ISO_TIME_RE.sub("<TIME>", text)
    text = _EPOCH_TIME_RE.sub("<TIME>", text)
    text = _UUID_RE.sub("<UUID>", text)
    # Paths before bare ID tokens: a path like /tmp/cloud_classifier_007/work
    # gets collapsed to <PATH> wholesale, which also removes the embedded
    # episode id -- that's fine, "episode-specific IDs" inside a path are
    # still a run-specific artifact either way.
    text = _ABS_PATH_RE.sub("<PATH>", text)
    text = _SEED_FLAG_RE.sub(r"\1<SEED>", text)
    text = _EPISODE_ID_FLAG_RE.sub(r"\1<ID>", text)
    text = _ID_TOKEN_RE.sub("<ID>", text)
    return text


# ---------------------------------------------------------------------------
# Episode discovery + loading
# ---------------------------------------------------------------------------

def parse_episode_id_ranges(spec: str) -> set[str] | None:
    """Parse '060-090,100-130' (or single ids, e.g. '005,007') into the set
    of zero-padded 3-digit episode_id strings it selects. Returns None for
    a falsy spec, meaning "no filter, include every episode" -- callers
    should treat None and "select everything" as the same case.
    """
    if not spec:
        return None
    ids: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            ids.update(f"{n:03d}" for n in range(int(lo), int(hi) + 1))
        else:
            ids.add(f"{int(part):03d}")
    return ids


def load_episodes(raw_dir: Path, episode_id_filter: set[str] | None = None) -> list[dict]:
    episodes_root = raw_dir / "episodes"
    episodes = []
    n_skipped_by_filter = 0
    for ep_dir in sorted(episodes_root.iterdir()):
        if episode_id_filter is not None and ep_dir.name not in episode_id_filter:
            n_skipped_by_filter += 1
            continue
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            print(f"[preprocess] WARNING: {ep_dir} has no metadata.json, skipping")
            continue
        with open(meta_path) as f:
            metadata = json.load(f)
        episodes.append({"dir": ep_dir, "metadata": metadata})
    if episode_id_filter is not None:
        print(f"[preprocess] --episode-ids filter active: {n_skipped_by_filter} episode(s) on disk excluded")
    return episodes


# ---------------------------------------------------------------------------
# Text features -- job/container + network + storage I/O metadata ONLY.
# See the module docstring's "TEXT FEATURE SCOPE" section for why shell
# commands and process names/command lines are excluded.
# ---------------------------------------------------------------------------

# Whitelisted metadata.json fields for the job/container text. Deliberately
# NOT the whole metadata dict: excludes `command` (contains the workload
# script name -- the original leak) and `params` (some keys encode the
# label directly, e.g. is_safe_filter).
JOB_METADATA_FIELDS = [
    "gpu_name", "gpu_index", "driver_version", "cuda_version", "torch_version",
    "python_version", "cpu_count", "system_ram_gb", "gpu_memory_total_gb",
    "duration_seconds",
]


def build_job_metadata_text(metadata: dict) -> str:
    return " ".join(f"{field}={metadata.get(field)}" for field in JOB_METADATA_FIELDS)


def build_network_text(ep_dir: Path) -> str:
    network_csv = ep_dir / "network.csv"
    if not network_csv.exists():
        return "network_bytes_sent=0 network_bytes_recv=0 network_unique_destinations=0"
    try:
        df = pd.read_csv(network_csv)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if df.empty:
        return "network_bytes_sent=0 network_bytes_recv=0 network_unique_destinations=0"

    total_sent = pd.to_numeric(df.get("bytes_sent_delta"), errors="coerce").fillna(0).sum()
    total_recv = pd.to_numeric(df.get("bytes_recv_delta"), errors="coerce").fillna(0).sum()

    all_addrs: set[str] = set()
    if "remote_addresses" in df.columns:
        for val in df["remote_addresses"].dropna():
            all_addrs.update(a for a in str(val).split(";") if a)

    return (
        f"network_bytes_sent={int(total_sent)} network_bytes_recv={int(total_recv)} "
        f"network_unique_destinations={len(all_addrs)}"
    )


def build_storage_text(ep_dir: Path) -> str:
    processes_csv = ep_dir / "processes.csv"
    if not processes_csv.exists():
        return "storage_read_bytes=0 storage_write_bytes=0"
    try:
        df = pd.read_csv(processes_csv)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if df.empty or "read_bytes" not in df.columns or "write_bytes" not in df.columns:
        return "storage_read_bytes=0 storage_write_bytes=0"

    df = df.copy()
    df["read_bytes"] = pd.to_numeric(df["read_bytes"], errors="coerce")
    df["write_bytes"] = pd.to_numeric(df["write_bytes"], errors="coerce")
    # read_bytes/write_bytes are cumulative per-process counters (see
    # instrumentation/process_logger.py), so the max observed value per pid
    # -- summed across all pids in the tree -- approximates total I/O for
    # the episode. This slightly undercounts a process's true total if it
    # exits between its last sample and process exit, which is an accepted
    # approximation, not a silent one.
    per_pid_max = df.groupby("pid")[["read_bytes", "write_bytes"]].max()
    total_read = per_pid_max["read_bytes"].sum()
    total_write = per_pid_max["write_bytes"].sum()

    return (
        f"storage_read_bytes={int(total_read) if pd.notna(total_read) else 0} "
        f"storage_write_bytes={int(total_write) if pd.notna(total_write) else 0}"
    )


def build_text_row(ep_dir: Path, metadata: dict) -> str:
    return " ".join([
        build_job_metadata_text(metadata),
        build_network_text(ep_dir),
        build_storage_text(ep_dir),
    ])


# ---------------------------------------------------------------------------
# NVML temporal features (engineered, mirroring the referenced paper)
# ---------------------------------------------------------------------------

NVML_NUMERIC_COLS = [
    "gpu_utilization", "memory_utilization", "memory_used_mb", "power_w",
    "temperature_c", "sm_clock_mhz", "memory_clock_mhz", "pcie_tx_mb", "pcie_rx_mb",
]


def _slope(values: list[float]) -> float:
    """Simple least-squares slope of value vs. sample index (a cheap proxy
    for "ramping up" vs. "flat" telemetry, same spirit as the temporal
    features in the referenced paper).
    """
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den else 0.0


def build_nvml_features(ep_dir: Path) -> dict:
    nvml_csv = ep_dir / "nvml.csv"
    features = {}
    if not nvml_csv.exists():
        return features
    try:
        df = pd.read_csv(nvml_csv)
    except pd.errors.EmptyDataError:
        return features

    features["nvml_sample_count"] = len(df)
    features["nvml_error_rate"] = float((df["error"].fillna("") != "").mean()) if "error" in df.columns and len(df) else 0.0

    for col in NVML_NUMERIC_COLS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
        if not values:
            for stat in ("mean", "std", "min", "max", "slope"):
                features[f"{col}_{stat}"] = 0.0
            continue
        features[f"{col}_mean"] = statistics.fmean(values)
        features[f"{col}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        features[f"{col}_min"] = min(values)
        features[f"{col}_max"] = max(values)
        features[f"{col}_slope"] = _slope(values)

    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--processed-dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument(
        "--episode-ids", default=None,
        help="Only include these episode ids, e.g. '060-090,100-130'. Default: all episodes "
             "on disk. Everything downstream (train_*.py, evaluate.py) reads only the CSVs "
             "this script writes, so this is the one place to scope a training run to a "
             "specific subset without touching data/raw/.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    episode_id_filter = parse_episode_id_ranges(args.episode_ids)
    episodes = load_episodes(raw_dir, episode_id_filter)

    index_rows, text_rows, nvml_rows = [], [], []
    n_success, n_failed = 0, 0

    for ep in episodes:
        meta = ep["metadata"]
        episode_id = meta["episode_id"]

        if meta.get("status") != "success":
            n_failed += 1
            print(f"[preprocess] excluding episode {episode_id}: status={meta.get('status')!r} "
                  f"({meta.get('error_message', 'no error message recorded')})")
            continue
        n_success += 1

        index_rows.append({
            "episode_id": episode_id,
            "label": meta["label"],
            "scenario_family": meta["scenario_family"],
            "status": meta["status"],
            "duration_seconds": meta.get("duration_seconds"),
            "gpu_name": meta.get("gpu_name"),
            "invocation_style": meta.get("invocation_style"),
            "workdir_style": meta.get("workdir_style"),
        })

        raw_text = build_text_row(ep["dir"], meta)
        text_rows.append({
            "episode_id": episode_id,
            "label": meta["label"],
            "scenario_family": meta["scenario_family"],
            "text_raw": raw_text,
            "text_normalized": normalize_text(raw_text),
        })

        nvml_feats = build_nvml_features(ep["dir"])
        nvml_rows.append({
            "episode_id": episode_id,
            "label": meta["label"],
            "scenario_family": meta["scenario_family"],
            **nvml_feats,
        })

    pd.DataFrame(index_rows).to_csv(processed_dir / "episodes_index.csv", index=False)
    pd.DataFrame(text_rows).to_csv(processed_dir / "text_features.csv", index=False)
    pd.DataFrame(nvml_rows).to_csv(processed_dir / "nvml_features.csv", index=False)

    print(f"\n[preprocess] {n_success} usable episodes, {n_failed} excluded (non-success status)")
    print(f"[preprocess] wrote {processed_dir / 'episodes_index.csv'}")
    print(f"[preprocess] wrote {processed_dir / 'text_features.csv'}")
    print(f"[preprocess] wrote {processed_dir / 'nvml_features.csv'}")


if __name__ == "__main__":
    main()
