"""
Raw -> processed preprocessing.

    python experiments/preprocess.py

Reads every episode under data/raw/episodes/, and writes THREE new files
under data/processed/ (never touching data/raw/):

  episodes_index.csv    one row per episode: id, label, family, status, ...
  text_features.csv     one row per episode: raw + normalized combined
                         shell/process text, for train_text_classifier.py
  nvml_features.csv     one row per episode: engineered temporal features
                         over the 1 Hz NVML trace, for
                         train_nvml_classifier.py (mirrors the feature-
                         engineering approach in the referenced paper)

Episodes whose metadata.json status != "success" are excluded entirely
(never silently coerced into a valid example) and are listed in the
printed summary.

Text normalization: we only strip obvious *run-specific* artifacts
(timestamps, UUIDs, seed values, absolute paths, episode IDs), not
semantic content -- model names, dataset/domain labels, and task/scenario
words are left untouched. A separate semantic-ablation pass is a later,
different experiment, not this one.
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

def load_episodes(raw_dir: Path) -> list[dict]:
    episodes_root = raw_dir / "episodes"
    episodes = []
    for ep_dir in sorted(episodes_root.iterdir()):
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            print(f"[preprocess] WARNING: {ep_dir} has no metadata.json, skipping")
            continue
        with open(meta_path) as f:
            metadata = json.load(f)
        episodes.append({"dir": ep_dir, "metadata": metadata})
    return episodes


# ---------------------------------------------------------------------------
# Text features
# ---------------------------------------------------------------------------

def build_text_row(ep_dir: Path) -> str:
    parts = []

    shell_log = ep_dir / "shell.log"
    if shell_log.exists():
        for line in shell_log.read_text().splitlines():
            # format: "<epoch_ts>\t<command>"
            _, _, cmd = line.partition("\t")
            if cmd:
                parts.append(cmd)

    processes_csv = ep_dir / "processes.csv"
    if processes_csv.exists():
        try:
            df = pd.read_csv(processes_csv)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        if not df.empty:
            for col in ("process_name", "command_line"):
                if col in df.columns:
                    parts.extend(df[col].dropna().astype(str).unique().tolist())

    return "\n".join(parts)


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
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(raw_dir)

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

        raw_text = build_text_row(ep["dir"])
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
