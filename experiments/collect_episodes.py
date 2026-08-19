"""
Batch episode collector.

    python experiments/collect_episodes.py --episodes 30 --seed 0

Builds the predefined scenario matrix (3 ordinary + 3 adversarial families,
replicated to fill --episodes total), randomizes *execution order* and
*GPU assignment* with --seed, then runs each episode via
instrumentation/episode_runner.py as a subprocess.

Two things are deliberately fixed rather than randomized, per the
experiment design:
  1. The scenario-family -> label mapping (ordinary_* -> ordinary,
     adversarial_* -> adversarial) -- this is the ground truth we're
     building a dataset to classify, not something to shuffle.
  2. Replicate *counts* per family are balanced as evenly as --episodes
     allows (see build_plan below).

Everything else (which episode runs in which order, which GPU it lands on,
which model/hyperparameters/invocation-style/workdir-style it uses) is
randomized from --seed, specifically so that label does not become
correlated with anything incidental like "ran first" or "ran on GPU 0"
(see the GPU-assignment note in build_plan).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from workloads.common import SMALL_DECODER_MODELS  # noqa: E402

# scenario_family -> label. This is the fixed ground truth for the dataset.
FAMILIES = {
    "ordinary_finetune": "ordinary",
    "ordinary_inference": "ordinary",
    "ordinary_training": "ordinary",
    "adversarial_finetune": "adversarial",
    "adversarial_inference": "adversarial",
    "adversarial_training": "adversarial",
}


# ---------------------------------------------------------------------------
# Per-family hyperparameter sampling.
#
# Paired families (ordinary_finetune/adversarial_finetune, etc.) intentionally
# share a sampler, so their *compute* profile (model pool, batch size, step
# count) is drawn from the same distribution -- any classifier signal that
# shows up between paired families has to come from workflow/metadata
# differences, not from us quietly giving one class bigger batches.
# ---------------------------------------------------------------------------

def _sample_finetune(rng: random.Random) -> dict:
    return {
        "model": rng.choice(SMALL_DECODER_MODELS),
        "batch_size": rng.choice([2, 4, 8]),
        "steps": rng.choice([20, 30, 40]),
        "lr": rng.choice([3e-5, 5e-5, 1e-4]),
    }


def _sample_inference(rng: random.Random) -> dict:
    return {
        "model": rng.choice(SMALL_DECODER_MODELS),
        "num_prompts": rng.choice([40, 60, 80]),
        "passes": rng.choice([1, 2, 3]),
        "max_new_tokens": rng.choice([16, 20, 24]),
    }


def _sample_training(rng: random.Random) -> dict:
    return {
        "model": rng.choice(SMALL_DECODER_MODELS),
        "batch_size": rng.choice([4, 8, 16]),
        "steps": rng.choice([40, 50, 60]),
        "lr": rng.choice([5e-4, 1e-3, 2e-3]),
    }


FAMILY_PARAM_SAMPLERS = {
    "ordinary_finetune": _sample_finetune,
    "ordinary_inference": _sample_inference,
    "ordinary_training": _sample_training,
    "adversarial_finetune": _sample_finetune,
    "adversarial_inference": _sample_inference,
    # Reuses _sample_training: adversarial_training is mechanically the same
    # from-scratch training loop as ordinary_training (see
    # workloads/adversarial_training.py), so its compute-side hyperparameter
    # distribution should match, not just its scenario shape.
    "adversarial_training": _sample_training,
}

INVOCATION_STYLES = ["direct", "module", "shell_wrapper"]
WORKDIR_STYLES = ["flat", "nested", "tmp"]


def detect_gpu_count() -> int:
    """Best-effort GPU count via `nvidia-smi -L`. Falls back to 1 (e.g. on a
    laptop with no NVIDIA GPU, so --dry-run / plan-building can still be
    exercised off of RunPod).
    """
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        return max(1, len(lines))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 1


def build_plan(total_episodes: int, seed: int, num_gpus: int) -> list[dict]:
    rng = random.Random(seed)
    families = list(FAMILIES.keys())
    n_families = len(families)

    base, remainder = divmod(total_episodes, n_families)
    if base == 0:
        raise ValueError(f"--episodes {total_episodes} is too small for {n_families} scenario families")
    counts = {f: base for f in families}
    # Distribute any remainder deterministically (fixed family iteration
    # order), so the scenario-assignment plan is reproducible; only
    # per-episode params and execution order are randomized below.
    for i in range(remainder):
        counts[families[i]] += 1

    plan = []
    for family in families:
        for replicate in range(counts[family]):
            plan.append({"scenario_family": family, "label": FAMILIES[family], "replicate": replicate})

    # Randomize execution order. This is what prevents "all ordinary
    # episodes ran first" or similar order confounds.
    rng.shuffle(plan)

    for idx, entry in enumerate(plan):
        entry["episode_id"] = f"{idx + 1:03d}"
        entry["seed"] = idx  # distinct, reproducible per-episode seed
        entry["params"] = FAMILY_PARAM_SAMPLERS[entry["scenario_family"]](rng)
        entry["invocation_style"] = rng.choice(INVOCATION_STYLES)
        entry["workdir_style"] = rng.choice(WORKDIR_STYLES)
        # GPU assignment is round-robin over the *already-shuffled* order,
        # not over the family-grouped order above -- so label and gpu_index
        # end up decorrelated even though today num_gpus is usually 1.
        entry["gpu_index"] = idx % num_gpus

    return plan


def run_plan(plan: list[dict], manifest_path: Path, data_dir: str) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def save_manifest():
        with open(manifest_path, "w") as f:
            json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "episodes": plan}, f, indent=2)

    save_manifest()  # record the full plan before running anything

    for entry in plan:
        cmd = [
            sys.executable, "instrumentation/episode_runner.py",
            "--scenario", entry["scenario_family"],
            "--episode-id", entry["episode_id"],
            "--data-dir", data_dir,
            "--gpu-index", str(entry["gpu_index"]),
            "--invocation-style", entry["invocation_style"],
            "--workdir-style", entry["workdir_style"],
            "--seed", str(entry["seed"]),
            "--params", json.dumps(entry["params"]),
        ]
        print(f"\n=== episode {entry['episode_id']} ({entry['scenario_family']}, gpu={entry['gpu_index']}) ===")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        entry["run_status"] = "success" if result.returncode == 0 else "failed"
        save_manifest()  # persist progress after every episode, not just at the end

    n_ok = sum(1 for e in plan if e["run_status"] == "success")
    print(f"\n[collect_episodes] {n_ok}/{len(plan)} episodes completed successfully. Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=None, help="Override auto-detected GPU count")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running anything")
    args = parser.parse_args()

    num_gpus = args.num_gpus if args.num_gpus is not None else detect_gpu_count()
    plan = build_plan(args.episodes, args.seed, num_gpus)

    print(f"[collect_episodes] {len(plan)} episodes across {len(FAMILIES)} families, {num_gpus} GPU(s) detected")
    for entry in plan:
        print(f"  {entry['episode_id']}  {entry['scenario_family']:<26} label={entry['label']:<8} "
              f"gpu={entry['gpu_index']} invocation={entry['invocation_style']:<13} workdir={entry['workdir_style']}")

    if args.dry_run:
        print("[collect_episodes] --dry-run: not executing.")
        return

    manifest_path = Path(args.data_dir) / "collection_manifest.json"
    run_plan(plan, manifest_path, args.data_dir)


if __name__ == "__main__":
    main()
