"""
Episode runner: executes exactly one labeled workload run and records
everything the experiment needs about it.

    python instrumentation/episode_runner.py \\
        --scenario ordinary_finetune \\
        --episode-id 001

An "episode" is one complete workload execution, isolated to its own
directory:

    data/raw/episodes/<episode_id>/
        metadata.json     # schema described in README.md
        shell.log         # commands executed by (only) this episode
        processes.csv      # 1 Hz process-tree sample
        nvml.csv           # 1 Hz NVML telemetry sample
        stdout.log          # workload stdout/stderr, for debugging failures

Scope note: this script only ever instruments the subprocess tree it itself
launches (see instrumentation/process_logger.py) and only ever writes to
shell.log the commands that subprocess tree runs (see workloads/common.py's
`run_shell`). It never touches your interactive shell history or unrelated
processes.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python instrumentation/episode_runner.py` (script mode,
# no package context) as well as `python -m instrumentation.episode_runner`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from instrumentation.nvml_logger import NVMLLogger, resolve_gpu_identity
from instrumentation.process_logger import ProcessLogger

REPO_ROOT = Path(__file__).resolve().parent.parent


def scenario_label(scenario_family: str) -> str:
    """Derive the ordinary/adversarial label from the scenario family name.

    We keep `label` and `scenario_family` as separate metadata fields (per
    spec) even though today label is a deterministic function of the family
    name, so that experiments/evaluate.py can do grouped-by-family analysis
    without conflating "the class" with "the specific family".
    """
    if scenario_family.startswith("ordinary_"):
        return "ordinary"
    if scenario_family.startswith("adversarial_"):
        return "adversarial"
    raise ValueError(
        f"Cannot infer label from scenario family '{scenario_family}': "
        "expected an 'ordinary_*' or 'adversarial_*' prefix."
    )


def resolve_outdir(workdir_style: str, episode_dir: Path, episode_id: str) -> Path:
    """Compute the working/output directory passed to the workload as
    --outdir, varying its shape per the experiment spec's requirement that
    directory structure not be constant across episodes.

    All three styles live under an episode-scoped path (or, for "tmp", a
    uniquely-named system temp path) so episodes never collide or leak into
    each other's artifacts.
    """
    if workdir_style == "flat":
        return episode_dir / "work"
    if workdir_style == "nested":
        return episode_dir / "work" / "project" / "runs" / f"run_{episode_id}" / "artifacts"
    if workdir_style == "tmp":
        return Path(tempfile.gettempdir()) / f"cloud_classifier_{episode_id}" / "work"
    raise ValueError(f"Unknown workdir_style '{workdir_style}'")


def build_command(scenario: str, invocation_style: str, workload_args: list[str]) -> list[str]:
    """Build the argv for launching the workload subprocess.

    Invocation style is itself part of the metadata signal we're studying
    (real workloads get started in different ways), so it's varied per
    episode by experiments/collect_episodes.py rather than fixed here.
    """
    script = f"workloads/{scenario}.py"
    if invocation_style == "direct":
        return [sys.executable, script] + workload_args
    if invocation_style == "module":
        return [sys.executable, "-m", f"workloads.{scenario}"] + workload_args
    if invocation_style == "shell_wrapper":
        return ["bash", "workloads/run_wrapper.sh", scenario] + workload_args
    raise ValueError(f"Unknown invocation_style '{invocation_style}'")


def gather_gpu_and_env_metadata(gpu_index: int) -> dict:
    identity = resolve_gpu_identity(gpu_index)

    cuda_version = None
    torch_version = None
    try:
        import torch  # imported lazily: episode_runner itself has no hard torch dependency
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception as e:
        cuda_version = f"<unavailable: {e}>"
        torch_version = f"<unavailable: {e}>"

    return {
        "gpu_name": identity.name,
        "gpu_uuid": identity.uuid,
        "driver_version": identity.driver_version,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, help="Scenario family, e.g. ordinary_finetune")
    parser.add_argument("--episode-id", required=True, help="Unique episode id, e.g. 001")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "raw"), help="Root for raw episode output")
    parser.add_argument("--gpu-index", type=int, default=0, help="NVML/CUDA device index to instrument and run on")
    parser.add_argument("--invocation-style", default="direct", choices=["direct", "module", "shell_wrapper"])
    parser.add_argument("--workdir-style", default="flat", choices=["flat", "nested", "tmp"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--params",
        default="{}",
        help="JSON object of extra --key value CLI args to forward to the workload script "
             "(e.g. '{\"model\": \"prajjwal1/bert-tiny\", \"batch_size\": 8}')",
    )
    args = parser.parse_args()

    label = scenario_label(args.scenario)
    episode_dir = Path(args.data_dir) / "episodes" / args.episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    shell_log_path = episode_dir / "shell.log"
    nvml_csv_path = episode_dir / "nvml.csv"
    processes_csv_path = episode_dir / "processes.csv"
    stdout_log_path = episode_dir / "stdout.log"
    metadata_path = episode_dir / "metadata.json"

    outdir = resolve_outdir(args.workdir_style, episode_dir, args.episode_id)
    outdir.mkdir(parents=True, exist_ok=True)

    params = json.loads(args.params)
    workload_args = ["--seed", str(args.seed), "--episode-id", args.episode_id, "--outdir", str(outdir)]
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                workload_args.append(flag)
        else:
            workload_args += [flag, str(value)]

    command = build_command(args.scenario, args.invocation_style, workload_args)

    # shell.log records only what this episode runs: the top-level launch
    # command here, plus anything the workload itself logs via
    # workloads/common.py's run_shell() (which appends to the same file,
    # located via the EPISODE_SHELL_LOG env var below).
    def log_shell(cmd_str: str) -> None:
        with open(shell_log_path, "a") as f:
            f.write(f"{time.time()}\t{cmd_str}\n")

    log_shell(" ".join(command))

    env = dict(os.environ)
    env["EPISODE_SHELL_LOG"] = str(shell_log_path)
    env["EPISODE_ID"] = args.episode_id
    env["EPISODE_LABEL"] = label
    env["EPISODE_SCENARIO_FAMILY"] = args.scenario
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    # Workloads should treat device 0 as "the assigned GPU" once
    # CUDA_VISIBLE_DEVICES scopes visibility; NVML instrumentation below
    # still addresses the real, unscoped index so telemetry stays correct.

    start_time = datetime.now(timezone.utc)
    print(f"[episode_runner] episode={args.episode_id} scenario={args.scenario} label={label}")
    print(f"[episode_runner] command: {' '.join(command)}")

    nvml_logger = NVMLLogger(gpu_index=args.gpu_index, out_path=nvml_csv_path, interval_s=1.0)
    nvml_logger.start()

    status = "success"
    error_message = ""
    return_code = None
    process_logger = None
    try:
        with open(stdout_log_path, "w") as stdout_f:
            proc = subprocess.Popen(
                command, cwd=str(REPO_ROOT), env=env,
                stdout=stdout_f, stderr=subprocess.STDOUT,
            )
            process_logger = ProcessLogger(root_pid=proc.pid, out_path=processes_csv_path, interval_s=1.0)
            process_logger.start()

            return_code = proc.wait()
            if return_code != 0:
                status = "failed"
                error_message = f"workload exited with return code {return_code}; see stdout.log"
    except Exception as e:
        status = "failed"
        error_message = f"episode_runner exception: {type(e).__name__}: {e}"
    finally:
        if process_logger is not None:
            process_logger.stop()
        nvml_logger.stop()

    end_time = datetime.now(timezone.utc)

    env_metadata = gather_gpu_and_env_metadata(args.gpu_index)

    metadata = {
        "episode_id": args.episode_id,
        "scenario_family": args.scenario,
        "label": label,
        "status": status,  # "success" or "failed" -- preprocessing must skip "failed" episodes
        "error_message": error_message,
        "return_code": return_code,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": round((end_time - start_time).total_seconds(), 3),
        "invocation_style": args.invocation_style,
        "workdir_style": args.workdir_style,
        "outdir": str(outdir),
        "seed": args.seed,
        "gpu_index": args.gpu_index,
        "command": command,
        "params": params,
        "nvml_sample_count": nvml_logger.sample_count,
        "nvml_error_count": nvml_logger.error_count,
        "process_sample_count": process_logger.sample_count if process_logger else 0,
        "process_error_count": process_logger.error_count if process_logger else 0,
        **env_metadata,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[episode_runner] status={status} duration={metadata['duration_seconds']}s -> {episode_dir}")
    if status == "failed":
        # Non-zero exit so a batch collector (experiments/collect_episodes.py)
        # can detect and record the failure without guessing from stdout.
        sys.exit(1)


if __name__ == "__main__":
    main()
