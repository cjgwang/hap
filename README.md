# Cloud Classifiers

Repo for a sprint during the Hardware Assurance Programme.

## Research question

Building on [arXiv:2606.19262](https://arxiv.org/pdf/2606.19262) (which
classifies GPU workloads as training/inference/non-ML using content-agnostic
1 Hz NVML telemetry + engineered temporal features + Random Forest), this
experiment asks a different question:

> Does job/container, network, and storage I/O metadata visible to a
> compute provider add useful signal, beyond NVML telemetry alone, for
> distinguishing **ordinary** ML activity from synthetic
> **adversarial-proxy** activity?

Two synthetic classes:

- **ordinary** -- benign, normal ML development and inference activity.
- **adversarial** -- synthetic, benign *proxy* workloads representing activity a
  provider might reasonably flag for additional (routine) safety review.
  These are still completely benign; only their metadata (filenames,
  project/dataset labels, workflow shape) is flavored to resemble something
  review-worthy. No workload in this repo does anything actually harmful,
  and the goal is not to detect real malicious behavior.

The two classes are designed so that, within all three paired scenario
families (finetune / inference / training), the **compute is mechanically
identical** between ordinary and adversarial versions -- so any
classification signal in that pair has to come from workflow/metadata, not
from us quietly making the "risky" workload compute-heavier.

## Project structure

```
cloud-classifier/
    README.md
    requirements.txt
    setup.sh                       # RunPod environment setup (see below)

    workloads/                     # the actual scenario scripts (see table below)
        common.py                  # shared harness: shell logging, seeding, small models/corpora
        run_wrapper.sh              # generic shell-invocation wrapper (see "invocation styles")
        ordinary_*.py / adversarial_*.py

    instrumentation/
        nvml_logger.py             # 1 Hz NVML telemetry sampler
        process_logger.py          # 1 Hz process-tree sampler, scoped to the episode's own subtree
        episode_runner.py          # runs ONE episode end-to-end and writes its output directory

    experiments/
        collect_episodes.py        # runs the full scenario matrix (batch of episodes)
        preprocess.py               # data/raw/ -> data/processed/ (never mutates raw data)
        train_text_classifier.py    # TF-IDF + logistic regression over job/network/storage metadata
        train_nvml_classifier.py    # engineered NVML features + Random Forest
        train_combined_classifier.py # text + NVML features fused, one Random Forest
        evaluate.py                  # combines/compares the classifiers above
        api_baseline.py              # optional: zero-shot Claude API baseline
        plot_results.py              # 3 PNG charts from episodes_index.csv + comparison_report.json

    data/
        raw/episodes/<id>/         # one directory per episode, see schema below
        processed/                  # derived features, never raw data
    results/                        # classifier outputs + comparison report
```

## Scenario families

| Ordinary family              | Adversarial family            | Shared compute? |
|-------------------------------|-------------------------------|------------------|
| `ordinary_finetune`           | `adversarial_finetune`        | yes -- identical SFT (`train_causal_lm_sft`) loop, pretrained weights |
| `ordinary_inference`          | `adversarial_inference`       | yes -- identical generation loop |
| `ordinary_training`           | `adversarial_training`        | yes -- identical SFT loop, random-init weights |

All three pairs draw from the SAME real public dataset within the pair and
differ only in which label value they filter for:

- **`ordinary_finetune` / `adversarial_finetune`** and **`ordinary_training`
  / `adversarial_training`** both draw real (prompt, response) pairs from
  [`PKU-Alignment/PKU-SafeRLHF-QA`](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-QA)
  -- an academic safety-alignment dataset (the same genre as Anthropic's
  HH-RLHF, built to help train models to be *less* harmful) -- filtered on
  its own `is_safe` annotation: the `ordinary_*` script uses `is_safe=True`
  rows, the `adversarial_*` script uses `is_safe=False` rows. Both do
  proper prompt-masked SFT (see "Training details" below). `finetune`
  starts from **pretrained** weights; `training` starts from a
  **randomly-initialized** model of the same architecture -- that's the
  only mechanical difference between the two pairs. Ungated, no HF login
  needed.
- **`ordinary_inference` / `adversarial_inference`** both draw real prompts
  from [`allenai/wildguardmix`](https://huggingface.co/datasets/allenai/wildguardmix)
  (`wildguardtest` config) -- a public AI2 benchmark built for training/evaluating
  content-moderation models -- filtered on its own `prompt_harm_label`
  annotation: `ordinary_inference` generates completions for
  `prompt_harm_label="unharmful"` prompts, `adversarial_inference` for
  `prompt_harm_label="harmful"` prompts. **This dataset is gated** -- see
  the RunPod setup step below.

Because the dataset *name* is identical between the ordinary and
adversarial episode of every pair -- only the label filter differs -- any
signal a classifier finds has to come from the label itself (and whatever
happens to be downstream of it, like output content), not from a
different-looking dataset name. See `workloads/common.py`'s module
docstring and each script's own docstring for the full reasoning on why
these are still benign workloads: models here are small (<=1.5B parameter)
public checkpoints (see `SMALL_DECODER_MODELS` in `workloads/common.py`)
doing at most a few dozen gradient steps over a couple hundred examples
per episode -- not enough exposure to meaningfully shift real-world
capability regardless of which label subset a given episode touches. This
is a narrower claim than "the model is a non-agentic toy" (true of a
2-layer GPT-2, not quite true of a real instruction-tuned model), which is
exactly why the model pool stays at 1.5B and below.

## RunPod setup

Copy-pasteable commands for a fresh RunPod GPU pod (H100 preferred, but the
code is GPU-model agnostic and detects everything at runtime -- nothing here
assumes a specific GPU or CUDA version).

**1. Create/connect to the pod.** From the RunPod console: deploy a pod with
an NVIDIA GPU (H100 if available) using a PyTorch or CUDA base template, then
connect via the provided SSH command, e.g.:

```bash
ssh root@<pod-ip> -p <pod-ssh-port> -i ~/.ssh/id_ed25519
```

Or use RunPod's web terminal if you don't have SSH set up.

**2. Clone/copy this repo onto the pod**, then `cd` into it:

```bash
cd /workspace
git clone <this-repo-url> cloud-classifier   # or scp/rsync it up
cd cloud-classifier
```

**3. Verify the NVIDIA driver, CUDA, and GPU are visible:**

```bash
nvidia-smi
```

You should see your GPU listed along with a driver version and a "CUDA
Version" (the max CUDA version the driver supports -- not necessarily the
version any installed toolkit uses).

**4. Run the setup script.** This creates a virtualenv, installs
dependencies, and verifies both PyTorch GPU access and NVML access,
failing loudly with diagnostics if either is broken:

```bash
bash setup.sh
```

Under the hood, `setup.sh`:

- creates `./venv` with `--system-site-packages` (so it reuses RunPod's
  preinstalled, driver-matched PyTorch when the pod image ships one, instead
  of forcing a slow/possibly-mismatched reinstall);
- runs `pip install -r requirements.txt` (installs a CUDA-enabled PyTorch
  wheel via pip's normal resolution if one isn't already present -- no
  hardcoded `cuXXX` version);
- runs `python -c "import torch; assert torch.cuda.is_available()"` and
  prints the detected device name;
- runs `pynvml.nvmlInit()` / `nvmlDeviceGetCount()` and prints each GPU's
  name and UUID.

If step 4 or 5 fails, the script prints a suggested manual fix (installing a
specific PyTorch build matching the driver's reported CUDA version from
<https://pytorch.org/get-started/locally/>).

**5. (Manual equivalent, if you want to run it by hand instead of `setup.sh`):**

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlDeviceGetCount())"
```

**6. Activate the environment in any new shell:**

```bash
source venv/bin/activate
```

**7. Authenticate with Hugging Face (required for `ordinary_inference` /
`adversarial_inference`).** Both pull prompts from `allenai/wildguardmix`,
which is a gated dataset -- you must accept its terms on the dataset page
and log in with a token that has access, or those two families' episodes
will fail at the `load_dataset` call (recorded as `"status": "failed"` in
`metadata.json`, not a silent skip):

```bash
huggingface-cli login   # paste a token from https://huggingface.co/settings/tokens
# or, non-interactively:
export HF_TOKEN=hf_...
```

`ordinary_training` / `adversarial_training` use `PKU-Alignment/PKU-SafeRLHF-QA`,
which is ungated and needs no login.

## Training details

All three families vary their hyperparameters per episode (see
`experiments/collect_episodes.py`'s `_sample_*` functions) within the
ranges below, both to keep episodes within the ~2-4 min target and so a
classifier can't just memorize one exact invocation. `ordinary_*` and
`adversarial_*` always share the same sampler/range within a pair -- the
point is that compute should look the same; only workflow/dataset metadata
should differ.

| Family | Model weights | Dataset | Objective | Steps | Batch size | LR |
|---|---|---|---|---|---|---|
| `*_finetune` | pretrained (`Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`) | PKU-SafeRLHF-QA (`is_safe`) | SFT, prompt-masked (`train_causal_lm_sft`) | 20-40 | 2-8 | 3e-5-1e-4 |
| `*_inference` | pretrained, same pool | WildGuardMix (`prompt_harm_label`) | no training -- `model.generate()`, sampled (`do_sample=True, top_k=50`) | n/a | n/a | n/a |
| `*_training` | **randomly initialized** (`AutoModelForCausalLM.from_config`, no pretrained weights) | PKU-SafeRLHF-QA (`is_safe`) | SFT, prompt-masked (`train_causal_lm_sft`) | 40-60 | 4-16 | 5e-4-2e-3 |

`*_inference` samples 40-80 prompts, 1-3 passes, 16-24 new tokens per
completion. `*_finetune` and `*_training` share the exact same SFT recipe
(`workloads/common.py`'s `train_causal_lm_sft`): tokenize prompt and
response separately, concatenate, and set the label to `-100` (PyTorch/HF's
"ignore this position" value) at every prompt-token position, so gradients
only come from predicting the response. The only mechanical difference
between the two is pretrained vs. randomly-initialized weights (and the
correspondingly higher step count / learning rate `*_training` needs to
make progress from nothing) -- both otherwise run identical code over the
identical dataset/filter their `ordinary`/`adversarial` counterpart uses.

## Running the experiment

```bash
# Sanity-check a single episode first (fast, ~2-4 min):
python instrumentation/episode_runner.py --scenario ordinary_inference --episode-id 000

# Collect the full dataset: 30 episodes, 3 ordinary + 3 adversarial families,
# 5 replicates each, execution order + GPU assignment randomized by --seed:
python experiments/collect_episodes.py --episodes 30 --seed 0

# Raw -> processed (text features + engineered NVML features):
python experiments/preprocess.py

# Train + cross-validate all three classifiers:
python experiments/train_text_classifier.py
python experiments/train_nvml_classifier.py
python experiments/train_combined_classifier.py   # text + NVML fused into one Random Forest

# Optional: zero-shot LLM API baseline (requires ANTHROPIC_API_KEY):
python experiments/api_baseline.py

# Compare everything:
python experiments/evaluate.py

# 3 PNG charts (episode distribution, overall metrics, per-family accuracy)
# written to results/ -- plain matplotlib, easy to copy into a notebook:
python experiments/plot_results.py
```

Each episode takes roughly 2-4 minutes, so 30 episodes is on the order of
1-2 hours end to end (sequential; this repo runs one GPU at a time by
default, see `--num-gpus` in `collect_episodes.py` if you have more than
one).

### Collecting more episodes without overwriting existing ones

`collect_episodes.py` numbers episodes starting from `--episode-id-start`,
which **defaults to auto-detect**: 1 + the highest episode-id already
present under `--data-dir/episodes`. So running it again with the same
command just adds a new batch after the existing one -- it will not reuse
(and therefore cannot overwrite) any episode directory from a previous
run:

```bash
# After an initial `--episodes 30` run (ids 001-030), this adds 20 MORE
# episodes as ids 031-050, safe to run concurrently with preprocess.py /
# train_*.py / evaluate.py on the existing 30 (disjoint episode-id ranges,
# no shared files):
python experiments/collect_episodes.py --episodes 20 --seed 1
```

Use a **different `--seed`** for each batch -- reusing the same seed as a
previous run doesn't collide episode IDs (that's handled automatically),
but it does mean the new batch's per-episode randomness (model/topic/
hyperparameter draws) repeats the same sequence, which is redundant rather
than genuinely new data. `data/raw/collection_manifest.json` accumulates
across runs (each run's plan is appended, not overwritten), so it stays a
complete record of every batch. Pass `--episode-id-start` explicitly if
you want to control numbering yourself instead of relying on auto-detect.

## Episode metadata schema

Every episode directory `data/raw/episodes/<episode_id>/` contains:

```
metadata.json      # see below
shell.log            # "<unix_timestamp>\t<command>" per line, this episode's commands only --
                       # RAW AUDIT LOG ONLY, not used to build text_features.csv (see preprocess.py)
processes.csv         # 1 Hz sample of this episode's process tree, incl. per-process disk I/O
nvml.csv               # 1 Hz sample of GPU telemetry
network.csv             # 1 Hz sample of network telemetry
stdout.log              # workload stdout/stderr, for debugging failures (not part of the core schema)
```

`metadata.json`:

```jsonc
{
  "episode_id": "001",
  "scenario_family": "ordinary_finetune",   // which of the 6 scenario scripts ran
  "label": "ordinary",                       // "ordinary" | "adversarial" -- kept separate from scenario_family
                                              // so evaluate.py can do grouped-by-family analysis
  "status": "success",                       // "success" | "failed" -- preprocess.py drops non-"success" episodes
  "error_message": "",
  "return_code": 0,
  "start_time": "2026-08-19T10:00:00+00:00",
  "end_time": "2026-08-19T10:02:41+00:00",
  "duration_seconds": 161.4,
  "invocation_style": "direct",              // "direct" | "module" | "shell_wrapper"
  "workdir_style": "flat",                   // "flat" | "nested" | "tmp"
  "outdir": "data/raw/episodes/001/work",
  "seed": 0,
  "gpu_index": 0,
  "command": ["python", "workloads/ordinary_finetune.py", "..."],
  "params": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "batch_size": 4, "steps": 30},
  "nvml_sample_count": 161,
  "nvml_error_count": 0,
  "process_sample_count": 161,
  "process_error_count": 0,
  "network_sample_count": 161,
  "network_error_count": 0,
  "gpu_name": "NVIDIA H100 80GB HBM3",
  "gpu_uuid": "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "driver_version": "550.90.07",
  "cuda_version": "12.4",
  "torch_version": "2.4.0",
  "hostname": "runpod-...",
  "python_version": "3.11.8",
  "cpu_count": 32,
  "system_ram_gb": 251.5,
  "gpu_memory_total_gb": 79.1
}
```

## NVML logging (`instrumentation/nvml_logger.py`)

Samples at 1 Hz: `timestamp, gpu_index, gpu_uuid, gpu_utilization,
memory_utilization, memory_used_mb, power_w, temperature_c, sm_clock_mhz,
memory_clock_mhz, pcie_tx_mb, pcie_rx_mb, error`. GPU identity is resolved
via UUID (not just index) at episode start, and stays in `metadata.json`.
Every per-field NVML call is wrapped individually -- a transient failure on
one metric (e.g. an unsupported field on a given GPU/driver combo) is
recorded in the row's `error` column and sampling continues; it is never
silently dropped.

## Process logging (`instrumentation/process_logger.py`)

Samples at 1 Hz: `timestamp, pid, process_name, command_line, cpu_percent,
memory_mb, read_bytes, write_bytes` (the last two are cumulative per-process
disk I/O counters via `psutil.Process.io_counters()`). Scoped strictly to
`psutil.Process(root_pid)` (the episode's subprocess) and its live
descendants -- it never scans the full system process table, so it can't
pick up unrelated users' or sessions' processes.

## Network logging (`instrumentation/network_logger.py`)

Samples at 1 Hz: `timestamp, bytes_sent_delta, bytes_recv_delta,
remote_addresses, error`. `remote_addresses` (the set of "ip:port" pairs
with an ESTABLISHED connection) is scoped to this episode's process tree
via each connection's `pid`; the byte-count deltas are **system-wide**
(`psutil` has no per-process network counter) -- a reasonable proxy on a
GPU pod running one job at a time, not a general per-process measurement.
This is stated explicitly in the module docstring rather than silently
assumed.

## Shell logging

`shell.log` only ever receives commands routed through one of two explicit
choke points: `episode_runner.py` logging the top-level launch command, and
`workloads/common.py`'s `run_shell()` helper, which every workload script
uses for any subprocess call it makes. There is no broader shell-history
capture; if a workload doesn't call `run_shell()`, the command doesn't get
logged. **This log is raw audit data only** -- `experiments/preprocess.py`
does NOT read it into `text_features.csv` (see "Text features" below for
why).

## Text features (`experiments/preprocess.py`)

`text_features.csv` is built ONLY from job/container metadata (GPU
name/index, CPU count, system RAM, GPU memory capacity, runtime versions,
duration -- from `metadata.json`), network metadata (bytes sent/received,
count of distinct remote addresses -- from `network.csv`), and storage I/O
metadata (total bytes read/written across the process tree -- from
`processes.csv`). It contains no shell commands, process names, or
command lines, and it excludes `metadata.json`'s `params` field entirely
(some of its keys, like `is_safe_filter`, directly encode the label).
This wasn't the original design: an earlier version included shell/process
text, and the workload script's filename in the launch command (e.g.
`workloads/adversarial_inference.py`) was a direct label leak regardless
of invocation style. See `preprocess.py`'s module docstring for the full
account.

## Data integrity

- `data/raw/` is write-once per episode; `experiments/preprocess.py` only
  ever writes to `data/processed/`, never back into `data/raw/`.
- Every row in every processed file carries `episode_id`.
- Episodes with `metadata.json["status"] != "success"` are excluded by
  `preprocess.py` (and reported, not silently dropped).
- `experiments/collect_episodes.py` writes `data/raw/collection_manifest.json`
  with the full planned scenario/param/GPU assignment *before* running
  anything, and updates each episode's `run_status` as it completes, so a
  partial/interrupted collection run is still fully auditable.

## A note on `.env`

`experiments/api_baseline.py` reads `ANTHROPIC_API_KEY` from the environment
or a local `.env` file. `.env` is gitignored -- never commit API keys.
