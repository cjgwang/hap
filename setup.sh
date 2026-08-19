#!/usr/bin/env bash
# Cloud Classifiers experiment setup, meant to run on a RunPod GPU pod.
#
# Deliberately does NOT hardcode a CUDA version: it lets pip resolve the
# default CUDA-enabled PyTorch wheel for the detected platform (modern
# Linux torch wheels bundle their own CUDA runtime and rely on NVIDIA's
# driver forward-compatibility, so this works across driver/CUDA
# combinations without us guessing a cuXXX tag), then VERIFIES both PyTorch
# GPU access and NVML access afterward and fails loudly with diagnostics if
# either doesn't work -- rather than silently continuing on a broken GPU
# setup.
#
# Safe to re-run: venv creation and pip install are both idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== 1. NVIDIA driver / GPU check ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. This script expects to run on a RunPod GPU pod." >&2
    echo "If you are on a CPU-only box, the experiment cannot collect GPU telemetry." >&2
    exit 1
fi
nvidia-smi
echo

echo "=== 2. Python virtual environment ==="
if [ ! -d venv ]; then
    # --system-site-packages: RunPod's official PyTorch templates ship a
    # CUDA-matched torch preinstalled system-wide. Inheriting it means we
    # don't force a slow, potentially mismatched reinstall on those images;
    # on a bare image where nothing is preinstalled, this flag is a no-op
    # and pip below installs everything fresh into the venv as normal.
    python3 -m venv --system-site-packages venv
    echo "created ./venv"
else
    echo "./venv already exists, reusing it"
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip wheel
echo

echo "=== 3. Installing dependencies ==="
pip install -r requirements.txt
echo

echo "=== 4. Verifying PyTorch can see the GPU ==="
if ! python - <<'PYEOF'
import sys
import torch
print(f"torch version: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit(1)
print(f"device: {torch.cuda.get_device_name(0)}")
PYEOF
then
    echo "ERROR: PyTorch cannot see a CUDA GPU." >&2
    echo "The installed torch build may not match this machine's driver." >&2
    echo "Try installing a specific build from https://pytorch.org/get-started/locally/" >&2
    echo "matching the 'CUDA Version' reported by nvidia-smi above, e.g.:" >&2
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu126" >&2
    exit 1
fi
echo

echo "=== 5. Verifying NVML access (pynvml / nvidia-ml-py) ==="
if ! python - <<'PYEOF'
import pynvml
pynvml.nvmlInit()
count = pynvml.nvmlDeviceGetCount()
print(f"NVML sees {count} GPU(s)")
for i in range(count):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(h)
    uuid = pynvml.nvmlDeviceGetUUID(h)
    name = name.decode() if isinstance(name, bytes) else name
    uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
    print(f"  [{i}] {name}  uuid={uuid}")
pynvml.nvmlShutdown()
PYEOF
then
    echo "ERROR: NVML access failed. Common causes: no GPU visible to this container," >&2
    echo "or insufficient permissions. instrumentation/nvml_logger.py cannot run without this." >&2
    exit 1
fi
echo

echo "=== 6. Preparing data/results directories ==="
mkdir -p data/raw/episodes data/processed results
echo "ok"
echo

echo "=== Setup complete ==="
echo "Activate the environment in new shells with: source venv/bin/activate"
echo "Then try a single episode:"
echo "  python instrumentation/episode_runner.py --scenario ordinary_dev --episode-id 000"
