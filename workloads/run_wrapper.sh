#!/usr/bin/env bash
# Generic shell wrapper used by episode_runner's "shell_wrapper" invocation
# style, i.e. episodes launched as:
#   bash workloads/run_wrapper.sh <scenario> [workload args...]
# instead of a bare `python workloads/<scenario>.py`. This exists purely to
# give the dataset invocation-style variation (some real workloads are
# started via a shell script rather than a direct interpreter call) -- it
# does nothing except exec the same script direct-invocation would run.
set -euo pipefail

SCENARIO="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "${REPO_ROOT}/workloads/${SCENARIO}.py" "$@"
