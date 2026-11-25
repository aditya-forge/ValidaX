#!/usr/bin/env bash
# Start the fraud risk operations console (Streamlit).
#
# Read-only: the console reads app/model.pkl, outputs/ and (optionally)
# data/processed/transactions_features.parquet. It never writes to any of them.
#
# Settings (environment variables, all optional):
#   PORT            port to bind            (default 8501)
#   ADDRESS         interface to bind       (default 0.0.0.0)
#   PYTHON          interpreter to use      (default: python)
#   OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS   native threads (default 4)
#   MALLOC_ARENA_MAX                        glibc malloc arenas (default 2, limits fragmentation)
#   MIN_MEMORY_MB   warn if the container memory limit is below this (default 3072)
set -euo pipefail
cd "$(dirname "$0")/.."   # repository root (this script lives in scripts/)

PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-0.0.0.0}"
PYTHON="${PYTHON:-python}"
MIN_MEMORY_MB="${MIN_MEMORY_MB:-3072}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

# Required artifacts: fail fast with a clear message instead of a broken page.
required=(
  app/app.py
  app/model.pkl
  outputs/scored_transactions.csv
  outputs/model_summary.json
  outputs/threshold_sweep.csv
  outputs/cost_sensitive_threshold.csv
  outputs/daily_capacity_simulation.csv
  outputs/feature_importance.csv
)
missing=0
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required file missing: $path (run src/model.py and src/scoring.py to create it)" >&2
    missing=1
  fi
done
if (( missing )); then
  exit 1
fi

if [[ ! -f data/processed/transactions_features.parquet ]]; then
  echo "WARNING: data/processed/transactions_features.parquet not found." >&2
  echo "         The Transaction investigation page will run without live SHAP explanations." >&2
fi

# Report the memory limit (cgroup v2, then v1) and warn if it is tight.
limit_bytes=""
if [[ -r /sys/fs/cgroup/memory.max ]]; then
  limit_bytes="$(cat /sys/fs/cgroup/memory.max)"
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
  limit_bytes="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
fi
if [[ "$limit_bytes" =~ ^[0-9]+$ ]] && (( limit_bytes < 1099511627776 )); then
  limit_mb=$(( limit_bytes / 1024 / 1024 ))
  echo "Memory limit: ${limit_mb} MB | threads: OMP=${OMP_NUM_THREADS} | port: ${PORT}"
  if (( limit_mb < MIN_MEMORY_MB )); then
    echo "WARNING: memory limit is below ${MIN_MEMORY_MB} MB. The Transaction investigation page" >&2
    echo "         loads the full test-window feature table (about 2 GB at peak)." >&2
  fi
else
  echo "Memory limit: not set | threads: OMP=${OMP_NUM_THREADS} | port: ${PORT}"
fi

exec "$PYTHON" -m streamlit run app/app.py \
  --server.port "$PORT" \
  --server.address "$ADDRESS" \
  --server.headless true \
  --server.fileWatcherType none \
  --server.runOnSave false \
  --browser.gatherUsageStats false
