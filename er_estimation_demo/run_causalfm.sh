#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
DEVICE="${DEVICE:-musa}"; DEVICES="${DEVICES:-8}"
[[ "$DEVICES" =~ ^[1-9][0-9]*$ ]] || { echo 'DEVICES must be positive' >&2; exit 2; }
command -v flock >/dev/null || { echo 'Missing flock: run this script in the Linux Docker container.' >&2; exit 2; }
exec 9>"${PFN_LOCK_FILE:-/tmp/causalfm_er_training.lock}"
flock -n 9 || { echo 'Another experiment holds the training lock' >&2; exit 1; }
"$PYTHON" - "$DEVICE" "$DEVICES" <<'PY'
import sys,torch,numpy,scipy,networkx,sklearn,matplotlib,torch_geometric
from train_local_network_interference import resolve_device
from baselines.hypersci import make_model
assert torch_geometric.__version__=='2.6.1','Install torch-geometric==2.6.1'
make_model(2,hidden=4)
d=resolve_device(sys.argv[1])
if d.type=='musa':
    assert torch.musa.is_available() and torch.musa.device_count()>=int(sys.argv[2])
print('Preflight passed:',sys.version,torch.__version__,d,flush=True)
PY
mkdir -p runs
JOB="${OUTPUT_DIR:-$(mktemp -d "$PWD/runs/causalfm_cepo_XXXXXXXX")}"
mkdir -p "$JOB"
printf '%s\n' "$JOB" > .latest_causalfm_job
START=$(date +%s)
trap 'rc=$?; printf "%s\n" "$rc" > "$JOB/exit_code.txt"; printf "%s\n" "$(( $(date +%s)-START ))" > "$JOB/elapsed_seconds.txt"' EXIT
GRAPH="${FIXED_GRAPH_FILE:-$PWD/data/fixed_er_N1000_p0.02_seed12345.json}"
CACHE="${CACHE_DIR:-$PWD/runs/task_cache}"
echo diagnose > "$JOB/stage.txt"
"$PYTHON" -m causalfm_experiment diagnose --graph-file "$GRAPH" --cache-dir "$CACHE" --output "$JOB/prior" \
  --tasks "${PRIOR_DIAGNOSTIC_TASKS:-128}" --data-seed "${DATA_SEED:-12345}"
echo training > "$JOB/stage.txt"
CMD=("$PYTHON")
if (( DEVICES > 1 )); then CMD+=(-m torch.distributed.run --standalone --nproc_per_node "$DEVICES"); fi
"${CMD[@]}" -m causalfm_experiment train --graph-file "$GRAPH" --cache-dir "$CACHE" --output "$JOB/training" \
  --device "$DEVICE" --train-tasks "${TRAIN_EPISODES:-40960}" --validation-tasks "${VALIDATION_EPISODES:-5120}" \
  --epochs "${EPOCHS:-125}" --batch-size "${BATCH_SIZE:-16}" --data-seed "${DATA_SEED:-12345}" --train-seed "${TRAIN_SEED:-0}" \
  --num-layers "${NUM_LAYERS:-10}" --d-model "${D_MODEL:-128}" --num-heads "${NUM_HEADS:-4}"
echo testing > "$JOB/stage.txt"
PYTHON="$PYTHON" DEVICE="$DEVICE" CACHE_DIR="$CACHE" \
  bash evaluate_causalfm.sh "$JOB/training/model_best.pt" "$JOB/evaluation"
echo complete > "$JOB/stage.txt"
cat "$JOB/evaluation/all_results.txt"
echo "Completed: $JOB"
