#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
8 MUSA ER-only N=300 PFN runner with topology embedding + standard attention.

Usage:
  bash run_8musa_train.sh
  bash run_8musa_train.sh --help

Defaults:
  Visible MUSA devices: 0,1,2,3,4,5,6,7
  Episodes: 25600 (80% train / 20% validation)
  Epochs: 125
  Per-device batch size: 16
  Global batch size: 16 x 8 = 128
  Graph family: ER only
  ER probability: 0.02
  Units: 300 (all observed and all queried)
  Training queries: majority-arm exposure queries
  Transformer: topology/network embedding + standard global self-attention
  Evaluation: PFN only (no comparison baselines)
  Test episodes: 64, seed 2026

Override the physical device list before launch, for example:
  MUSA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 bash run_8musa_train.sh
USAGE
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
EXPECTED_SCHEMA="majority_arm_unified_ite_ate_v10_er_only_topology_embedding"
DATA_CSV="data/er_n300_p002_newdgp_v10_ddp_25600_seed12345.csv"
TRAIN_CSV="data/er_n300_p002_newdgp_v10_ddp_25600_seed12345_train.csv"
VALIDATION_CSV="data/er_n300_p002_newdgp_v10_ddp_25600_seed12345_validation.csv"
CHECKPOINT_BASE="runs/er_newDGP_topology_embedding_ddp.pt"
BEST_POINT_CHECKPOINT="runs/er_newDGP_topology_embedding_ddp_best_point.pt"
RESULT_DIR="results/final_er_newDGP_topology_embedding_seed2026_n64"

csv_is_current() {
  local path="$1"
  python - "$path" "$EXPECTED_SCHEMA" <<'PY_SCHEMA'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_schema = sys.argv[2]
try:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader)
    ok = (
        first.get("schema_version") == expected_schema
        and int(first.get("n_units", -1)) == 300
    )
except (OSError, StopIteration, TypeError, ValueError):
    ok = False
raise SystemExit(0 if ok else 1)
PY_SCHEMA
}

if [[ ${1:-} == "--worker" ]]; then
  if csv_is_current "$DATA_CSV" && csv_is_current "$TRAIN_CSV" && csv_is_current "$VALIDATION_CSV"; then
    echo "===== Reusing completed N=300 ER-only topology-embedding CSV files ====="
  else
    echo "===== Preparing N=300 ER-only topology-embedding training CSV before torchrun ====="
    echo "This CPU data-preparation stage can take a long time; the 8 MUSA ranks have not started yet."
    python -u train_local_network_interference.py \
        --prepare-data-only \
        --device cpu \
        --n-units 300 \
        --data-csv "$DATA_CSV" \
        --csv-datasets 25600 \
        --csv-generation-batch-size 64 \
        --data-seed 12345 \
        --validation-fraction 0.2 \
        --split-seed 0 \
        --overwrite-data-csv \
        --er-edge-probability 0.02
  fi

  echo "===== CSV preparation completed; starting 8-rank torchrun ====="
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    train_local_network_interference.py \
      --device musa \
      --n-units 300 \
      --data-csv "$DATA_CSV" \
      --csv-datasets 25600 \
      --csv-generation-batch-size 64 \
      --data-seed 12345 \
      --validation-fraction 0.2 \
      --split-seed 0 \
      --er-edge-probability 0.02 \
      --epochs 125 \
      --batch-size 16 \
      --learning-rate 1e-3 \
      --weight-decay 1e-5 \
      --scheduler-factor 0.5 \
      --scheduler-patience 5 \
      --training-seed 0 \
      --checkpoint-path "$CHECKPOINT_BASE" \
      --skip-test-after-training

  rm -rf "$RESULT_DIR"

  env MUSA_VISIBLE_DEVICES="${VISIBLE_DEVICES%%,*}" \
    python -u train_local_network_interference.py \
      --test-only \
      --device musa:0 \
      --checkpoint-path "$BEST_POINT_CHECKPOINT" \
      --test-seed 2026 \
      --test-datasets 64 \
      --test-output-dir "$RESULT_DIR"
  exit 0
fi

mkdir -p data runs logs results
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_8musa_${RUN_ID}.log"

nohup env \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=utf-8 \
  OMP_NUM_THREADS=1 \
  MUSA_VISIBLE_DEVICES="$VISIBLE_DEVICES" \
  bash "$0" --worker \
  > "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > .latest_train_pid
printf '%s\n' "$LOG_FILE" > .latest_train_log
printf '%s\n' "$RESULT_DIR" > .latest_result_dir

cat <<STATUS
8 MUSA training started.
PID: $PID
Log: $LOG_FILE
Visible devices: $VISIBLE_DEVICES
Per-device batch: 16
Global batch: 128

Watch:
  tail -f "$LOG_FILE"

After completion:
  python -m json.tool "$RESULT_DIR/algorithm2_summary.json"
  head -n 20 "$RESULT_DIR/algorithm2_dataset_metrics.csv"
STATUS
