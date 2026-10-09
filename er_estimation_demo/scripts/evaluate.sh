#!/usr/bin/env bash
# Evaluate an existing CEPO checkpoint; ATE and ITE share all test datasets.
set -euo pipefail
[[ $# -ge 2 ]] || { echo 'Usage: bash scripts/evaluate.sh CHECKPOINT NEW_OUTPUT_DIR [evaluate options]' >&2; exit 2; }
# Preserve caller-relative checkpoint/output paths even when run outside the project.
CHECKPOINT="$1"; OUTPUT="$2"; shift 2
[[ "$CHECKPOINT" = /* ]] || CHECKPOINT="$PWD/$CHECKPOINT"
[[ "$OUTPUT" = /* ]] || OUTPUT="$PWD/$OUTPUT"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
if [[ -n "${TEST_SEED:-}" && -n "${TEST_SEEDS:-}" ]]; then
  echo 'Set only one of TEST_SEED and TEST_SEEDS.' >&2; exit 2
fi
if [[ -n "${TEST_SEED:-}" ]]; then
  SEED_ARGS=(--test-seed "$TEST_SEED")
else
  read -r -a SEED_VALUES <<< "${TEST_SEEDS:-2025 2026 2027 2028 2029}"
  SEED_ARGS=(--test-seeds "${SEED_VALUES[@]}")
fi
"$PYTHON" -m pfn_interference evaluate --checkpoint "$CHECKPOINT" --output "$OUTPUT" \
  --device "${DEVICE:-musa}" --cache-dir "${CACHE_DIR:-$PWD/runs/task_cache}" \
  --test-tasks "${TEST_TASKS:-2}" "${SEED_ARGS[@]}" \
  --baseline-epochs "${BASELINE_EPOCHS:-500}" --tnet-epochs "${TNET_EPOCHS:-160}" \
  --hypersci-arm-samples "${HYPERSCI_ARM_SAMPLES:-4096}" "$@"
