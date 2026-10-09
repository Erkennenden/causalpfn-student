#!/usr/bin/env bash
# Run on server host after unzipping. Training continues after SSH disconnects.
set -euo pipefail
CONTAINER="${PFN_CONTAINER:-musacode}"
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == status ]]; then
  docker exec "$CONTAINER" bash -lc '
    set -e; cd "$(cat /workspace/latest_causalfm_project.txt)"
    echo "项目：$PWD"
    if [[ -f .latest_causalfm_job ]]; then
      j=$(cat .latest_causalfm_job); echo "结果：$j"
      for f in stage.txt exit_code.txt elapsed_seconds.txt; do [[ ! -f "$j/$f" ]] || { echo "$f:"; cat "$j/$f"; }; done
    fi
    if [[ -f .host_exit_code ]]; then echo "后台退出码：$(cat .host_exit_code)"; else ps -p "$(cat .host_pid)" -o pid,etime,args || true; fi
    tail -n 40 host.log'
  exit
fi
[[ $# == 0 ]] || { echo 'Usage: bash launch_causalfm_docker.sh [status]' >&2; exit 2; }
[[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" == true ]]
PROJECT="$(docker exec "$CONTAINER" mktemp -d /workspace/causalfm_cepo_XXXXXXXX)"
docker cp "$SOURCE/." "$CONTAINER:$PROJECT/"
ENV_ARGS=()
for name in DEVICES NUM_LAYERS D_MODEL NUM_HEADS TRAIN_EPISODES VALIDATION_EPISODES EPOCHS BATCH_SIZE DATA_SEED TRAIN_SEED TEST_SEED TEST_SEEDS TEST_TASKS BASELINE_EPOCHS TNET_EPOCHS HYPERSCI_ARM_SAMPLES PRIOR_DIAGNOSTIC_TASKS MUSA_VISIBLE_DEVICES; do
  if [[ -n "${!name:-}" ]]; then ENV_ARGS+=(-e "$name=${!name}"); fi
done
docker exec -i "${ENV_ARGS[@]}" "$CONTAINER" bash -s -- "$PROJECT" <<'INSIDE'
set -euo pipefail
cd "$1"
export DEVICE=musa OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PFN_CONTROL_TIMEOUT=3600 PFN_TRAIN_TIMEOUT=600 MCCL_DEBUG=WARN
nohup bash -c '
  set -euo pipefail
  trap '\''rc=$?; printf "%s\n" "$rc" > .host_exit_code'\'' EXIT
  bash run_causalfm.sh
' > host.log 2>&1 < /dev/null &
printf '%s\n' "$!" > .host_pid
printf '%s\n' "$PWD" > /workspace/latest_causalfm_project.txt
echo "已提交 nohup，PID：$(cat .host_pid)"
echo "目录：${PWD}；日志：${PWD}/host.log"
sleep 2
tail -n 20 host.log
if [[ -f .host_exit_code ]]; then
  rc=$(cat .host_exit_code)
  echo "后台任务已结束，退出码：$rc"
  exit "$rc"
fi
if ! kill -0 "$(cat .host_pid)" 2>/dev/null; then
  echo '后台进程已退出且没有退出码，请检查 host.log' >&2
  exit 1
fi
echo '后台进程仍在运行；请使用 status 查看训练进度。'
INSIDE
