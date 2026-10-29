#!/usr/bin/env bash
set -uo pipefail

REPO=/home/shrutiw/projects/openvla-oft
PREP="$REPO/EXPERIMENT1_BASELINE_SPATIAL_PREP_20261026_142053"
LAUNCHER="$PREP/run_baseline_spatial.sh"
WATCH_DIR="$PREP/gpu_watch"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-5400}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
CONFIRM_SECONDS="${CONFIRM_SECONDS:-60}"
PID_FILE="$PREP/gpu_watch.pid"
STATUS_FILE="$PREP/gpu_watch_status.txt"
SENTINEL="$PREP/gpu_watch_smoke_completed"
LOCK_FILE="$PREP/gpu_watch.lock"

mkdir -p "$WATCH_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$WATCH_DIR/gpu_watch_${STAMP}.log"
exec >>"$LOG" 2>&1

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date -Is)] another gpu watch is already running; exiting"
  exit 0
fi

echo "$$" > "$PID_FILE"
cleanup() {
  rm -f "$PID_FILE"
}
trap cleanup EXIT

log() {
  echo "[$(date -Is)] $*"
}

write_status() {
  local state="$1"
  local free_mib="${2:-unknown}"
  local used_mib="${3:-unknown}"
  local util_gpu="${4:-unknown}"
  {
    echo "updated_at=$(date -Is)"
    echo "state=$state"
    echo "free_mib=$free_mib"
    echo "used_mib=$used_mib"
    echo "util_gpu=$util_gpu"
    echo "min_free_mib=$MIN_FREE_MIB"
    echo "interval_seconds=$INTERVAL_SECONDS"
    echo "pid=$$"
    echo "log=$LOG"
  } > "$STATUS_FILE"
}

if [ -f "$SENTINEL" ]; then
  log "smoke completion sentinel already exists at $SENTINEL; exiting"
  write_status "completed-earlier"
  exit 0
fi

log "monitor started pid=$$ interval=${INTERVAL_SECONDS}s threshold=${MIN_FREE_MIB}MiB confirm=${CONFIRM_SECONDS}s"
log "launcher=$LAUNCHER"
write_status "started"

while true; do
  if ! stats=$(nvidia-smi --query-gpu=memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1); then
    log "nvidia-smi query failed; will retry after ${INTERVAL_SECONDS}s"
    write_status "nvidia-smi-failed"
    sleep "$INTERVAL_SECONDS"
    continue
  fi

  IFS=',' read -r free_raw used_raw util_raw <<< "$stats"
  free_mib=$(echo "$free_raw" | tr -d ' ')
  used_mib=$(echo "$used_raw" | tr -d ' ')
  util_gpu=$(echo "$util_raw" | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)

  log "GPU free=${free_mib} MiB used=${used_mib} MiB util=${util_gpu} apps=${apps:-none}"
  write_status "waiting" "$free_mib" "$used_mib" "$util_gpu"

  if [ "$free_mib" -ge "$MIN_FREE_MIB" ]; then
    log "threshold met; confirming after ${CONFIRM_SECONDS}s"
    sleep "$CONFIRM_SECONDS"
    if ! confirm_stats=$(nvidia-smi --query-gpu=memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1); then
      log "confirmation query failed; returning to wait loop"
      write_status "confirmation-failed"
      sleep "$INTERVAL_SECONDS"
      continue
    fi
    IFS=',' read -r confirm_free_raw confirm_used_raw confirm_util_raw <<< "$confirm_stats"
    confirm_free=$(echo "$confirm_free_raw" | tr -d ' ')
    confirm_used=$(echo "$confirm_used_raw" | tr -d ' ')
    confirm_util=$(echo "$confirm_util_raw" | tr -d ' ')
    log "confirmation free=${confirm_free} MiB used=${confirm_used} MiB util=${confirm_util}"
    if [ "$confirm_free" -ge "$MIN_FREE_MIB" ]; then
      RUN_NOTE="baseline_spatial_smoke_auto_$(date +%Y%m%d_%H%M%S)"
      log "launching smoke test run_note=$RUN_NOTE"
      write_status "running" "$confirm_free" "$confirm_used" "$confirm_util"
      set +e
      MIN_FREE_MIB="$MIN_FREE_MIB" "$LAUNCHER" 1 "$RUN_NOTE"
      rc=$?
      set -e
      log "smoke test exited rc=$rc"
      if [ "$rc" -eq 2 ]; then
        log "guard refused because free memory changed; returning to wait loop"
        write_status "guard-refused" "$confirm_free" "$confirm_used" "$confirm_util"
      else
        if [ "$rc" -eq 0 ]; then
          touch "$SENTINEL"
          log "smoke test completed successfully; sentinel=$SENTINEL"
          write_status "completed" "$confirm_free" "$confirm_used" "$confirm_util"
        else
          log "smoke test failed rc=$rc; stopping monitor for investigation"
          write_status "failed" "$confirm_free" "$confirm_used" "$confirm_util"
        fi
        exit "$rc"
      fi
    else
      log "confirmation dropped below threshold; returning to wait loop"
    fi
  fi

  log "sleeping ${INTERVAL_SECONDS}s"
  sleep "$INTERVAL_SECONDS"
done
