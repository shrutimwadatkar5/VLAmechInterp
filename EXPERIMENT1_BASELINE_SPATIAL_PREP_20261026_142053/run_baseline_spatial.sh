#!/usr/bin/env bash
set -euo pipefail

REPO=/home/shrutiw/projects/openvla-oft
VENV="$REPO/.venv_libero"
CKPT="$REPO/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
NUM_TRIALS="${1:-1}"
RUN_NOTE="${2:-baseline_spatial}"
LOG_DIR="$REPO/EXPERIMENT1_BASELINE_SPATIAL_PREP_20261026_142053/baseline_logs"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "GPU free memory: ${free_mib} MiB (minimum required: ${MIN_FREE_MIB} MiB)"
if [ "${free_mib}" -lt "${MIN_FREE_MIB}" ]; then
  echo "REFUSING TO START: not enough free GPU memory. Wait for other jobs to finish." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH="$HOME/.libero"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
exec "$VENV/bin/python" experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CKPT" \
  --task_suite_name libero_spatial \
  --num_trials_per_task "$NUM_TRIALS" \
  --seed 7 \
  --center_crop True \
  --local_log_dir "$LOG_DIR" \
  --run_id_note "$RUN_NOTE"
