#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG =========
EXPERIMENT_ROOT="/persist/achentouf/NT/Math4AI/experiment_runs"
DUMP_ROOT="/persist/achentouf/NT/Math4AI/final_eval_runs"

ENV_NAME="arithmetic"
OP="gcd"
BASE=10
METADATA="../Math4AI/data/oct29_gcd/gcd/details.csv"   # (double gcd/) as you said
SPLITS="../Math4AI/data/oct29_gcd/gcd/gcd.test ../Math4AI/data/oct29_gcd/gcd/gcd.robust"

PYTHON_BIN="python"
TRAIN_SCRIPT="train.py"

NUM_GPUS=8
GPUS=(0 1 2 3 4 5 6 7)

# Speed knobs (tweak if your script supports them)
BATCH_TEST=4096
BATCH_ROBUST=2048

# Dry run just prints commands (no mkdir, no launches)
DRY_RUN=false

# Optional: faster log flushing
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ========= COLLECT CHECKPOINTS =========
mapfile -t CKPTS < <(find "$EXPERIMENT_ROOT" -type f -name "checkpoint.pth" | sort)

gpu_idx=0

launch_job () {
  local gpu_id="$1" ckpt="$2" split="$3" dump_dir="$4" eval_size="$5" batch_eval="$6"
  # inside launch_job()
  local cmd=( "$PYTHON_BIN" "$TRAIN_SCRIPT"
    --eval_only True
    --reload_model "$ckpt"
    --env_name "$ENV_NAME"
    --operation "$OP"
    --base "$BASE"
    --eval_data "$split"
    --metadata_path "$METADATA"
    --eval_size "$eval_size"
    --batch_size_eval "$batch_eval"
    --dump_path "$DUMP_ROOT"
    --exp_name "${exp_dir}_${hash_dir}_$(basename "$split")"
  )

  echo "CUDA_VISIBLE_DEVICES=$gpu_id ${cmd[*]}"
  if [ "$DRY_RUN" = false ]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" &
  fi


  echo "GPU $gpu_id :: $(basename "$split") -> $dump_dir (size=$eval_size, batch=$batch_eval)"
  if [ "$DRY_RUN" = true ]; then
    echo "CUDA_VISIBLE_DEVICES=$gpu_id $cmd"
  else
    mkdir -p "$dump_dir"
    CUDA_VISIBLE_DEVICES="$gpu_id" bash -lc "eval $cmd" &
  fi
}

for ckpt in "${CKPTS[@]}"; do
  hash_dir="$(basename "$(dirname "$ckpt")")"
  exp_dir="$(basename "$(dirname "$(dirname "$ckpt")")")"

  echo "==> Checkpoint: ${exp_dir}/${hash_dir}"
  echo "    $ckpt"

  for split in $SPLITS; do
    split_name="$(basename "$split")"
    dump_dir="${DUMP_ROOT}/${exp_dir}_${hash_dir}/${split_name}"

    case "$split_name" in
      gcd.test)   eval_size=10000;  batch="$BATCH_TEST"   ;;
      gcd.robust) eval_size=900000; batch="$BATCH_ROBUST" ;;
      *)          eval_size=900000; batch="$BATCH_ROBUST" ;;
    esac

    gpu="${GPUS[$(( gpu_idx % NUM_GPUS ))]}"
    gpu_idx=$((gpu_idx + 1))

    launch_job "$gpu" "$ckpt" "$split" "$dump_dir" "$eval_size" "$batch"

    # Cap concurrent jobs to NUM_GPUS (one per GPU); wait for any to finish before launching next
    if [ "$DRY_RUN" = false ]; then
      while [ "$(jobs -r | wc -l)" -ge "$NUM_GPUS" ]; do
        wait -n || true
      done
    fi
  done
done

# Wait for remaining jobs
if [ "$DRY_RUN" = false ]; then
  wait
fi

echo "Queue complete."
