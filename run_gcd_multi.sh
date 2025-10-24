#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG (edit me)
# =========================
NGPU=8
ROOT="../Math4AI"
DATA="${ROOT}/data/gcd/gcd"             # gcd.train / gcd.train.mixed_p* / gcd.valid / gcd.test / gcd.robust / details.csv
RUNS="${ROOT}/experiment_runs"

# Environment / data flags
ENV_FLAGS="--env_name arithmetic --operation gcd --base 10"
EVAL_DATA="--eval_data ${DATA}/gcd.valid,${DATA}/gcd.test,${DATA}/gcd.robust"
META="--metadata_path ${DATA}/details.csv"

# Runtime / logging
SAVE_PERIODIC=0                         # 0 = only best+checkpoint, 1 = snapshot every epoch
REPORT_EVERY=200
NUM_WORKERS=0                           # trainer asserts {0,1}; use 0 for stability
EVAL_SIZE=100000                        # during training; do -1 later in eval-only

# Training length — aim to use your wall-clock
EPOCH_SIZE=200000                       # ~longer epoch than 120k to better use time
MAX_EPOCH=10

# Grid (reasonable, time-bounded)
SEEDS=(1982 1986 2010 2014)                       # 2 seeds keep variance visible without doubling wall time too much
MIXES=(orig p25 p50 p75)                # train_data variants
DEPTHS=("4x4" "6x6")                    # encoder/decoder depth pairs
LRS=(0.0001 0.0005 0.001)                     # two learning rates
DROPS=(0.0 0.1 0.2)                         # model dropout

# Batch sizes (keep memory safe). If you want to try 96/128 on 24GB GPUs, add here.
BZS=(64 128)

# Core heads/emb width (baseline width is robust; feel free to add 384 later for a width ablation)
ENC_EMB=256
DEC_EMB=256
N_HEADS=8

# =========================
# HELPERS
# =========================
train_file_for_mix () {
  local m="$1"
  case "$m" in
    orig) echo "${DATA}/gcd.train" ;;
    p25)  echo "${DATA}/gcd.train.mixed_p25" ;;
    p50)  echo "${DATA}/gcd.train.mixed_p50" ;;
    p75)  echo "${DATA}/gcd.train.mixed_p75" ;;
    *)    echo "Unknown mix: $m" >&2; exit 2 ;;
  esac
}

layers_for_depth () {
  local d="$1"
  case "$d" in
    4x4) echo "4 4" ;;
    6x6) echo "6 6" ;;
    *)   echo "Unknown depth: $d" >&2; exit 2 ;;
  esac
}

# queue machinery
jid=0
launch () {
  local gpu_id="$1"
  local train_file="$2"
  local exp_tag="$3"
  local seed="$4"
  local enc_layers="$5"
  local dec_layers="$6"
  local lr="$7"
  local drop="$8"
  local bs="$9"

  local exp_name="${exp_tag}_e${enc_layers}d${dec_layers}_h${N_HEADS}_emb${ENC_EMB}_lr${lr//./}d${drop/./}_bs${bs}_s${seed}"
  local log_dir="${RUNS}/${exp_name}"
  mkdir -p "${log_dir}"

  echo ">>> [GPU ${gpu_id}] ${exp_name}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  python train.py \
    --exp_name "${exp_name}" \
    --dump_path "${RUNS}" \
    --save_periodic ${SAVE_PERIODIC} \
    ${ENV_FLAGS} \
    --enc_emb_dim ${ENC_EMB} --dec_emb_dim ${DEC_EMB} \
    --n_enc_heads ${N_HEADS} --n_dec_heads ${N_HEADS} \
    --n_enc_layers ${enc_layers} --n_dec_layers ${dec_layers} \
    --dropout ${drop} \
    --optimizer "adam,lr=${lr}" --clip_grad_norm 5 \
    --batch_size ${bs} --batch_size_eval 256 \
    --report_loss_every ${REPORT_EVERY} \
    --num_workers ${NUM_WORKERS} \
    --epoch_size ${EPOCH_SIZE} --max_epoch ${MAX_EPOCH} \
    --train_data "${train_file}" \
    ${EVAL_DATA} \
    ${META} \
    --env_base_seed "${seed}" \
    > "${log_dir}/stdout.log" 2>&1 &
}

# =========================
# QUEUE: keep 8 GPUs full
# =========================
for seed in "${SEEDS[@]}"; do
  for mix in "${MIXES[@]}"; do
    train_file="$(train_file_for_mix "${mix}")"
    for depth in "${DEPTHS[@]}"; do
      read -r ENC_LAY DEC_LAY <<<"$(layers_for_depth "${depth}")"
      for lr in "${LRS[@]}"; do
        for drop in "${DROPS[@]}"; do
          for bs in "${BZS[@]}"; do
            exp_tag="gcd_${mix}"
            gpu_id=$(( jid % NGPU ))
            launch "${gpu_id}" "${train_file}" "${exp_tag}" "${seed}" "${ENC_LAY}" "${DEC_LAY}" "${lr}" "${drop}" "${bs}"
            jid=$((jid + 1))
            if (( jid % NGPU == 0 )); then
              echo ">>> Waiting for a wave of ${NGPU} jobs to finish ..."
              wait
            fi
          done
        done
      done
    done
  done
done

wait
echo "All runs completed. See ${RUNS}/<exp_name>/stdout.log + checkpoints."
