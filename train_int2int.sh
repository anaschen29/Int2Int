#!/usr/bin/env bash
set -euo pipefail

# ---- GPU selection ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---- Paths ----
DATA_ROOT="${1:-data}"          # ROOT dir with subfolders: data/{gcd,modexp,ec_rank}
DUMP_ROOT="${2:-runs/train}"    # where the trainer writes experiments

# ---- Common training knobs ----
MAX_EPOCH=50
EPOCH_SIZE=300000
BATCH_SIZE=128
LR="adam,lr=0.0002"
SEEDS=(1337 2025)
LAYERS=(4)
DIMS=(256)
HEADS=(8)

train_one () {
  local OP="$1"                  # operation: gcd | modexp | ec_rank
  local DATA_ROOT_DIR="$2"       # root data dir: e.g., data
  local EXP_NAME="$3"

  local TASK_DIR="${DATA_ROOT_DIR}/${OP}"
  local TRAIN_FILE="${TASK_DIR}/train.src"   # << required as a FILE (EnvDataset asserts isfile)
  local VALID_FILE="${TASK_DIR}/valid.src"   # (not strictly required by this assert, but we validate early)
  local TEST_FILE="${TASK_DIR}/test.src"

  # Sanity checks so we fail fast with a clear message
  [[ -f "${TRAIN_FILE}" ]] || { echo "[ERR] Missing ${TRAIN_FILE}"; exit 1; }
  [[ -f "${TASK_DIR}/train.tgt" ]] || { echo "[ERR] Missing ${TASK_DIR}/train.tgt"; exit 1; }
  [[ -f "${VALID_FILE}" ]] || { echo "[WARN] Missing ${VALID_FILE} (validation may fall back)"; }
  [[ -f "${TASK_DIR}/valid.tgt" ]] || { echo "[WARN] Missing ${TASK_DIR}/valid.tgt"; }
  [[ -f "${TEST_FILE}" ]] || { echo "[WARN] Missing ${TEST_FILE} (test may fall back)"; }
  [[ -f "${TASK_DIR}/test.tgt" ]] || { echo "[WARN] Missing ${TASK_DIR}/test.tgt"; }

  python3 ../Int2Int/train.py \
    --dump_path "${DUMP_ROOT}" \
    --exp_name "${EXP_NAME}" \
    --epoch_size "${EPOCH_SIZE}" \
    --max_epoch "${MAX_EPOCH}" \
    --optimizer "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --env_name arithmetic \
    --tasks arithmetic \
    --operation "${OP}" \
    --train_data "${TRAIN_FILE}" \
    --eval_data "${TASK_DIR}" \
    --eval_size 10000 \
    --validation_metrics valid_acc \
    --report_loss_every 200 \
    --enc_emb_dim "${EMB_DIM}" \
    --dec_emb_dim "${EMB_DIM}" \
    --n_enc_layers "${ENC_LAYERS}" \
    --n_dec_layers "${DEC_LAYERS}" \
    --n_enc_heads "${HEADS_CNT}" \
    --n_dec_heads "${HEADS_CNT}" \
    --enc_has_pos_emb true \
    --dec_has_pos_emb true \
    --architecture encoder_decoder
}

for SEED in "${SEEDS[@]}"; do
  for ENC_LAYERS in "${LAYERS[@]}"; do
    DEC_LAYERS=$ENC_LAYERS
    for EMB_DIM in "${DIMS[@]}"; do
      for HEADS_CNT in "${HEADS[@]}"; do

        if [[ -d "${DATA_ROOT}/gcd" ]]; then
          EXP="gcd_e${ENC_LAYERS}_d${DEC_LAYERS}_h${HEADS_CNT}_d${EMB_DIM}_s${SEED}"
          train_one "gcd" "${DATA_ROOT}" "${EXP}"
        fi

        if [[ -d "${DATA_ROOT}/modexp" ]]; then
          EXP="modexp_e${ENC_LAYERS}_d${DEC_LAYERS}_h${HEADS_CNT}_d${EMB_DIM}_s${SEED}"
          train_one "modexp" "${DATA_ROOT}" "${EXP}"
        fi

        if [[ -d "${DATA_ROOT}/ec_rank" ]]; then
          EXP="ec_rank_e${ENC_LAYERS}_d${DEC_LAYERS}_h${HEADS_CNT}_d${EMB_DIM}_s${SEED}"
          train_one "ec_rank" "${DATA_ROOT}" "${EXP}"
        fi

      done
    done
  done
done
