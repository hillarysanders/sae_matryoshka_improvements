#!/usr/bin/env bash
set -euo pipefail

# Creates a tmux session with 8 windows (gpu0..gpu7) and runs one job per GPU.
# Each job is pinned via CUDA_VISIBLE_DEVICES.

SESSION="${SESSION:-sae8}"
OUT_DIR="${OUT_DIR:-runs}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

# ---- knobs chosen from your l0_sweep ----
LAM_L1_UNIFORM="4.8e-5"
LAM_P_ANNEAL="3.7e-5"
LAM_FREQ="1.15e-4"
LAM_COMBINED="1.05e-4"
TARGET_L0="40"
P_END="0.5"
P_START="1.0"

NUM_STEPS=250
BATCH_SIZE=16
SEQ_LEN=128
EVAL_BATCHES=5

TRAIN_FLAGS="--num_steps ${NUM_STEPS} --batch_size ${BATCH_SIZE} --seq_len ${SEQ_LEN} --eval_num_batches ${EVAL_BATCHES} --ckpt_every ${NUM_STEPS}"

# ---- Job spec format ----
# name | sparsity | seed | extra_args...
#
# IMPORTANT: keep run_name stable across seeds so plots aggregate.
# run_name is the first field here.
JOBS=(
  "ec2_l1_uniform|l1_uniform|0|--lambda_base ${LAM_L1_UNIFORM}"
  "ec2_p_anneal|p_annealing|0|--lambda_base ${LAM_P_ANNEAL} --p_start ${P_START} --p_end ${P_END}"
  "ec2_freq_l1|l1_freq_weighted|0|--lambda_base ${LAM_FREQ} --fw_warmup_steps 25"
  "ec2_combined|p_annealing_freq|0|--lambda_base ${LAM_COMBINED} --p_start ${P_START} --p_end ${P_END} --fw_warmup_steps 25"
  "ec2_batchtopk|batchtopk|0|--target_l0 ${TARGET_L0}"

  # extra capacity (use for repeats)
  "ec2_l1_uniform|l1_uniform|1|--lambda_base ${LAM_L1_UNIFORM}"
  "ec2_p_anneal|p_annealing|1|--lambda_base ${LAM_P_ANNEAL} --p_start ${P_START} --p_end ${P_END}"
  "ec2_freq_l1|l1_freq_weighted|1|--lambda_base ${LAM_FREQ} --fw_warmup_steps 25"
)

# Optional: pin device/dtype explicitly for EC2 GPU runs
# If you keep Config defaults, you can omit these.
DEVICE_FLAGS="--device cuda --dtype bf16"

# Create session if it doesn't exist
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n "gpu0"
fi

# Ensure we have 8 windows named gpu0..gpu7
for i in $(seq 0 7); do
  if [[ "${i}" -eq 0 ]]; then
    tmux rename-window -t "${SESSION}:0" "gpu0"
  else
    # create if missing
    if ! tmux list-windows -t "${SESSION}" | grep -q "gpu${i}"; then
      tmux new-window -t "${SESSION}" -n "gpu${i}"
    fi
  fi
done

# Launch jobs
for gpu in $(seq 0 7); do
  if [[ $gpu -ge ${#JOBS[@]} ]]; then
    continue
  fi
  job="${JOBS[$gpu]}"
  IFS="|" read -r RUN_NAME SPARSITY SEED EXTRA <<< "${job}"

  stamp="$(date +%Y%m%d_%H%M%S)"
  log_file="${LOG_DIR}/${RUN_NAME}_seed${SEED}_gpu${gpu}_${stamp}.log"

  cmd=$(cat <<EOF
cd "$(pwd)" && \
export CUDA_VISIBLE_DEVICES=${gpu} && \
export TOKENIZERS_PARALLELISM=false && \
echo "[start] gpu=${gpu} run_name=${RUN_NAME} sparsity=${SPARSITY} seed=${SEED} $(date)" && \
python3 train.py \
  --run_name "${RUN_NAME}" \
  --out_dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --sparsity "${SPARSITY}" \
  ${DEVICE_FLAGS} \
  ${TRAIN_FLAGS} \
  ${EXTRA} \
  2>&1 | tee "${log_file}"
EOF
)

# send to window gpuN
tmux send-keys -t "${SESSION}:gpu${gpu}" "${cmd}" C-m
done

echo "Launched ${#JOBS[@]} jobs in tmux session ${SESSION}"
echo "Attach with: tmux attach -t ${SESSION}"
