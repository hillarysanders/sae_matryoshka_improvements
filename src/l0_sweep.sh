#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=false

# caffeinate -dimsu ./l0_sweep.sh |& tee runs/l0_sweep_log.txt


# I know these are unsorted, I just wanted to do the likely ones first in case this is slow... :|
# LAMBDAS=(1e-6 3e-6 1e-5 1.5e-5 2e-5 2.5e-5 3e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-2)
TARGET_L0=40
SEED=0

# setting eval_num_batches to 5 for speed
COMMON_ARGS=(
  --out_dir runs
  --seed "${SEED}"
  --device mps
  --dtype fp16
  --num_steps 2000
  --eval_num_batches 5
  --ckpt_every 2000
)

run() {
  "$@" || echo "[warn] FAILED: $*" >&2
}


# for lam in "${LAMBDAS[@]}"; do
for lam in 2.5e-5 3e-5 3.4e-5 3.7e-5 4e-5 4.3e-5 4.7e-5 5.2e-5; do
  echo "=== l1_uniform lam=${lam} ==="
  run python3 train.py \
    --run_name "mac_l1u_renorm_lam_${lam}" \
    --sparsity l1_uniform \
    --lambda_base "${lam}" \
    --p_start 1.0 --p_end 1.0 \
    "${COMMON_ARGS[@]}"

  echo "=== p_annealing lam=${lam} ==="
  run python3 train.py \
    --run_name "mac_p_anneal_renorm_lam_${lam}" \
    --sparsity p_annealing \
    --lambda_base "${lam}" \
    --p_start 1.0 --p_end 0.5 \
    "${COMMON_ARGS[@]}"


for lam in 6e-5 7e-5 8e-5 9e-5 1e-4 1.1e-4 1.2e-4 1.4e-4; do
  echo "=== l1_freq_weighted lam=${lam} ==="
  run python3 train.py \
    --run_name "mac_freq_l1_renorm_lam_${lam}" \
    --sparsity l1_freq_weighted \
    --lambda_base "${lam}" \
    --p_start 1.0 --p_end 1.0 \
    --fw_alpha 0.5 \
    --fw_warmup_steps 200 \
    "${COMMON_ARGS[@]}"

  echo "=== p_annealing_freq lam=${lam} ==="
  run python3 train.py \
    --run_name "mac_combined_renorm_lam_${lam}" \
    --sparsity p_annealing_freq \
    --lambda_base "${lam}" \
    --p_start 1.0 --p_end 0.5 \
    --fw_alpha 0.5 \
    --fw_warmup_steps 200 \
    "${COMMON_ARGS[@]}"
done

echo "=== batchtopk target_l0=${TARGET_L0} ==="
run python3 train.py \
  --run_name "mac_batchtopk_k_${TARGET_L0}" \
  --sparsity batchtopk \
  --target_l0 "${TARGET_L0}" \
  --btq_tie_break random \
  "${COMMON_ARGS[@]}"