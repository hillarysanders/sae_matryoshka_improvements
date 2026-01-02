#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=false

# caffeinate -dimsu ./l0_sweep.sh |& tee runs/l0_sweep_log.txt


# I know these are unsorted, I just wanted to do the likely ones first in case this is slow... :|
# LAMBDAS=(1e-6 3e-6 1e-5 1.5e-5 2e-5 2.5e-5 3e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-2)
TARGET_L0=40
SEED=0

# setting eval_num_batches to 5 for speed
# 8x fewer steps because we've bumped batch_size up by 8x to 16: max before OOMing on a g5
COMMON_ARGS=(
  --out_dir runs
  --seed "${SEED}"
  --device cuda
  --dtype bf16
  --num_steps 250
  --batch_size 16
  --eval_num_batches 5
  --ckpt_every 250
)

echo "COMMON ARGS:"
echo "${COMMON_ARGS[@]}"

run() {
  "$@" || echo "[warn] FAILED: $*" >&2
}


# for lam in "${LAMBDAS[@]}"; do
# for lam in 2.5e-5 3e-5; do
# for lam in 3.4e-5 3.7e-5; do
# for lam in 4e-5 4.3e-5; do
# for lam in 4.7e-5 5.2e-5; do
# for lam in 2e-5 2.3e-5; do
for lam in 2.1e-5 2.4e-5; do
#   echo "=== l1_uniform lam=${lam} ==="
#   run python3 train.py \
#     --run_name "ec2_l1u_renorm_lam_${lam}" \
#     --sparsity l1_uniform \
#     --lambda_base "${lam}" \
#     --p_start 1.0 --p_end 1.0 \
#     "${COMMON_ARGS[@]}"
  echo "=== p_annealing lam=${lam} ==="
  run python3 train.py \
    --run_name "ec2_p_anneal_renorm_lam_${lam}" \
    --sparsity p_annealing \
    --lambda_base "${lam}" \
    --p_start 1.0 --p_end 0.5 \
    "${COMMON_ARGS[@]}"
done

# for lam in 1e-6; do
# for lam in 3e-6; do
# for lam in 5e-6; do
# for lam in 6e-6; do
# for lam in 7e-6; do
# for lam in 7.5e-6; do
# for lam in 6e-5 7e-5; do
# for lam in 8e-5 9e-5; do
# for lam in 1e-4 1.1e-4; do
# for lam in 1.2e-4 1.4e-4; do
#   echo "=== l1_freq_weighted lam=${lam} ==="
#   run python3 train.py \
#     --run_name "ec2_freq_l1_renorm_lam_${lam}" \
#     --sparsity l1_freq_weighted \
#     --lambda_base "${lam}" \
#     --p_start 1.0 --p_end 1.0 \
#     --fw_alpha 0.5 \
#     --fw_warmup_steps 25 \
#     "${COMMON_ARGS[@]}"

#   echo "=== p_annealing_freq lam=${lam} ==="
#   run python3 train.py \
#     --run_name "ec2_combined_renorm_lam_${lam}" \
#     --sparsity p_annealing_freq \
#     --lambda_base "${lam}" \
#     --p_start 1.0 --p_end 0.5 \
#     --fw_alpha 0.5 \
#     --fw_warmup_steps 25 \
#     "${COMMON_ARGS[@]}"
# done

# echo "=== batchtopk target_l0=${TARGET_L0} ==="
# run python3 train.py \
#   --run_name "ec2_batchtopk_k_${TARGET_L0}" \
#   --sparsity batchtopk \
#   --target_l0 "${TARGET_L0}" \
#   --btq_tie_break random \
#   "${COMMON_ARGS[@]}"

