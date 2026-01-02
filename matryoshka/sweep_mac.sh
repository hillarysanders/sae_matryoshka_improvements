#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------
# Final-run sweep: multiple seeds per method
# Uses Config defaults except for knobs we set explicitly.
# ---------------------------------------

# Seeds to run
SEEDS=(0 1 2)

# Lambdas chosen from your l0_sweep results
LAM_L1_UNIFORM="5e-5"
LAM_P_ANNEAL="5e-5"
LAM_FREQ="1e-4"
LAM_COMBINED="1e-4"
TARGET_L0="40"

# p-annealing target
P_END="0.5"   # keep p_start at Config default (1.0) unless you want to override too

# Stable run names (plots_for_days.py groups by these)
RUN_L1_UNIFORM="mac_l1_uniform"
RUN_P_ANNEAL="mac_p_anneal"
RUN_FREQ="mac_freq_l1"
RUN_COMBINED="mac_combined"
RUN_BTOPK="mac_batchtopk"

# Optional: log directory
LOG_DIR="runs/logs"
mkdir -p "$LOG_DIR"

run_one () {
  local run_name="$1"
  local sparsity="$2"
  local lambda_base="$3"
  local seed="$4"
  shift 4

  echo ""
  echo "============================================================"
  echo "run_name=${run_name} sparsity=${sparsity} lambda_base=${lambda_base} seed=${seed}"
  echo "============================================================"

  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="${LOG_DIR}/${run_name}_seed${seed}_${stamp}.log"

  # Build args
  args=(python3 train.py
    --run_name "${run_name}"
    --seed "${seed}"
    --sparsity "${sparsity}"
  )

  # Only pass lambda_base for penalty-based methods
  if [[ "${sparsity}" != "batchtopk" ]]; then
    args+=(--lambda_base "${lambda_base}")
  fi

  # Add any extra args
  args+=("$@")

  "${args[@]}" 2>&1 | tee "${log_file}"
}

for seed in "${SEEDS[@]}"; do
  # 1) Baseline
  run_one "${RUN_L1_UNIFORM}" "l1_uniform" "${LAM_L1_UNIFORM}" "${seed}"

  # 2) P-annealing only (override just p_end; leave schedule defaults in Config)
  run_one "${RUN_P_ANNEAL}" "p_annealing" "${LAM_P_ANNEAL}" "${seed}" \
    --p_end "${P_END}"

  # 3) Frequency-weighted L1
  run_one "${RUN_FREQ}" "l1_freq_weighted" "${LAM_FREQ}" "${seed}"

  # 4) Combined
  run_one "${RUN_COMBINED}" "p_annealing_freq" "${LAM_COMBINED}" "${seed}" \
    --p_end "${P_END}"

  # BatchTopK
  run_one "${RUN_BTOPK}" "batchtopk" "-" "${seed}" \
    --target_l0 "${TARGET_L0}"


done

echo ""
echo "All runs complete."
echo "Next:"
echo "  python3 plots_for_days.py --train_run_names ${RUN_L1_UNIFORM} ${RUN_P_ANNEAL} ${RUN_FREQ} ${RUN_COMBINED} ${RUN_BTOPK}"
