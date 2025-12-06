#!/usr/bin/env bash

PRUNE_CKPT_DIR="/mnt/home/ya255/projects/SOL/current_valid/nLRL_LCE_Prune-20251205-111230"
QUANT_CKPT_DIR="/mnt/home/ya255/projects/SOL/current_valid/nLRL_LCE_Quant-20251205-111229"
TOKSPARSE_CKPT_DIR="/mnt/home/ya255/projects/SOL/current_valid/nLRL_LCE_TokSparse-20251205-233007"
ALL_CKPT_DIR="/mnt/home/ya255/projects/SOL/current_valid/nLRL_LCE_All-20251205-233007"


CRITERIA="quest"
MODE="latest"            # passed through to target_meff.py
DATASET_NAME="wikitext"
BASE_OUTP_PREFIX="dec6"

START=0.15
END=0.93
STEP=0.03

# ------------ which experiment to run? ------------
RUN_MODE="${1:-prune}"   # prune | quant | toksparse | all

case "${RUN_MODE}" in
  prune)
    CKPT_DIR="${PRUNE_CKPT_DIR}"
    OUTP_PREFIX="${BASE_OUTP_PREFIX}_prune"
    SWEEP_FLAG="tgt_prune_keep"
    ;;
  quant)
    CKPT_DIR="${QUANT_CKPT_DIR}"
    OUTP_PREFIX="${BASE_OUTP_PREFIX}_quant"
    SWEEP_FLAG="tgt_quant_ratio"
    ;;
  toksparse)
    CKPT_DIR="${TOKSPARSE_CKPT_DIR}"
    OUTP_PREFIX="${BASE_OUTP_PREFIX}_toksparse"
    SWEEP_FLAG="tgt_keep"
    ;;
  all)
    CKPT_DIR="${ALL_CKPT_DIR}"
    OUTP_PREFIX="${BASE_OUTP_PREFIX}_all"
    ;;
  *)
    echo "Unknown mode: ${RUN_MODE}"
    echo "Usage: $0 {prune|quant|toksparse|all}"
    exit 1
    ;;
esac

echo "[run] RUN_MODE=${RUN_MODE}"
echo "[run] CKPT_DIR=${CKPT_DIR}"
echo "[run] OUTP_PREFIX=${OUTP_PREFIX}"

# ------------ single-parameter sweeps ------------
if [[ "${RUN_MODE}" != "all" ]]; then
  i=0
  total=$(python - << 'EOF'
start = 0.15
end = 0.93
step = 0.03
import math
n = int(math.floor((end - start) / step + 1e-9)) + 1
print(n)
EOF
)

  for p in $(seq "${START}" "${STEP}" "${END}"); do
    echo "=== [${RUN_MODE}] sweep: ${SWEEP_FLAG}=${p} (${i}/${total}) ==="
    time python target_meff.py \
      --ckpt_dir "${CKPT_DIR}" \
      --criteria "${CRITERIA}" \
      --mode "${MODE}" \
      --dataset_name "${DATASET_NAME}" \
      --outp "${OUTP_PREFIX}" \
      --${SWEEP_FLAG} "${p}"

    i=$((i + 1))
    echo "[${RUN_MODE}] [${i}/${total}] done ${SWEEP_FLAG}=${p}"
  done

  exit 0
fi

# ------------ ALL mode: 3×3×3 grid over 0.3, 0.5, 0.8 ------------
vals=(0.3 0.5 0.8)
total=$(( ${#vals[@]} * ${#vals[@]} * ${#vals[@]} ))
i=0

for prune_keep in "${vals[@]}"; do
  for quant_ratio in "${vals[@]}"; do
    for tok_keep in "${vals[@]}"; do
      echo "=== [all] combo: prune=${prune_keep}, quant=${quant_ratio}, tok=${tok_keep} (${i}/${total}) ==="

      # Optional: put the combo into the output prefix so files are unique
      combo_outp="${OUTP_PREFIX}"

      time python target_meff.py \
        --ckpt_dir "${CKPT_DIR}" \
        --criteria "${CRITERIA}" \
        --mode "${MODE}" \
        --dataset_name "${DATASET_NAME}" \
        --outp "${combo_outp}" \
        --tgt_prune_keep "${prune_keep}" \
        --tgt_quant_ratio "${quant_ratio}" \
        --tgt_keep "${tok_keep}"

      i=$((i + 1))
      echo "[all] [${i}/${total}] done prune=${prune_keep}, quant=${quant_ratio}, tok=${tok_keep}"
    done
  done
done