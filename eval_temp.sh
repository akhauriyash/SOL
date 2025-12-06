
CKPT_DIR="/mnt/home/ya255/projects/SOL/checkpoints/nLRL_LCE_Quant-20251205-111229"
CRITERIA="quest"
MODE="latest"
DATASET_NAME="wikitext"
OUTP_PREFIX="quantv_f_withtarget"

START=0.21
END=0.93
STEP=0.03

i=0
total=$(python - << 'EOF'
start = 0.21
end = 0.93
step = 0.03
import math
n = int(math.floor((end - start) / step + 1e-9)) + 1
print(n)
EOF
)

for p in $(seq ${START} ${STEP} ${END}); do
  echo "=== Running prune sweep: tgt_prune_keep=${p} (${i}/${total}) ==="
  time python target_meff.py \
    --ckpt_dir "${CKPT_DIR}" \
    --criteria "${CRITERIA}" \
    --mode "${MODE}" \
    --dataset_name "${DATASET_NAME}" \
    --outp "${OUTP_PREFIX}" \
    --tgt_quant_ratio "${p}"

  i=$((i + 1))
  echo "[${i}/${total}] done tgt_quant_ratio=${p}"
done

