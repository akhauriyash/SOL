# Sparsity Only: /mnt/home/ya255/projects/SOL/newckpt/v2RL_LCE_SparseQ4-20251127-150927
# Quantization Only: /mnt/home/ya255/projects/SOL/newckpt/v2RL_LCE_Quant-20251126-181143
# Pruning Only: /mnt/home/ya255/projects/SOL/newckpt/v2RL_LCE_Prune-20251127-095012
# Joint Method: /mnt/home/ya255/projects/SOL/newckpt/v2RL_LCE_Q8_PQ-20251126-181144



# CKPT_DIR="/mnt/home/ya255/projects/SOL/checkpoints/nLRL_LCE_Quant-20251201-224315"
# CRITERIA="quest"
# MODE="latest"
# DATASET_NAME="wikitext"
# OUTP_PREFIX="quant_withtarget"

# START=-100
# END=0.7
# STEP=0.1

# i=0
# total=$(python - << 'EOF'
# start = -100
# end = 0.7
# step = 0.1
# import math
# n = int(math.floor((end - start) / step + 1e-9)) + 1
# print(n)
# EOF
# )

# for p in $(seq ${START} ${STEP} ${END}); do
#   echo "=== Running prune sweep: tgt_quant_ratio=${p} (${i}/${total}) ==="
#   time python target_meff.py \
#     --ckpt_dir "${CKPT_DIR}" \
#     --criteria "${CRITERIA}" \
#     --mode "${MODE}" \
#     --dataset_name "${DATASET_NAME}" \
#     --outp "${OUTP_PREFIX}" \
#     --tgt_quant_ratio "${p}"

#   i=$((i + 1))
#   echo "[${i}/${total}] done tgt_quant_ratio=${p}"
# done


CKPT_DIR="/mnt/home/ya255/projects/SOL/checkpoints/nLRL_LCE_Prune-20251204-114619"
CRITERIA="quest"
MODE="latest"
DATASET_NAME="wikitext"
OUTP_PREFIX="prunev5_withtarget"

START=-0.4
END=0.8
STEP=0.05

i=0
total=$(python - << 'EOF'
start = -0.4
end = 0.8
step = 0.05
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
    --tgt_prune_keep "${p}"

  i=$((i + 1))
  echo "[${i}/${total}] done tgt_prune_keep=${p}"
done


# # Prune Only
# SPARSITY_LIST=(0.0)
# # PRUNE_LIST=(24 26 28 30 34 36 38 42 45 48 50 54 58 62 64 -64 -62 -60 -58 -54 -50 -48 -45 -42 -40 -38 -36 -34 -30 -28 -26 -24)
# PRUNE_LIST=(0 5 10 15 20 25 30 35 40 45 50)
# QUANT_LIST=(0.0)
# for s in "${SPARSITY_LIST[@]}"; do
#   for p in "${PRUNE_LIST[@]}"; do
#     for q in "${QUANT_LIST[@]}"; do
#       time python target_meff.py \
#         --ckpt_dir /mnt/home/ya255/projects/SOL/checkpoints/nLRL_LCE_Prune-20251201-224314 \
#         --criteria "quest" \
#         --mode "latest" \
#         --dataset_name wikitext \
#         --outp "prune_withtarget_spb" \
#         --sparsity_bias "$s" \
#         --prune_bias "$p" \
#         --quant_bias "$q"
#       i=$(( ${i:-0} + 1 )); total=$(( ${#SPARSITY_LIST[@]} * ${#PRUNE_LIST[@]} * ${#QUANT_LIST[@]} )); echo "[${i}/${total}] done s=${s} p=${p} q=${q}"
#     done
#   done
# done

# # Sparse Only
# # SPARSITY_LIST=(-10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10 12)
# SPARSITY_LIST=(-50 -45 -40 -35 -30 -25 -20 -18 -16 -14 -12 12 14 16 18 20 25 30 35 40 45 50)
# PRUNE_LIST=(0.0)
# QUANT_LIST=(0.0)
# for s in "${SPARSITY_LIST[@]}"; do
#   for p in "${PRUNE_LIST[@]}"; do
#     for q in "${QUANT_LIST[@]}"; do
#       time python multi_efficiency_test.py \
#         --ckpt_dir /mnt/home/ya255/projects/SOL/newckpt/v2RL_LCE_SparseQ4-20251127-150927 \
#         --mode "latest" \
#         --dataset_name wikitext \
#         --criteria "quest" \
#         --outp "sparse_only" \
#         --sparsity_bias "$s" \
#         --prune_bias "$p" \
#         --quant_bias "$q"
#       i=$(( ${i:-0} + 1 )); total=$(( ${#SPARSITY_LIST[@]} * ${#PRUNE_LIST[@]} * ${#QUANT_LIST[@]} )); echo "[${i}/${total}] done s=${s} p=${p} q=${q}"
#     done
#   done
# done

# # Quant Only
# SPARSITY_LIST=(0.0)
# PRUNE_LIST=(0.0)
# QUANT_LIST=(-20 -18 -16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16 18 20)
# # QUANT_LIST=(-17.8 -17.6 -17.4 -17.2 -16.8 -16.6 -16.4 -16.2  -15.8 -15.6 -15.4 -15.2 -14.8 -14.6 -14.4 -14.2 -13.5)
# # QUANT_LIST=(-17.1 -17.05 -16.95 -16.9 -16.85 -16.8 -16.75 -16.7 -16.65 -16.6 -16.55 -16.5 -16.45 -16.4 -16.35 -16.3 -16.25 -16.2 -16.15 -16.1 -16.05 -15.95 -15.9 -15.85 -15.8 -15.75 -15.7 -15.65 -15.6 -15.55 -15.5 -15.45 -15.4 -15.35 -15.3 -15.25 -15.2 -15.15 -15.1 -15.05)
# for s in "${SPARSITY_LIST[@]}"; do
#   for p in "${PRUNE_LIST[@]}"; do
#     for q in "${QUANT_LIST[@]}"; do
#       time python multi_efficiency_test.py \
#         --ckpt_dir /mnt/home/ya255/projects/SOL/checkpoints/v3RL_LCE_Quant-20251129-183433 \
#         --mode "latest" \
#         --dataset_name wikitext \
#         --criteria "quest" \
#         --outp "quant_only" \
#         --sparsity_bias "$s" \
#         --prune_bias "$p" \
#         --quant_bias "$q"
#       i=$(( ${i:-0} + 1 )); total=$(( ${#SPARSITY_LIST[@]} * ${#PRUNE_LIST[@]} * ${#QUANT_LIST[@]} )); echo "[${i}/${total}] done s=${s} p=${p} q=${q}"
#     done
#   done
# done
