#### v0 models ####
# Sparsity Only: /mnt/home/ya255/projects/SOL/newckpt/v2RL_LCE_SparseQ4-20251127-150927
# Quantization Only: /mnt/home/ya255/projects/SOL/newckpt/v2RL_LCE_Quant-20251126-181143
# Pruning Only: /mnt/home/ya255/projects/SOL/newckpt/v2RL_LCE_Prune-20251127-095012
# Joint Method: /mnt/home/ya255/projects/SOL/newckpt/v2RL_LCE_Q8_PQ-20251126-181144

#### v1 models ####
# Sparsity Only: 
# Quantization Only: /mnt/home/ya255/projects/SOL/checkpoints/v3RL_LCE_Quant-20251129-183433
# Pruning Only: /mnt/home/ya255/projects/SOL/checkpoints/v3RL_LCE_Prune-20251129-183448
# Joint Method: 


# SPARSITY_LIST=(-10 -8 -6 -4 -2 0 2 4 6 8 10 12)
# PRUNE_LIST=(-30 -26 -16 -8 -4 0 4 8 16 26 30) 
# QUANT_LIST=(-20 -16 -8 -2 0 2 8 16 20)
# for s in "${SPARSITY_LIST[@]}"; do
#   for p in "${PRUNE_LIST[@]}"; do
#     for q in "${QUANT_LIST[@]}"; do
#       time python multi_efficiency_test.py \
#         --ckpt_dir /mnt/home/ya255/projects/SOL/newckpt/v2RL_LCE_Q8_PQ-20251126-181144 \
#         --mode "latest" \
#         --criteria "quest" \
#         --outp "joint_method" \
#         --dataset_name wikitext \
#         --sparsity_bias "$s" \
#         --prune_bias "$p" \
#         --quant_bias "$q"
#       i=$(( ${i:-0} + 1 )); total=$(( ${#SPARSITY_LIST[@]} * ${#PRUNE_LIST[@]} * ${#QUANT_LIST[@]} )); echo "[${i}/${total}] done s=${s} p=${p} q=${q}"
#     done
#   done
# done


# Quant Only
SPARSITY_LIST=(0.0)
PRUNE_LIST=(0.0)
QUANT_LIST=(-20 -18 -16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16 18 20)
# QUANT_LIST=(-45 -40 -35 -30 -25 25 30 35 40 45)
# QUANT_LIST=(-31 -32 -33 -34 -35 -36 -37 -38 -39)
# QUANT_LIST=(-17.8 -17.6 -17.4 -17.2 -16.8 -16.6 -16.4 -16.2  -15.8 -15.6 -15.4 -15.2 -14.8 -14.6 -14.4 -14.2 -13.5)
# QUANT_LIST=(-17.1 -17.05 -16.95 -16.9 -16.85 -16.8 -16.75 -16.7 -16.65 -16.6 -16.55 -16.5 -16.45 -16.4 -16.35 -16.3 -16.25 -16.2 -16.15 -16.1 -16.05 -15.95 -15.9 -15.85 -15.8 -15.75 -15.7 -15.65 -15.6 -15.55 -15.5 -15.45 -15.4 -15.35 -15.3 -15.25 -15.2 -15.15 -15.1 -15.05)
for s in "${SPARSITY_LIST[@]}"; do
  for p in "${PRUNE_LIST[@]}"; do
    for q in "${QUANT_LIST[@]}"; do
      time python multi_efficiency_test.py \
        --ckpt_dir /mnt/home/ya255/projects/SOL/checkpoints/v3RL_LCE_Quant60pc-20251130-101520 \
        --mode "latest" \
        --dataset_name wikitext \
        --criteria "quest" \
        --outp "quant_only_60pc" \
        --sparsity_bias "$s" \
        --prune_bias "$p" \
        --quant_bias "$q"
      i=$(( ${i:-0} + 1 )); total=$(( ${#SPARSITY_LIST[@]} * ${#PRUNE_LIST[@]} * ${#QUANT_LIST[@]} )); echo "[${i}/${total}] done s=${s} p=${p} q=${q}"
    done
  done
done
