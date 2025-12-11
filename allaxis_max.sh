#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="/mnt/home/ya255/projects/SOL/checkpoints/AllVariantsvMax-20251210-154726"
DATASET="wikitext"
EVAL_BATCHES=7
NUM_TRIALS=1           # tip: use 0 for a quick sweep
CSV_PATH="allaxis_v1.csv"

# Build lists with integer steps to avoid float drift
KEEP_LIST=()
for k in $(seq 15 5 95); do
  KEEP_LIST+=($(awk -v x=$k 'BEGIN{printf "%.2f", x/100}'))
done
# Uncomment the next line if you also want to include 1.00 (not reached by 0.05+0.1*n)
# KEEP_LIST+=("1.00")

PRUNE_LIST=()
for p in $(seq 40 5 100); do
  PRUNE_LIST+=($(awk -v x=$p 'BEGIN{printf "%.2f", x/100}'))
done

BITS_LIST=($(seq 5 13))

# Show grid size
K=${#KEEP_LIST[@]}
P=${#PRUNE_LIST[@]}
Q=${#BITS_LIST[@]}
echo "Grid size: $((K*P*Q)) combinations (K=$K, P=$P, Q=$Q)"

# Sweep
for keep in "${KEEP_LIST[@]}"; do
  for prune in "${PRUNE_LIST[@]}"; do
    for bits in "${BITS_LIST[@]}"; do
      python policy_action_variability.py \
        --ckpt_dir "$CKPT_DIR" \
        --dataset_name "$DATASET" \
        --eval_batches "$EVAL_BATCHES" \
        --tgt_keep "$keep" \
        --tgt_prune_keep "$prune" \
        --tgt_quant_bits "$bits" \
        --num_trials "$NUM_TRIALS" \
        --csv_path "$CSV_PATH"
    done
  done
done
