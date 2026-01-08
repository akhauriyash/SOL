#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="/mnt/home/ya255/projects/SOL/checkpoints/Llama8Bi-20260102-163733"
DATASET="wikitext"
EVAL_BATCHES=7
NUM_TRIALS=1
CSV_PATH="Llama8Bi_PPLEval.csv"

# ---- random sampling controls ----
SAMPLES=0              # 0 = run entire grid (in random order). else run this many unique random points
SEED=12345             # reproducible shuffle seed (set SEED=$(date +%s) for new order each run)
# ----------------------------------

# Build lists with integer steps to avoid float drift
KEEP_LIST=()
for k in $(seq 15 10 95); do
  KEEP_LIST+=($(awk -v x=$k 'BEGIN{printf "%.2f", x/100}'))
done

PRUNE_LIST=()
for p in $(seq 40 10 100); do
  PRUNE_LIST+=($(awk -v x=$p 'BEGIN{printf "%.2f", x/100}'))
done

BITS_LIST=($(seq 5 13))

K=${#KEEP_LIST[@]}
P=${#PRUNE_LIST[@]}
Q=${#BITS_LIST[@]}
TOTAL_ALL=$((K*P*Q))

# Build ALL combos once
COMBOS=()
for keep in "${KEEP_LIST[@]}"; do
  for prune in "${PRUNE_LIST[@]}"; do
    for bits in "${BITS_LIST[@]}"; do
      COMBOS+=("$keep $prune $bits")
    done
  done
done

# Shuffle combos in-place (seeded Fisher–Yates)
RANDOM=$SEED
for ((i=${#COMBOS[@]}-1; i>0; i--)); do
  j=$((RANDOM % (i+1)))
  tmp=${COMBOS[i]}
  COMBOS[i]=${COMBOS[j]}
  COMBOS[j]=$tmp
done

# Decide how many to run
if (( SAMPLES > 0 && SAMPLES < TOTAL_ALL )); then
  TOTAL=$SAMPLES
else
  TOTAL=$TOTAL_ALL
fi

echo "Grid size: $TOTAL_ALL combinations (K=$K, P=$P, Q=$Q)"
echo "Running:   $TOTAL unique random combinations (seed=$SEED)"

# -------- ETA helpers --------
START_EPOCH=$(date +%s)
ITER=0

fmt_hms () {
  # seconds -> HH:MM:SS
  local s=$1
  printf '%02d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60))
}
# ----------------------------

for ((idx=0; idx<TOTAL; idx++)); do
  ITER=$((ITER+1))
  read -r keep prune bits <<< "${COMBOS[idx]}"

  python policy_action_variability.py \
    --ckpt_dir "$CKPT_DIR" \
    --dataset_name "$DATASET" \
    --eval_batches "$EVAL_BATCHES" \
    --tgt_keep "$keep" \
    --tgt_prune_keep "$prune" \
    --tgt_quant_bits "$bits" \
    --num_trials "$NUM_TRIALS" \
    --csv_path "$CSV_PATH"

  NOW_EPOCH=$(date +%s)
  ELAPSED=$((NOW_EPOCH - START_EPOCH))

  AVG_PER_ITER=$(( ELAPSED / ITER ))
  REM=$(( TOTAL - ITER ))
  ETA=$(( AVG_PER_ITER * REM ))

  printf '\n[%d/%d] keep=%s prune=%s bits=%s | elapsed=%s | eta=%s | done_at=%s' \
    "$ITER" "$TOTAL" "$keep" "$prune" "$bits" \
    "$(fmt_hms "$ELAPSED")" "$(fmt_hms "$ETA")" \
    "$(date -d "@$((NOW_EPOCH + ETA))" '+%Y-%m-%d %H:%M:%S')"
done

echo

