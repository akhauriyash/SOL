#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="/mnt/home/ya255/projects/SOL/checkpoints/Llama8Bi-20260102-163733"
MODE="latest"
TASKS="mmlu_conceptual_physics_continuation,mmlu_high_school_chemistry_continuation,mmlu_international_law_continuation"
BATCH_SIZE=1             # pl keep batch size 1 for simplicity
LIMIT=""                 # empty = no limit, else e.g. 200
POLICY_TEMPERATURE=0.6

# Optional runtime knobs 
EPISODE_LEN=16           
DENSE_REFRESH_TAIL=16    

# Biases
SPARSITY_BIAS=0.0
PRUNE_BIAS=0.0
QUANT_BIAS=0.0

OUT_DIR="8b_lmeval_scan_$(basename "$CKPT_DIR")_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

# ---- random sampling controls ----
SAMPLES=0              # 0 = run entire grid (in random order). else run this many unique random points
SEED=12345             # reproducible shuffle seed
# ----------------------------------
PARETO_JSON="[[0.15, 0.4, 5.0], [0.15, 0.4, 6.0], [0.15, 0.4, 7.0], [0.15, 0.4, 8.0], [0.15, 0.5, 5.0], [0.15, 0.5, 6.0], [0.15, 0.5, 7.0], [0.15, 0.5, 8.0], [0.15, 0.6, 5.0], [0.15, 0.6, 6.0], [0.15, 0.6, 7.0], [0.15, 0.7, 5.0], [0.15, 0.7, 6.0], [0.25, 0.4, 7.0], [0.25, 0.5, 6.0], [0.25, 0.6, 6.0], [0.25, 0.7, 5.0], [0.25, 0.7, 6.0], [0.25, 0.7, 7.0], [0.25, 0.7, 8.0], [0.35, 0.6, 6.0], [0.35, 0.6, 7.0], [0.35, 0.7, 5.0], [0.35, 0.7, 6.0], [0.35, 0.7, 7.0], [0.35, 0.7, 8.0], [0.35, 1.0, 8.0], [0.45, 0.6, 5.0], [0.45, 0.7, 6.0], [0.45, 0.7, 8.0], [0.45, 0.7, 9.0], [0.45, 0.8, 8.0], [0.55, 0.8, 8.0], [0.55, 0.8, 9.0], [0.55, 0.9, 8.0], [0.55, 0.9, 9.0], [0.55, 1.0, 8.0], [0.65, 0.8, 9.0], [0.65, 0.9, 9.0], [0.65, 0.9, 10.0], [0.65, 1.0, 8.0], [0.75, 0.8, 9.0], [0.75, 0.9, 8.0], [0.75, 0.9, 9.0], [0.75, 0.9, 10.0], [0.85, 0.9, 9.0], [0.85, 0.9, 11.0], [0.85, 1.0, 12.0], [0.95, 1.0, 13.0]]"
# ---------------------------------------------------------------

# ---- parallel partition controls ----
WORKERS=4
WORKER_ID=2
# -------------------------------------
mapfile -t COMBOS < <(python - "$PARETO_JSON" <<'PY'
import json, sys

tuples = json.loads(sys.argv[1])
for k, p, b in tuples:
    b = int(round(float(b)))
    print(f"{float(k):.2f} {float(p):.2f} {b}")
PY
)

# Hard fail if parsing produced nothing
if (( ${#COMBOS[@]} == 0 )); then
  echo "[error] COMBOS is empty. PARETO_JSON likely failed to parse." >&2
  exit 2
fi

TOTAL_ALL=${#COMBOS[@]}

mapfile -t COMBOS < <(
  printf "%s\n" "${COMBOS[@]}" |
  python -c 'import sys,random
seed=int(sys.argv[1])
random.seed(seed)
lines=sys.stdin.read().splitlines()
random.shuffle(lines)
print("\n".join(lines))' "$SEED"
)

# Partition: each worker takes every WORKERS-th item
PARTITIONED=()
for ((i=0; i<${#COMBOS[@]}; i++)); do
  if (( i % WORKERS == WORKER_ID )); then
    PARTITIONED+=("${COMBOS[i]}")
  fi
done
COMBOS=("${PARTITIONED[@]}")
TOTAL_ALL=${#COMBOS[@]}

# Optional random subsampling within this worker's partition
if (( SAMPLES > 0 && SAMPLES < TOTAL_ALL )); then
  TOTAL=$SAMPLES
else
  TOTAL=$TOTAL_ALL
fi

echo "Pareto list size: ${#PARTITIONED[@]} for worker ${WORKER_ID}/${WORKERS} (seed=$SEED)"
echo "Running: $TOTAL points"

echo "Pareto combos in this worker: $TOTAL_ALL"
echo "Running:   $TOTAL unique random combinations (seed=$SEED)"
echo "Output:    $OUT_DIR"

# -------- helper to build optional args --------
LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

EP_ARGS=()
if [[ -n "${EPISODE_LEN}" ]]; then
  EP_ARGS+=(--episode_len "$EPISODE_LEN")
fi
if [[ -n "${DENSE_REFRESH_TAIL}" ]]; then
  EP_ARGS+=(--dense_refresh_tail "$DENSE_REFRESH_TAIL")
fi
# ---------------------------------------------

# -------- ETA helpers --------
START_EPOCH=$(date +%s)
ITER=0

fmt_hms () {
  local s=$1
  printf '%02d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60))
}
# ----------------------------

# 1) Dense baseline ONCE
DENSE_JSON="$OUT_DIR/dense_only.json"
if [[ ! -f "$DENSE_JSON" ]]; then
  echo
  echo "[dense-only] Running dense baseline once..."
  python eval_policy_lmeval.py \
    --ckpt_dir "$CKPT_DIR" \
    --mode "$MODE" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    "${LIMIT_ARGS[@]}" \
    "${EP_ARGS[@]}" \
    --policy_temperature "$POLICY_TEMPERATURE" \
    --sparsity_bias "$SPARSITY_BIAS" \
    --prune_bias "$PRUNE_BIAS" \
    --quant_bias "$QUANT_BIAS" \
    --only_dense \
    --export_sparsity_json "$DENSE_JSON"
fi

# 2) Grid: policy + fixed_from_policy per target
for ((idx=0; idx<TOTAL; idx++)); do
  ITER=$((ITER+1))
  read -r keep prune bits <<< "${COMBOS[idx]}"

  keep_tag=${keep//./p}
  prune_tag=${prune//./p}
  OUT_JSON="$OUT_DIR/policy_k${keep_tag}_p${prune_tag}_b${bits}.json"

  if [[ -f "$OUT_JSON" ]]; then
    echo
    echo "[${ITER}/${TOTAL}] (skip existing) keep=$keep prune=$prune bits=$bits → $(basename "$OUT_JSON")"
    continue
  fi

  echo
  echo "[${ITER}/${TOTAL}] keep=$keep prune=$prune bits=$bits"
  CUDA_VISIBLE_DEVICES=3 python eval_policy_lmeval.py \
    --ckpt_dir "$CKPT_DIR" \
    --mode "$MODE" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    "${LIMIT_ARGS[@]}" \
    "${EP_ARGS[@]}" \
    --policy_temperature "$POLICY_TEMPERATURE" \
    --sparsity_bias "$SPARSITY_BIAS" \
    --prune_bias "$PRUNE_BIAS" \
    --quant_bias "$QUANT_BIAS" \
    --tgt_keep "$keep" \
    --tgt_prune_keep "$prune" \
    --tgt_quant_bits "$bits" \
    --fixed_from_policy \
    --export_sparsity_json "$OUT_JSON"

  NOW_EPOCH=$(date +%s)
  ELAPSED=$((NOW_EPOCH - START_EPOCH))

  AVG_PER_ITER=$(( ELAPSED / ITER ))
  REM=$(( TOTAL - ITER ))
  ETA=$(( AVG_PER_ITER * REM ))

  printf '[%d/%d] elapsed=%s | eta=%s | done_at=%s\n' \
    "$ITER" "$TOTAL" \
    "$(fmt_hms "$ELAPSED")" "$(fmt_hms "$ETA")" \
    "$(date -d "@$((NOW_EPOCH + ETA))" '+%Y-%m-%d %H:%M:%S')"
done

echo
echo "Done."
echo "Dense baseline: $DENSE_JSON"
echo "Grid outputs:   $OUT_DIR"
