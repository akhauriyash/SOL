
# 50% pruning target

python action_variability.py \
  --tgt_keep 1.0 \
  --tgt_prune 0.6 \
  --tgt_quant 1.0 \
  --quant_choices q16 \
  --prune_choices s40,s100 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path prune_action_variability.csv

python action_variability.py \
  --tgt_keep 1.0 \
  --tgt_prune 0.6 \
  --tgt_quant 1.0 \
  --quant_choices q16 \
  --prune_choices s30,s100 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path prune_action_variability.csv

python action_variability.py \
  --tgt_keep 1.0 \
  --tgt_prune 0.6 \
  --tgt_quant 1.0 \
  --quant_choices q16 \
  --prune_choices s50,s100 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path prune_action_variability.csv


python action_variability.py \
  --tgt_keep 1.0 \
  --tgt_prune 0.6 \
  --tgt_quant 1.0 \
  --quant_choices q16 \
  --prune_choices s40,s80 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path prune_action_variability.csv


python action_variability.py \
  --tgt_keep 1.0 \
  --tgt_prune 0.6 \
  --tgt_quant 1.0 \
  --quant_choices q16 \
  --prune_choices s50,s70 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path prune_action_variability.csv

