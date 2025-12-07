
# 50% pruning target

python action_variability.py \
  --tgt_keep 0.5 \
  --tgt_prune 1.0 \
  --tgt_quant 1.0 \
  --quant_choices q16 \
  --prune_choices s100 \
  --keep_fracs 1.0,0.1 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path toksparse_action_variability.csv

python action_variability.py \
  --tgt_keep 0.5 \
  --tgt_prune 1.0 \
  --tgt_quant 1.0 \
  --quant_choices q16 \
  --prune_choices s100 \
  --keep_fracs 1.0,0.2 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path toksparse_action_variability.csv



python action_variability.py \
  --tgt_keep 0.5 \
  --tgt_prune 1.0 \
  --tgt_quant 1.0 \
  --quant_choices q16 \
  --prune_choices s100 \
  --keep_fracs 1.0,0.4 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path toksparse_action_variability.csv


python action_variability.py \
  --tgt_keep 0.5 \
  --tgt_prune 1.0 \
  --tgt_quant 1.0 \
  --quant_choices q16 \
  --prune_choices s100 \
  --keep_fracs 0.6,0.4 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path toksparse_action_variability.csv


python action_variability.py \
  --tgt_keep 0.5 \
  --tgt_prune 1.0 \
  --tgt_quant 1.0 \
  --quant_choices q16 \
  --prune_choices s100 \
  --keep_fracs 0.7,0.3 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path toksparse_action_variability.csv
