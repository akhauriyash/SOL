# 6 bit target

python action_variability.py \
  --tgt_keep 1.0 \
  --tgt_prune 1.0 \
  --tgt_quant 0.4375 \
  --quant_choices q5,q16 \
  --prune_choices s100 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path quant_action_variability.csv


python action_variability.py \
  --tgt_keep 1.0 \
  --tgt_prune 1.0 \
  --tgt_quant 0.4375 \
  --quant_choices q6,q16 \
  --prune_choices s100 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path quant_action_variability.csv


python action_variability.py \
  --tgt_keep 1.0 \
  --tgt_prune 1.0 \
  --tgt_quant 0.4375 \
  --quant_choices q5,q8 \
  --prune_choices s100 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path quant_action_variability.csv

python action_variability.py \
  --tgt_keep 1.0 \
  --tgt_prune 1.0 \
  --tgt_quant 0.4375 \
  --quant_choices q6,q8 \
  --prune_choices s100 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path quant_action_variability.csv

python action_variability.py \
  --tgt_keep 1.0 \
  --tgt_prune 1.0 \
  --tgt_quant 0.4375 \
  --quant_choices q5,q10 \
  --prune_choices s100 \
  --eval_batches 100 \
  --seed 5612 \
  --num_trials 10 \
  --csv_path quant_action_variability.csv
