OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --wandb_project SOL_Reason --wandb_run_name DistilLlama8B \
  --config /mnt/home/ya255/projects/SOL/official_configs/ContextualDistil8B.yml --total_updates 7800
