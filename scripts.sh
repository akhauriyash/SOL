# Single GPU Training Example

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29513 train.py   \
  --wandb_project SOL_RLS_MSC     --wandb_run_name Llama8Bi    \
   --config /mnt/home/ya255/projects/SOL/official_configs/All_Variants_Llama8Bi.yml

# Multi GPU (4) Training Example

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --wandb_project SOL_RLS_MSC8B --wandb_run_name Llama8Bi \
  --config /mnt/home/ya255/projects/SOL/official_configs/All_Variants_Llama8Bi.yml --total_updates 1560

