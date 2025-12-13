
### New Longer Runs post FALURE
## Non lagrangian
# v1 failed
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29513 train.py   \
  --wandb_project SOLmv     --wandb_run_name AllVariantsv1    \
   --config /mnt/home/ya255/projects/SOL/official_configs/All_Variants.yml

# v2 finished
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 29515 train.py   \
  --wandb_project SOLmv     --wandb_run_name AllVariantsv2    \
   --config /mnt/home/ya255/projects/SOL/official_configs/All_Variantsv2.yml




CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29518 train.py   \
  --wandb_project SOLmv     --wandb_run_name AllVariantsv4    \
   --config /mnt/home/ya255/projects/SOL/official_configs/All_Variantsv4.yml



CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 29516 train.py   \
  --wandb_project SOLmv     --wandb_run_name AllVariantsv3    \
   --config /mnt/home/ya255/projects/SOL/official_configs/All_Variantsv3.yml



CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29516 train.py   \
  --wandb_project SOLmv     --wandb_run_name AllVariantsvMax    \
   --config /mnt/home/ya255/projects/SOL/official_configs/All_VariantsvMax.yml



CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29544 train.py   \
  --wandb_project SOLmv2     --wandb_run_name AllVariants_vMax2    \
   --config /mnt/home/ya255/projects/SOL/official_configs/All_Variants_vMax2.yml



CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29554 train.py   \
  --wandb_project SOLmv2     --wandb_run_name AllVariants_vMax    \
   --config /mnt/home/ya255/projects/SOL/official_configs/All_Variants_vMax.yml
