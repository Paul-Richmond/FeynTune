#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:3
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su008-apw822

#set -x  #<-- uncomment for verbose output

module purge
module load GCC/12.2.0 CUDA/12.2.0 Python/3.10.8

source ~/vllm_take2/bin/activate

# Uncomment to allow wandb tracking from last saved checkpoint
#export WANDB_RESUME=allow
#export WANDB_RUN_ID=

cd ~/repos/hepthLlama/src/

# ---------- Run finetuning on base model ----------
srun -n 1 -G 3 python3 finetune.py training.training_args_cfg.run_name=s2-L-3.1-8B-qkv \
dataset.name=LLMsForHepth/hep-ph_gr-qc_primary \
model.lora_cfg.target_modules='[k_proj, q_proj, v_proj]'

#srun -n 1 -G 3 python3 finetune.py training.training_args_cfg.run_name=s2-L-3.1-8B-qkv \
#dataset.name=LLMsForHepth/hep-ph_gr-qc_primary \
#model.lora_cfg.target_modules='[k_proj, q_proj, v_proj]' \
#training.resume_from_checkpoint=true \
#training.training_args_cfg.eval_on_start=false