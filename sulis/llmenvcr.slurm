#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --account=su008-apw281

module load CUDA/11.8.0

srun python3 -m venv LLM

source LLM/bin/activate

srun python3 -m pip install --upgrade pip
srun python3 -m pip install huggingface_hub datasets transformers torch wandb bitsandbytes accelerate peft python-dotenv flash-attn hydra-core

export BNB_CUDA_VERSION=118

srun python3 -m bitsandbytes


