#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su008-apw822

#set -x  #<-- uncomment for verbose output

module purge
module load GCC/12.2.0 CUDA/12.2.0 Python/3.10.8

source ~/vllm_take2/bin/activate





cd ~/repos/hepthLlama/src/

# ---------- Run inference on finetuned model ----------
srun python3 infer_hf.py ~dataset.splits.train=train \
inference.model_name=LLMsForHepth/s1-L-3.1-8B-base
