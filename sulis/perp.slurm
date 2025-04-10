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

# ---------- Get perplexities for ground truth abstracts ----------
srun python3 get_perplexity.py 'LLMsForHepth/hep-th_primary' -a s1 s2 s3 s4 s5 s6 s7 s8 s9 s10

srun python3 get_perplexity.py 'LLMsForHepth/hep-th_perplexities' -a s1_qkv s2_qkv s3_qkv s4_qkv s5_qkv s6_qkv s7_qkv s8_qkv s9_qkv s10_qkv -nb