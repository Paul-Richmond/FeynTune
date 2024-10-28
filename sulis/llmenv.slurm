#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --account=su008-apw822

cd ~

module load GCC/12.2.0
module load Python/3.10.8
module load CUDA/12.2.0

srun python3 -m venv ~/vllm_take2

source ~/vllm_take2/bin/activate

srun python3 -m pip install --upgrade pip
cd ~/repos/hepthLlama/sulis/
srun python3 -m pip install -r vllm3_freeze.txt
srun python3 -m pip install --upgrade accelerate==0.34.2 
srun python3 -m pip install --upgrade flash-attn==2.6.3 --no-build-isolation

srun python3 -m torch.utils.collect_env
