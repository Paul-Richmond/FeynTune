#!/bin/bash
#$ -l h_rt=1:0:0
#$ -l h_vmem=11G
#$ -pe smp 8
#$ -l gpu=1
#$ -cwd
#$ -j y
#$ -N create_env

cd ~

module load anaconda3
module load cuda/11.8

mamba create -y -n llm python=3.8
mamba env update -n llm -f environment.yml
mamba activate llm

python3 -m pip install --upgrade pip
python3 -m pip install huggingface_hub datasets wandb bitsandbytes accelerate peft python-dotenv flash-attn

export BNB_CUDA_VERSION=118
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/llm/lib
echo $LD_LIBRARY_PATH

python3 -m bitsandbytes