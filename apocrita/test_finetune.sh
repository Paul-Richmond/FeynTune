#!/bin/bash
#$ -l h_rt=1:0:0
#$ -l h_vmem=7.5G
#$ -pe smp 24
#$ -l gpu=2
#$ -l gpu_type=ampere
#$ -cwd
#$ -j y

cd ~

module load anaconda3
module load cuda/11.8

mamba activate llm

# change where the huggingface dataset and model are cached to
# see https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
export HF_HOME=/data/scratch/$USER/.cache/huggingface/hub

cd $HOME/hepthLlama

python3 src/finetune.py training.run_name="CHANGE_ME"