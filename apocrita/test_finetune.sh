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

cd $HOME/hepthLlama

python3 src/finetune.py training.run_name="CHANGE_ME"