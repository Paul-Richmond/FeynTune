#!/bin/bash
#$ -l h_rt=1:0:0
#$ -l h_vmem=11G
#$ -pe smp 16
#$ -l gpu=2
#$ -cwd
#$ -j y
#$ -N finetune_gpu2

cd ~

module load anaconda3
module load cuda/11.8

mamba activate llm

cd $HOME/hepthLlama

python3 finetune.py -c "./configs/config.yaml"