#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l h_vmem=7.5G
#$ -pe smp 12
#$ -l gpu=1
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

# We remove the train dataset from the configuration leaving only the test dataset
# We also provide the wand_runpath where the completed abstracts are to be logged
python3 src/inference.py ~dataset.splits.train=train inference.wandb_runpath=llms-for-hepth/huggingface/REPLACE
