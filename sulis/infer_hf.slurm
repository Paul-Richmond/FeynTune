#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --account=su008-apw822

# Datasets are
# LLMsForHepth/hep-th_primary
# LLMsForHepth/hep-ph_gr-qc_primary
# LLMsForHepth/hep-th_hep-ph_gr-qc_primary_v3

# Base model is
# meta-llama/Meta-Llama-3.1-8B

# Finetuned LoRA adapters are
# LLMsForHepth/s1-L-3.1-8B-base (trained on hep-th_primary)
# LLMsForHepth/s2-L-3.1-8B-base (trained on hep-ph_gr-qc_primary)
# LLMsForHepth/s3-L-3.1-8B-base_v3 (trained on hep-th_hep-ph_gr-qc_primary_v3)

#set -x  #<-- uncomment for verbose output

module purge
module load GCC/12.2.0 CUDA/12.2.0 Python/3.10.8

source ~/vllm_take2/bin/activate

cd ~/repos/hepthLlama/src/

# ---------- Run inference on finetuned model ----------
# Run meta-llama/Meta-Llama-3.1-8B on LLMsForHepth/hep-th_primary test set
srun python3 infer_hf.py ~dataset.splits.train=train \
inference=infer_hf \
inference.model_name=LLMsForHepth/s1-L-3.1-8B-base \
inference.generation_cfg.temperature=0.7 \
inference.repo_name=LLMsForHepth/infer_hep_th
