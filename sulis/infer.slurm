#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su008-apw281

# Datasets are
# LLMsForHepth/hep-th_primary
# LLMsForHepth/hep-ph_gr-qc_primary
# LLMsForHepth/hep-th_hep-ph_gr-qc_primary

# Base model is always
# meta-llama/Meta-Llama-3.1-8B

# Finetuned LoRA adapters are
# LLMsForHepth/s1-L-3.1-8B-base (trained on hep-th_primary)
# LLMsForHepth/s2-L-3.1-8B-base (trained on hep-ph_gr-qc_primary)
# LLMsForHepth/s3-L-3.1-8B-base (trained on hep-th_hep-ph_gr-qc_primary)

set -x

module load CUDA/11.8.0

cd ~/hepthLlama-main/src/

# ---------- Run inference on base LLama-3.1-8B models ----------
# Run meta-llama/Meta-Llama-3.1-8B on LLMsForHepth/hep-th_primary test set
srun python3 infer.py ~dataset.splits.train=train \
~inference.lora_adapter_name \
inference.output_dname=??? \
inference.output_fname=???

# Run meta-llama/Meta-Llama-3.1-8B on LLMsForHepth/hep-ph_gr-qc_primary test set
srun python3 infer.py dataset.name=LLMsForHepth/hep-ph_gr-qc_primary \
~dataset.splits.train=train \
~inference.lora_adapter_name \
inference.output_dname=??? \
inference.output_fname=???

# Run meta-llama/Meta-Llama-3.1-8B on LLMsForHepth/hep-ph_gr-qc_primary test set
srun python3 infer.py dataset.name=LLMsForHepth/hep-th_hep-ph_gr-qc_primary \
~dataset.splits.train=train \
~inference.lora_adapter_name \
inference.output_dname=??? \
inference.output_fname=???

# ---------- Run inference on finetuned models ----------
# Run finetuned LLMsForHepth/s1-L-3.1-8B-base on LLMsForHepth/hep-th_primary test set
#srun python3 infer.py ~dataset.splits.train=train \
#inference.output_dname=??? \
#inference.output_fname=???

# Run finetuned LLMsForHepth/s2-L-3.1-8B-base on LLMsForHepth/hep-ph_gr-qc_primary test set
srun python3 infer.py dataset.name=LLMsForHepth/hep-ph_gr-qc_primary \
~dataset.splits.train=train \
inference.lora_adapter_name=LLMsForHepth/s2-L-3.1-8B-base \
inference.output_dname=??? \
inference.output_fname=???

# Run finetuned LLMsForHepth/s3-L-3.1-8B-base on LLMsForHepth/hep-ph_gr-qc_primary test set
srun python3 infer.py dataset.name=LLMsForHepth/hep-th_hep-ph_gr-qc_primary \
~dataset.splits.train=train \
inference.lora_adapter_name=LLMsForHepth/s3-L-3.1-8B-base \
inference.output_dname=??? \
inference.output_fname=???