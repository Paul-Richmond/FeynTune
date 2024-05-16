module load python/3.11.6
virtualenv HepthLlama
source HepthLlama/bin/activate

python3 -m pip install huggingface_hub datasets transformers torch wandb bitsandbytes accelerate peft python-dotenv
