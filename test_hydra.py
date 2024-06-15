import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import transformers
from hydra.utils import instantiate
from dotenv import load_dotenv
import wandb
import huggingface_hub
import os

@hydra.main(version_base=None, config_path="configs", config_name="default")
def my_app(cfg: DictConfig) -> None:
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
    WANDB_TOKEN = os.getenv("WANDB_API_KEY")

    huggingface_hub.login(token=HF_TOKEN)
    wandb.login(key=WANDB_TOKEN)

    print(OmegaConf.to_yaml(cfg))




if __name__ == "__main__":
    my_app()
