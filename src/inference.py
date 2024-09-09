import os

from dotenv import load_dotenv
import huggingface_hub
import hydra
import logging
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf
from transformers import AutoTokenizer
import wandb

from utils.callbacks import AbstractCompleter
from utils.instantiators import load_automodelforcausallm, load_dataset_splits

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
wandb_token = os.getenv("WANDB_API_KEY")

huggingface_hub.login(token=hf_token)
wandb.login(key=wandb_token)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="default_infer")
def main(cfg: DictConfig) -> None:

    try:
        wandb_runpath = cfg.inference.wandb_runpath
        if len(wandb_runpath.split('/')) != 3:
            logger.error(f"wandb_runpath should be in the form '<entity>/<project>/<run_id>' but \n"
                         f"wandb_runpath='{wandb_runpath}' was set instead.")
            raise ValueError
        else:
            entity, project, run_id = wandb_runpath.split('/')
            run = wandb.init(entity=entity, project=project, id=run_id, resume="allow")
    except MissingMandatoryValue:
        logger.info(f"You have not set a value for wandb_runpath, initializing using wandb.init() instead.")
        run = wandb.init()

    ds = load_dataset_splits(cfg.dataset)  # expect DatasetDict object with a single key
    ds = ds[list(ds)[0]]  # extract the single Dataset object from the DatasetDict object
    ds = ds.remove_columns(list(set(ds.column_names) - {'id', 'abstract'}))  # keep only 'id' and 'abstract' columns
    model = load_automodelforcausallm(cfg.inference)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.inference.model_cfg.name, padding_side="left")
    if 'generation_cfg' in cfg.inference.keys():
        generation_config = OmegaConf.to_container(cfg.inference.generation_cfg, resolve=True)
    else:
        generation_config = None
    batch_size = cfg.inference.batch_size

    completer = AbstractCompleter(model=model,
                                  tokenizer=tokenizer,
                                  dataset=ds,
                                  batch_size=batch_size,
                                  generation_config=generation_config)

    ds_with_completions = completer.get_predictions()
    inference_table = wandb.Table(dataframe=ds_with_completions.to_pandas())
    gen_cfg_table = wandb.Table(columns=list(completer.generation_config.keys()),
                                data=list(completer.generation_config.values()))
    run.log({"inference": inference_table, "inference_gen_cfg": gen_cfg_table})


if __name__ == "__main__":
    main()
