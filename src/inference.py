import os

from dotenv import load_dotenv
import huggingface_hub
import hydra
import logging
from omegaconf import DictConfig, MissingMandatoryValue
from transformers import AutoTokenizer
import wandb

from utils.instantiators import load_automodelforcausallm, load_dataset_splits

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
wandb_token = os.getenv("WANDB_API_KEY")

huggingface_hub.login(token=hf_token)
wandb.login(key=wandb_token)

logger = logging.getLogger(__name__)


class AbstractCompleter:
    def __init__(self, model, tokenizer, dataset, batch_size=None, generation_config=None):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token
        self.dataset = dataset

        self.batch_size = 16 if batch_size is None else batch_size
        if generation_config is None:
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.1,
                "repetition_penalty": 1.18,
                "top_k": 40,
                "do_sample": True,
                "max_new_tokens": 1024,
                "pad_token_id": self.tokenizer.pad_token_id,
                "max_time": 30
            }
        else:
            self.generation_config = generation_config
        self.label = 'abstract'

    def _predict(self, batch):
        texts = batch['prompt']
        model_inputs = self.tokenizer(texts,
                                      padding='longest',
                                      truncation=True,
                                      max_length=self.tokenizer.model_max_length,
                                      pad_to_multiple_of=8,
                                      add_special_tokens=False,
                                      return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**model_inputs, **self.generation_config)
        batch["predictions"] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return batch

    def get_predictions(self):
        self.dataset = self.dataset.map(lambda example: {'prompt': example[self.label][:len(example[self.label]) // 2]},
                                        desc='Generating prompts')
        self.dataset = self.dataset.map(self._predict,
                                        batched=True,
                                        batch_size=self.batch_size,
                                        desc='Generating model output')
        return self.dataset


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

    ds = load_dataset_splits(cfg.dataset)
    model = load_automodelforcausallm(cfg.inference)
    tokenizer = AutoTokenizer.from_pretrained(cfg.inference.model_cfg.name, padding_side="left")
    generation_config = cfg.inference.generation_config
    batch_size = cfg.inference.batch_size

    completer = AbstractCompleter(model=model,
                                  tokenizer=tokenizer,
                                  dataset=ds['test'],
                                  batch_size=batch_size,
                                  generation_config=generation_config)

    ds_with_completions = completer.get_predictions()
    table = wandb.Table(dataframe=ds_with_completions.to_pandas())
    run.log({"predictions": table})


if __name__ == "__main__":
    main()
