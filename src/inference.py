import os

from dotenv import load_dotenv
import huggingface_hub
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import wandb

from utils.instantiators import (load_automodelforcausallm,
                                 load_optimizer,
                                 load_tokenizer,
                                 instantiate_callbacks,
                                 load_dataset_splits,
                                 instantiate_training)

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
wandb_token = os.getenv("WANDB_API_KEY")

huggingface_hub.login(token=hf_token)
wandb.login(key=wandb_token)


class AbstractCompleter:
    def __init__(self, model, tokenizer, dataset, batch_size=None, generation_config=None):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token
        self.dataset = dataset

        self.batch_size = 16 if batch_size is None else batch_size
        self.label = 'abstract'
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
    ds = load_dataset_splits(cfg.dataset)
    model = load_automodelforcausallm(cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_cfg.name, padding_side="left")
    generation_config = cfg.generation_config
    batch_size = cfg.batch_size

    completer = AbstractCompleter(model=model,
                                  tokenizer=tokenizer,
                                  dataset=ds['test'],
                                  batch_size=batch_size,
                                  generation_config=generation_config)

    ds_with_completions = completer.get_predictions()


if __name__ == "__main__":
    main()
