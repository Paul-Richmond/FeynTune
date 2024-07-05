import os
from dotenv import load_dotenv
import wandb
import huggingface_hub
from datasets import load_dataset
from transformers import (DataCollatorForLanguageModeling,
                          TrainingArguments,
                          Trainer,
                          )
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.instantiators import (load_automodelforcausallm,
                                 load_optimizer,
                                 load_tokenizer,
                                 instantiate_callbacks)


def create_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    wandb_token = os.getenv("WANDB_API_KEY")

    huggingface_hub.login(token=hf_token)
    wandb.login(key=wandb_token)

    # datasets get cached, the default cache directory is ~/.cache/huggingface/datasets
    # the default can be changed through the cache_dir parameter of load_dataset
    ds = load_dataset(cfg.dataset.name)  # we assume there always exists a dictionary key called 'train'

    tokenizer = load_tokenizer(cfg.tokenizer)
    tokenised_ds = ds.map(lambda examples: tokenizer(examples["abstract"], **cfg.tokenizer.tokenizer_args),
                          batched=True,
                          remove_columns=ds['train'].column_names)

    lm_dataset = tokenised_ds.map(create_labels, batched=True)
    train_dataset = lm_dataset.get('train')
    eval_dataset = lm_dataset.get('eval', None)

    model = load_automodelforcausallm(cfg.mode)
    # we need to use OmegaConf.to_container to avoid json serialization errors when saving model
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.training, resolve=True))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    callbacks = instantiate_callbacks(cfg.callbacks)
    optimizer = load_optimizer(cfg.optimizer, model)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      data_collator=data_collator,
                      optimizers=(optimizer, None),
                      callbacks=callbacks,
                      )

    trainer.train()
    trainer.push_to_hub()
    huggingface_hub.logout()
    wandb.finish()


if __name__ == "__main__":
    main()
