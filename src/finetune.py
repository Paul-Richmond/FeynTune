import os
from dotenv import load_dotenv
import wandb
import huggingface_hub
from transformers import DataCollatorForLanguageModeling, TrainingArguments
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.instantiators import (load_automodelforcausallm,
                                 load_optimizer,
                                 load_tokenizer,
                                 instantiate_callbacks,
                                 load_dataset_splits)
from utils.trainers import PerplexityTrainer
from utils.metrics import compute_perplexities, metric_perplexity

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
wandb_token = os.getenv("WANDB_API_KEY")

huggingface_hub.login(token=hf_token)
wandb.login(key=wandb_token)


@hydra.main(version_base=None, config_path="../configs", config_name="apoc")
def main(cfg: DictConfig) -> None:
    ds = load_dataset_splits(cfg.dataset)
    tokenizer = load_tokenizer(cfg.tokenizer)
    tokenised_ds = ds.map(lambda examples: tokenizer(examples[cfg.dataset.column_to_tokenize],
                                                     **cfg.tokenizer.tokenizer_args),
                          batched=True,
                          remove_columns=ds['train'].column_names)
    tokenised_ds = tokenised_ds.map(lambda example: {"labels": example["input_ids"]}, batched=True)

    model = load_automodelforcausallm(cfg.model)
    # we need to use OmegaConf.to_container to avoid json serialization errors when saving model
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.training, resolve=True))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    callbacks = instantiate_callbacks(cfg.callbacks)
    optimizer = load_optimizer(cfg.optimizer, model)

    trainer = PerplexityTrainer(model=model,
                                args=training_args,
                                train_dataset=tokenised_ds.get('train'),
                                eval_dataset=tokenised_ds.get('eval', None),
                                data_collator=data_collator,
                                tokenizer=tokenizer,
                                optimizers=(optimizer, None),
                                callbacks=callbacks,
                                compute_metrics=metric_perplexity,
                                preprocess_logits_for_metrics=compute_perplexities
                                )

    trainer.train()
    trainer.push_to_hub()
    huggingface_hub.logout()
    wandb.finish()


if __name__ == "__main__":
    main()
