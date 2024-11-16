import os

from dotenv import load_dotenv
import huggingface_hub
import hydra
from omegaconf import DictConfig
from transformers import DataCollatorForSeq2Seq
import wandb

from utils.instantiators import (load_automodelforcausallm,
                                 load_optimizer,
                                 load_tokenizer,
                                 instantiate_callbacks,
                                 instantiate_training)
from utils.io import load_dataset_splits
from utils.metrics import compute_perplexities, metric_perplexity

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
wandb_token = os.getenv("WANDB_API_KEY")

huggingface_hub.login(token=hf_token)
wandb.login(key=wandb_token)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ds = load_dataset_splits(cfg.dataset)
    tokenizer = load_tokenizer(cfg.tokenizer)
    tokenised_ds = ds.map(lambda examples: tokenizer(examples[cfg.dataset.column_to_tokenize],
                                                     **cfg.tokenizer.tokenizer_args),
                          batched=True,
                          remove_columns=ds['train'].column_names,
                          desc="Tokenizing")
    tokenised_ds = tokenised_ds.map(lambda example: {"labels": example["input_ids"]},
                                    batched=True,
                                    desc="Adding labels")

    model = load_automodelforcausallm(cfg.model)
    trainer_cls, training_args = instantiate_training(cfg.training)
    # why pad_to_multiple_of=8? see https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#opt-tensor-cores
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)  # dynamically pads a batch to all have same tensor shapes
    callbacks = instantiate_callbacks(cfg.callbacks)
    optimizer = load_optimizer(cfg.optimizer, model)

    trainer = trainer_cls(model=model,
                          args=training_args,
                          train_dataset=tokenised_ds.get('train'),
                          eval_dataset=tokenised_ds,  # will evaluate all datasets within tokenised_ds
                          data_collator=data_collator,
                          tokenizer=tokenizer,
                          optimizers=(optimizer, None),
                          callbacks=callbacks,
                          compute_metrics=metric_perplexity,
                          preprocess_logits_for_metrics=compute_perplexities
                          )

    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    trainer.push_to_hub()
    huggingface_hub.logout()
    wandb.finish()


if __name__ == "__main__":
    main()
