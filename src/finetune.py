import os

from dotenv import load_dotenv
import huggingface_hub
import hydra
from omegaconf import DictConfig
from transformers import DataCollatorWithPadding
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


def group_abstracts(examples, block_size=512):
    """Concatenates the data and then divides into fixed-length chunks of size block_size."""
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ds = load_dataset_splits(cfg.dataset)
    tokenizer = load_tokenizer(cfg.tokenizer)
    tokenised_ds = ds.map(lambda examples: tokenizer(examples[cfg.dataset.column_to_tokenize],
                                                     **cfg.tokenizer.tokenizer_args),
                          batched=True,
                          remove_columns=ds['train'].column_names,
                          desc="Tokenizing")
    if cfg.tokenizer.post_processing.get('add_labels'):
        tokenised_ds = tokenised_ds.map(lambda example: {"labels": example["input_ids"]},
                                        batched=True,
                                        desc="Adding labels")
    if cfg.tokenizer.post_processing.get('concatenate_and_split_length') is not None:
        tokenised_ds = tokenised_ds.map(group_abstracts,
                                        fn_kwargs={'block_size': cfg.tokenizer.post_processing.concatenate_and_split_length},
                                        batched=True,
                                        desc="Applying group abstracts")

    model = load_automodelforcausallm(cfg.model)
    trainer_cls, training_args = instantiate_training(cfg.training)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  #pads a batch to all have same sequence length
    callbacks = instantiate_callbacks(cfg.callbacks)
    optimizer = load_optimizer(cfg.optimizer, model)

    trainer = trainer_cls(model=model,
                          args=training_args,
                          train_dataset=tokenised_ds.get('train'),
                          eval_dataset=tokenised_ds.get('test', None),
                          data_collator=data_collator,
                          tokenizer=tokenizer,
                          optimizers=(optimizer, None),
                          callbacks=callbacks,
                          compute_metrics=None,
                          preprocess_logits_for_metrics=None
                          )

    trainer.train()
    trainer.push_to_hub()
    huggingface_hub.logout()
    wandb.finish()


if __name__ == "__main__":
    main()
