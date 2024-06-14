import os
import argparse
import yaml
from dotenv import load_dotenv
import wandb
import huggingface_hub
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          Trainer,
                          )
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup  # needs transformers >= 4.40.0
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import math
import gc
from functools import partial


def tokenize_fn(batch, padding='max_length', max_length=720):
    """Tokenize a batch and pad each entry to max_length."""
    return tokenizer(batch['abstract'], padding=padding, max_length=max_length)


# def group_abstracts(examples):
#     """Concatenates the data and then divides into fixed-length chunks of size 1024."""
#     # Concatenate all texts.
#     block_size = 1024
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#     # customize this part to your needs.
#     if total_length >= block_size:
#         total_length = (total_length // block_size) * block_size
#     # Split by chunks of block_size.
#     result = {
#         k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result


def create_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples


def compute_optimization_steps(trainer):
    """Computes the number of optimization steps required to train the model.

    Essentially, copies the code to compute max_steps from the _inner_training_loop method of
    the Trainer class.
    """
    train_dataloader = trainer.get_train_dataloader()
    len_dataloader = len(train_dataloader)
    num_update_steps_per_epoch = len_dataloader // trainer.args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    max_steps = math.ceil(trainer.args.num_train_epochs * num_update_steps_per_epoch)
    return max_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="Filepath for config file", required=True)
    filepath = parser.parse_args().config

    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
    WANDB_TOKEN = os.getenv("WANDB_API_KEY")

    huggingface_hub.login(token=HF_TOKEN)
    wandb.login(key=WANDB_TOKEN)

    with open(filepath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # datasets get cached, the default cache directory is ~/.cache/huggingface/datasets
    # the default can be changed through the cache_dir parameter of load_dataset
    # ds = load_dataset(cfg['names_cfg']['dataset_name'])
    ds_train, ds_test, ds_valid = load_dataset(cfg['dataset_cfg']['dataset_name'],
                                               split=[f"train[{cfg['dataset_cfg']['dataset_percent']}]",
                                                      f"test[{cfg['dataset_cfg']['dataset_percent']}]",
                                                      f"validation[{cfg['dataset_cfg']['dataset_percent']}]"])
    ds = DatasetDict({"train": ds_train, "test": ds_test, "validation": ds_valid})
    #
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenised_ds = ds.map(partial(tokenize_fn, **cfg['tokenizer']),
                          batched=True,
                          remove_columns=ds['train'].column_names)
    lm_dataset = tokenised_ds.map(create_labels, batched=True)

    # see: https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
    # and: https://huggingface.co/docs/transformers/v4.40.1/en/tasks/language_modeling#preprocess
    # Data collator used for language modeling.
    # Inputs are dynamically padded to the maximum length of a batch if they are
    # not all the same length.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(**cfg['training_args_cfg'])

    quant_config = BitsAndBytesConfig(**cfg['bnb_cfg'])
    lora_config = LoraConfig(**cfg['lora_cfg'])

    foundation_model = AutoModelForCausalLM.from_pretrained(cfg['model_name'],
                                                            device_map="auto",
                                                            quantization_config=quant_config,
                                                            trust_remote_code=True,
                                                            attn_implementation=cfg['attn_implementation']
                                                            )
    model = prepare_model_for_kbit_training(foundation_model)
    model = get_peft_model(model, lora_config)

    # Instantiate a dummy trainer to allow us to compute num_training_steps
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=lm_dataset['train'],
                      eval_dataset=lm_dataset['test'],
                      data_collator=data_collator,
                      )

    num_training_steps = compute_optimization_steps(trainer)
    del trainer
    gc.collect()
    # now that we have the number of training steps we can set up the optimzer and learning rate schedule

    optimizer = torch.optim.AdamW(model.parameters(), **cfg['optim_cfg'])
    lr_schedule_cfg = cfg['lr_schedule_cfg']
    lr_schedule_cfg.update({'num_training_steps': num_training_steps})
    lr_schedule_cfg.update({'num_warmup_steps': round(0.1 * num_training_steps)})

    lr_schedule = get_cosine_with_min_lr_schedule_with_warmup(optimizer, **lr_schedule_cfg)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=lm_dataset['train'],
                      eval_dataset=lm_dataset['test'],
                      data_collator=data_collator,
                      optimizers=(optimizer, lr_schedule)
                      )

    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # following training, we push the fine-tuned model to Huggingface
    trainer.push_to_hub()
    wandb.finish()
