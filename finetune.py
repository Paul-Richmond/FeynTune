import os
from dotenv import load_dotenv
import wandb
import huggingface_hub
from datasets import load_dataset
from transformers import (AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          Trainer,
                          )
import transformers.optimization
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import math
import gc
import hydra
from omegaconf import DictConfig, OmegaConf


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


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
    WANDB_TOKEN = os.getenv("WANDB_API_KEY")

    huggingface_hub.login(token=HF_TOKEN)
    wandb.login(key=WANDB_TOKEN)

    # datasets get cached, the default cache directory is ~/.cache/huggingface/datasets
    # the default can be changed through the cache_dir parameter of load_dataset
    ds = load_dataset(cfg.dataset.name)  # we assume there always exists a dictionary key called 'train'

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenised_ds = ds.map(lambda examples: tokenizer(examples["abstract"], **cfg.tokenizer.tokenizer_args),
                          batched=True,
                          remove_columns=ds['train'].column_names)

    lm_dataset = tokenised_ds.map(create_labels, batched=True)
    train_dataset = lm_dataset.get('train')
    eval_dataset = lm_dataset.get('eval', None)

    # see: https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
    # and: https://huggingface.co/docs/transformers/v4.40.1/en/tasks/language_modeling#preprocess
    # Data collator used for language modeling.
    # Inputs are dynamically padded to the maximum length of a batch if they are
    # not all the same length.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(**cfg.model.training_args_cfg)

    quant_config = BitsAndBytesConfig(**cfg.model.bnb_cfg)
    lora_config = LoraConfig(**cfg.model.lora_cfg)

    foundation_model = AutoModelForCausalLM.from_pretrained(cfg.model.name,
                                                            device_map="auto",
                                                            quantization_config=quant_config,
                                                            trust_remote_code=True,
                                                            attn_implementation=cfg.model.attn_implementation
                                                            )
    model = prepare_model_for_kbit_training(foundation_model)
    model = get_peft_model(model, lora_config)

    # Instantiate a dummy trainer to allow us to compute num_training_steps
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      data_collator=data_collator,
                      )

    num_training_steps = compute_optimization_steps(trainer)
    del trainer
    gc.collect()
    # now that we have the number of training steps we can set up the optimzer and learning rate schedule

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters(), **cfg.optimizer)
    lr_schedule_cfg = cfg.lr.lr_schedule_args
    lr_schedule_cfg.update({'num_training_steps': num_training_steps})
    lr_schedule_cfg.update({'num_warmup_steps': round(0.1 * num_training_steps)})

    lr_schedule = hydra.utils.instantiate(cfg.lr, optimizer, **lr_schedule_cfg)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
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


if __name__ == "__main__":
    main()
