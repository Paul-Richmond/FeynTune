import os
import argparse
import yaml
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
                          get_cosine_schedule_with_warmup,
                          AdamW)
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


def num_gpus():
    """Get the number of available GPUs."""
    return torch.cuda.device_count()


def _get_steps(train_dataset, training_args):
    """Compute total number of optimization steps during training."""
    num_devices = num_gpus()
    bs = training_args.per_device_eval_batch_size * num_devices
    grad_acc = training_args.gradient_accumulation_steps
    try:
        num_of_batches = len(train_dataset) / (bs * grad_acc)
    except ZeroDivisionError:
        num_of_batches = len(train_dataset)
    epochs = training_args.num_train_epochs
    return int(num_of_batches * epochs)


def update_lr_schedule_cfg(cfg, train_dataset, training_args):
    """Update lr_schedule_cfg with computed values for
    num_warmup_steps and num_training_steps."""
    max_steps = _get_steps(train_dataset, training_args)
    w_steps = int(max_steps * cfg['lr_schedule_cfg']['warmup_ratio'])
    cfg['lr_schedule_cfg'] = {"num_warmup_steps": w_steps,
                              "num_training_steps": max_steps,
                              "num_cycles": cfg['lr_schedule_cfg']['num_cycles']
                              }


def tokenize_fn(batch):
    return tokenizer(batch['abstract'])


def group_abstracts(examples):
    """Concatenates the data and then divides into fixed-length chunks of size 512."""
    # Concatenate all texts.
    block_size = 512
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
    result["labels"] = result["input_ids"].copy()
    return result


def compute_metrics(eval_pred):
    """Computes the perplexity metric"""
    logits = torch.from_numpy(eval_pred.predictions)
    labels = torch.from_numpy(eval_pred.label_ids)
    loss = F.cross_entropy(logits, labels)
    return {'perplexity': torch.exp(loss), 'calculated_loss': loss}


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        PR: Subclassed to compute training perplexity.

        How the loss is computed by Trainer. By default, all models return the loss in
        the first element.

        Subclass and override for custom behavior.
        """

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # -------------
        # PR: Added following lines to log perplexity
        perp_loss = loss.detach()
        perplexity = torch.exp(perp_loss).item()  # return as number for json serialising
        self.log({"perplexity": perplexity})
        # -------------

        return (loss, outputs) if return_outputs else loss




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
    ds = load_dataset(cfg['names_cfg']['dataset_name'])
    #
    tokenizer = AutoTokenizer.from_pretrained(cfg['names_cfg']['tokenizer_name'])
    tokenised_ds = ds.map(tokenize_fn,
                          batched=True,
                          remove_columns=ds['train'].column_names)
    # concatenate the tokenised data and then divide it into chunks
    lm_dataset = tokenised_ds.map(group_abstracts, batched=True)

    # see: https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling
    tokenizer.pad_token = tokenizer.eos_token
    # Data collator used for language modeling.
    # Inputs are dynamically padded to the maximum length of a batch if they are
    # not all the same length.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(**cfg['training_args_cfg'])
    update_lr_schedule_cfg(cfg, lm_dataset['train'], training_args)

    quant_config = BitsAndBytesConfig(**cfg['bnb_cfg'])
    lora_config = LoraConfig(**cfg['lora_cfg'])

    foundation_model = AutoModelForCausalLM.from_pretrained(cfg['names_cfg']['model_name'],
                                                            device_map="auto",
                                                            quantization_config=quant_config,
                                                            trust_remote_code=True,
                                                            )

    model = get_peft_model(foundation_model, lora_config)

    optimizer = AdamW(model.parameters(), **cfg['optim_cfg'])
    lr_schedule = get_cosine_schedule_with_warmup(optimizer, **cfg['lr_schedule_cfg'])

    trainer = CustomTrainer(model=model,
                            args=training_args,
                            train_dataset=lm_dataset['train'],
                            eval_dataset=lm_dataset['test'],
                            data_collator=data_collator,
                            compute_metrics=compute_metrics,
                            optimizers=(optimizer, lr_schedule)
                            )

    trainer.train()
    # following training, we push the fine-tuned model to Huggingface
    trainer.push_to_hub()
    wandb.finish()
