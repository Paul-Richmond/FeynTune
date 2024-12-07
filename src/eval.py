import sys
import logging
import json
import os
import gc
import torch
from dotenv import load_dotenv
import huggingface_hub
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForSeq2Seq,
                          BitsAndBytesConfig)
from datasets import Dataset, load_dataset

from utils.metrics import compute_perplexities, metric_perplexity

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)

# Create a logger and set level to INFO.
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def load_json_file(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logger.error(f'File {filename} not found')
    except json.JSONDecodeError:
        logger.error(f"Error: '{filename}' is not a valid JSON file.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
    return None


def save_dict_to_json(data, directory, filename):
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    # Check filename has extension and add if not
    if not filename.endswith('.json'):
        filename += '.json'
    # Construct the full file path
    file_path = os.path.join(directory, filename)

    # Write the dictionary to a JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    logger.info(f"Dictionary saved to {file_path}")


if __name__ == "__main__":
    ds_repo = ''
    ds_col = ''
    model_repo = ''

    ds = load_dataset(ds_repo, split='test')

    model = AutoModelForCausalLM.from_pretrained(model_repo,
                                                 device_map='auto',
                                                 trust_remote_code=True,
                                                 attn_implementation='flash_attention_2',
                                                 torch_dtype='bfloat16')
    tokenizer = AutoTokenizer.from_pretrained(model_repo)

    tokenised_ds = ds.map(lambda examples: tokenizer(examples[ds_col],
                                                     padding='do_not_pad',
                                                     truncation='do_not_truncate'),
                          batched=True,
                          remove_columns=ds.column_names,
                          desc="Tokenizing")
    tokenised_ds = tokenised_ds.map(lambda example: {"labels": example["input_ids"]},
                                    batched=True,
                                    desc="Adding labels")

    training_args = TrainingArguments(output_dir="tmp_trainer",
                                      bf16=True,
                                      per_device_eval_batch_size=16,
                                      report_to=None,
                                      logging_steps=1)

    # why pad_to_multiple_of=8? see https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#opt-tensor-cores
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           pad_to_multiple_of=8)  # dynamically pads a batch to all have same tensor shapes
    callbacks = None
    optimizer = None

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=None,
                      eval_dataset=tokenised_ds,  # will evaluate all datasets within tokenised_ds
                      data_collator=data_collator,
                      tokenizer=tokenizer,
                      optimizers=(optimizer, None),
                      callbacks=callbacks,
                      compute_metrics=metric_perplexity,
                      preprocess_logits_for_metrics=compute_perplexities
                      )

    res = trainer.evaluate()
    print(res)
    # del trainer
    # torch.cuda.empty_cache()
    # gc.collect()
