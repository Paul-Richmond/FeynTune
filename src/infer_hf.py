import logging
import os
import gc

import huggingface_hub
import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          DataCollatorForSeq2Seq)

from utils.io import load_dataset_splits, save_dict_to_json
from utils.processing import split_abstracts

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
wandb_token = os.getenv("WANDB_API_KEY")

huggingface_hub.login(token=hf_token)
wandb.login(key=wandb_token)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in Gb
initial_batch_size = 192  # 'LLMsForHepth/hep-th_primary' uses 73,070 Mb on H100 with batch size 192

if gpu_memory > 42.0:  # probably on 80Gb
    batch_size = initial_batch_size
elif gpu_memory > 35.0:  # probably on 40Gb
    batch_size = int(initial_batch_size / 2)
else:
    batch_size = int(initial_batch_size / 4)

dataset_name = 'LLMsForHepth/hep-th_primary'
model_name = 'meta-llama/Meta-Llama-3.1-8B'

model_cfg = {"attn_implementation": "sdpa",  # "flash_attention_2" doesn't work with torch.compile
             "device_map": "auto",
             "torch_dtype": torch.float16
             }
dataloader_params = {"batch_size": batch_size,
                     "num_workers": 0,
                     "pin_memory": True,
                     "persistent_workers": False,
                     }
generation_config = {'max_new_tokens': 1024,
                     'temperature': 0.3,
                     'max_time': 30,
                     'top_p': 0.1,
                     'do_sample': True,
                     }


def main():
    ds = load_dataset(dataset_name, split='test')
    ds = ds.map(split_abstracts,
                batched=False,
                desc='Splitting abstracts')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # left pad for inference
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    generation_config['pad_token_id'] = tokenizer.pad_token_id
    tokenised_ds = ds.map(lambda examples: tokenizer(examples['prompt'],
                                                     padding='do_not_pad',
                                                     truncation='do_not_truncate',
                                                     add_special_tokens=False, ),
                          batched=True,
                          desc="Tokenizing")
    tokenised_ds = tokenised_ds.map(lambda examples: {"labels": examples["input_ids"]},
                                    batched=True,
                                    desc="Adding labels")

    tokenised_ds = tokenised_ds.map(lambda example: {"token_num": len(example["input_ids"])},
                                    batched=False,
                                    desc="Getting token counts")
    tokenised_ds = tokenised_ds.sort(column_names='token_num', reverse=True)
    sorted_ids = tokenised_ds['id']
    sorted_prompts = tokenised_ds['prompt']
    tokenised_ds = tokenised_ds.remove_columns(list(set(tokenised_ds.column_names)
                                                    - {'input_ids', 'attention_mask', 'labels'}))

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)
    dataloader_params['collate_fn'] = data_collator
    dataloader = DataLoader(tokenised_ds, **dataloader_params)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_cfg)
    model.generation_config.cache_implementation = "static"
    model = torch.compile(model, mode='reduce-overhead', fullgraph=True)

    output_dict = {'model': model_name.split('/')[-1],
                   'dataset': dataset_name.split('/')[-1],
                   'ids': sorted_ids,
                   'prompts': sorted_prompts,
                   'generation_config': generation_config,
                   }
    filename = f"hf_{output_dict['model']}_{output_dict['dataset']}"
    predictions = []
    model.eval()
    for idx, batch in enumerate(dataloader):
        logger.info(f"Generating batch {idx}")
        with torch.no_grad():
            batch.to(model.device)
            outputs_tok = model.generate(**batch, **generation_config)
            batch_predictions = tokenizer.batch_decode(outputs_tok, skip_special_tokens=True)
            predictions.extend(batch_predictions)
            output_dict['predictions'] = predictions
            output_dict['batch'] = idx

        if idx % 10 == 0:  # save every 10 batches
            save_dict_to_json(data=output_dict,
                              directory='infer_hf_outputs',
                              filename=filename)

    output_dict['batch'] = 'end of inference'
    save_dict_to_json(data=output_dict,
                      directory='infer_hf_outputs',
                      filename=filename)


if __name__ == '__main__':
    main()
