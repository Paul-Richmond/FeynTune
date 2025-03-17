import logging
import os

import huggingface_hub
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          DataCollatorForSeq2Seq,
                          set_seed)

from utils.io import load_dataset_splits
from utils.processing import split_abstracts

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")

huggingface_hub.login(token=hf_token)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

set_seed(42)

gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in Gb
initial_batch_size = 56

if gpu_memory > 42.0:  # probably on 80Gb
    batch_size = initial_batch_size
elif gpu_memory > 35.0:  # probably on 40Gb
    batch_size = int(initial_batch_size / 2)
else:
    batch_size = int(initial_batch_size / 4)

model_cfg = {"attn_implementation": "sdpa",  # "flash_attention_2" doesn't work with torch.compile
             "device_map": "auto",
             "torch_dtype": torch.float16
             }
dataloader_params = {"batch_size": batch_size,
                     "num_workers": 0,
                     "pin_memory": True,
                     "persistent_workers": False,
                     }


def parse_completions(abstracts, completions):
    return [completion[len(abstract):] for abstract, completion in zip(abstracts, completions)]


@hydra.main(version_base=None, config_path="../configs", config_name="default_infer")
def main(cfg: DictConfig) -> None:
    ds = load_dataset_splits(cfg.dataset)  # expect DatasetDict object with a single key
    ds = ds[list(ds)[0]]  # extract the single Dataset object from the DatasetDict object
    ds = ds.map(split_abstracts,
                batched=False,
                desc='Splitting abstracts')

    tokenizer = AutoTokenizer.from_pretrained(cfg.inference.model_name)
    tokenizer.padding_side = "left"  # left pad for inference
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenised_ds = ds.map(lambda examples: tokenizer(examples['prompt'],
                                                     padding='do_not_pad',
                                                     truncation='do_not_truncate',
                                                     add_special_tokens=False, ),
                          batched=True,
                          desc="Tokenizing")
    tokenised_ds = tokenised_ds.map(lambda examples: {"labels": examples["input_ids"]},
                                    batched=True,
                                    desc="Adding labels")

    tokenised_ds = tokenised_ds.remove_columns(list(set(tokenised_ds.column_names)
                                                    - {'input_ids', 'attention_mask', 'labels'}))

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)
    dataloader_params['collate_fn'] = data_collator
    dataloader = DataLoader(tokenised_ds, **dataloader_params)

    model = AutoModelForCausalLM.from_pretrained(cfg.inference.model_name, **model_cfg)
    model.generation_config.cache_implementation = "static"
    model = torch.compile(model, mode='reduce-overhead', fullgraph=True)

    generation_cfg = OmegaConf.to_container(cfg.inference.generation_cfg, resolve=True)
    generation_cfg['pad_token_id'] = tokenizer.pad_token_id

    completions = []
    model.eval()
    logger.info("--------------------------")
    logger.info(f"Generating using model {cfg.inference.model_name}")
    logger.info("--------------------------")
    for idx, batch in enumerate(dataloader):
        logger.info(f"Generating batch {idx}")
        with torch.inference_mode():
            batch.to(model.device)
            outputs_tok = model.generate(**batch, **generation_cfg)
            batch_predictions = tokenizer.batch_decode(outputs_tok, skip_special_tokens=True)
            completions.extend(batch_predictions)

    y_pred = parse_completions(ds['prompt'], completions)

    ds = ds.add_column(name=f"comp_{cfg.inference.model_name.split('/')[-1]}", column=completions)
    ds = ds.add_column(name=f"preds_{cfg.inference.model_name.split('/')[-1]}", column=y_pred)
    ds.push_to_hub(cfg.inference.repo_name, split='test')
    huggingface_hub.logout()


if __name__ == '__main__':
    main()
