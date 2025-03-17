import argparse
import logging
import os
import gc
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional, List

from utils.io import save_dict_to_json
from utils.metrics import compute_perplexities

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)

# Create a logger and set level to INFO.
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

BASE_MODEL_REPO = 'meta-llama/Meta-Llama-3.1-8B'

UPLOAD_REPO = 'LLMsForHepth/hep-th_perplexities'

ADAPTER_TO_REPO = {'s1': 'LLMsForHepth/s1-L-3.1-8B-base',
                   's2': 'LLMsForHepth/s2-L-3.1-8B-base',
                   's3': 'LLMsForHepth/s3-L-3.1-8B-base_v3',
                   's4': 'LLMsForHepth/s4-L-3.1-8B-base',
                   's5': 'LLMsForHepth/s5-L-3.1-8B-base',
                   's6': 'LLMsForHepth/s6-L-3.1-8B-base',
                   's7': 'LLMsForHepth/s7-L-3.1-8B-base',
                   's8': 'LLMsForHepth/s8-L-3.1-8B-base',
                   's9': 'LLMsForHepth/s9-L-3.1-8B-base',
                   's10': 'LLMsForHepth/s10-L-3.1-8B-base',
                   's1_qkv': 'LLMsForHepth/s1-L-3.1-8B-qkv_v2',
                   's2_qkv': 'LLMsForHepth/s2-L-3.1-8B-qkv',
                   }

def validate_adapters(adapters: Optional[List[str]]) -> Optional[List[str]]:
    """
    Validates adapter names against a predefined dictionary of valid adapters and their repositories.

    Args:
        adapters: List of adapter names, or None to indicate no adapters.

    Returns:
        The validated list of adapter names, or None if input is None.

    Raises:
        KeyError: If any adapter name is not found in ADAPTER_TO_REPO.keys().
    """
    if adapters is None:
        return None

    # Validate all adapters exist in repository
    invalid_adapters = [adapter for adapter in adapters
                        if adapter not in ADAPTER_TO_REPO.keys()]

    if invalid_adapters:
        valid_adapters = sorted(ADAPTER_TO_REPO.keys())
        raise KeyError(
            f"Invalid adapter name(s): {invalid_adapters}. "
            f"Valid adapters are: {valid_adapters}"
        )

    return adapters

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('dataset_repo', help='Huggingface dataset repository')

    # Optional arguments
    parser.add_argument('-a', '--adapters', nargs='+',
                        help='''Optional. Adapters to use (space-separated list). 
                        Choose from llama, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s1_qkv, s2_qkv''')
    parser.add_argument('-c', '--column', type=str, default='abstract',
                        help='Dataset column to tokenize (default: abstract)')
    parser.add_argument('-b', '--batch_size', type=int, default=12,
                        help='Batch size (default: 12)')
    parser.add_argument('-nb', '--no_base', action='store_true',
                        help='Do not evaluate using base model')
    # Parse the arguments
    args = parser.parse_args()
    return args


def get_perplexities(model, dataloader):
    model.eval()

    all_perplexities = []

    for step, inputs in enumerate(tqdm(dataloader)):
        with torch.inference_mode():
            # Clear cache before processing batch
            torch.cuda.empty_cache()
            gc.collect()

            inputs = inputs.to(model.device)
            outputs = model(**inputs, output_hidden_states=True)

            batch_perplexities = compute_perplexities(outputs['logits'], inputs['labels'])
            all_perplexities.extend(batch_perplexities.tolist())

    return all_perplexities


if __name__ == "__main__":
    # Get the parsed arguments
    args = parse_arguments()

    ds_repo = args.dataset_repo
    adapters = validate_adapters(args.adapters)  # this is either None or a list
    ds_col = args.column
    batch_size = args.batch_size
    no_base = args.no_base

    if (adapters is None) and (no_base is True):
        raise ValueError('Must specify adapters or no_base')

    ds = load_dataset(ds_repo, split='test')

    if ds_col not in ds.column_names:
        raise ValueError(f'Column {ds_col} not found in dataset {ds_repo}')

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_REPO)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    tokenised_ds = ds.map(lambda examples: tokenizer(examples[ds_col],
                                                     padding='do_not_pad',
                                                     truncation='do_not_truncate'),
                          batched=True,
                          remove_columns=ds.column_names,
                          desc="Tokenizing")
    tokenised_ds = tokenised_ds.map(lambda example: {"labels": example["input_ids"]},
                                    batched=True,
                                    desc="Adding labels")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_REPO,
                                                 device_map='auto',
                                                 trust_remote_code=True,
                                                 attn_implementation='flash_attention_2',
                                                 torch_dtype='float16')

    if no_base is False:
        dataloader = DataLoader(tokenised_ds, batch_size=batch_size, collate_fn=data_collator)
        logger.info('-' * 50)
        logger.info(f'Evaluating using model {BASE_MODEL_REPO}')
        logger.info('-' * 50)
        perpexities = get_perplexities(model, dataloader)
        ds = ds.add_column(name=f"perplexity_Llama-3.1-8B", column=perpexities)
        ds.push_to_hub(UPLOAD_REPO, split='test')
        torch.cuda.empty_cache()
        gc.collect()

    if adapters is not None:
        for adapter in adapters:
            model.load_adapter(peft_model_id=ADAPTER_TO_REPO[adapter],
                               adapter_name=adapter)

    for adapter in adapters:
        dataloader = DataLoader(tokenised_ds, batch_size=batch_size, collate_fn=data_collator)
        model_repo = ADAPTER_TO_REPO[adapter]
        model.set_adapter(adapter)
        logger.info('-' * 50)
        logger.info(f'Active adapter is {model.active_adapter}')
        logger.info(f'Evaluating using model {ADAPTER_TO_REPO[adapter]}')
        logger.info('-' * 50)
        perpexities = get_perplexities(model, dataloader)
        ds = ds.add_column(name=f"perplexity_{model_repo.split('/')[-1]}", column=perpexities)
        ds.push_to_hub(UPLOAD_REPO, split='test')
        torch.cuda.empty_cache()
        gc.collect()
