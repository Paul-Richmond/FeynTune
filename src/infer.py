import gc
import logging
import json
import os
import re

import huggingface_hub
import hydra
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.callbacks import AbstractCompleter, SemsScore
from utils.instantiators import load_automodelforcausallm, load_dataset_splits

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
wandb_token = os.getenv("WANDB_API_KEY")

huggingface_hub.login(token=hf_token)
wandb.login(key=wandb_token)

logger = logging.getLogger(__name__)


def split_abstracts(example):
    """
    Splits an abstract into a prompt and ground truth.

    The prompt is created from the first half (or slightly more) of the sentences in the abstract,
    and the ground truth is the remaining sentences.
    If there is only a single sentence then we split instead on spaces to get, roughly, words
    and take the first half (or slightly more) of these.

    Args:
        example (dict): A dictionary containing the 'abstract' text to be split.

    Returns:
        dict: A dictionary with 'prompt' and 'y_true' keys containing the split abstract parts.
    """
    text = example['abstract']
    # Split the abstract into sentences (i.e. text sequences which end with any of .!? and a space)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Calculate the split point
    total_sentences = len(sentences)
    if total_sentences > 1:  # more than 1 sentence so can split
        split_point = (total_sentences + 1) // 2  # Ensures the prompt has >= number of sentences than y_true
        # Join the sentences back into two parts
        prompt = ' '.join(sentences[:split_point])
        y_true = ' '.join(sentences[split_point:])
    else:  # only a single sentence so split on words (latex commands between $$ might get split)
        words = text.split()
        total_words = len(words)
        split_point = (total_words + 1) // 2  # Ensures the prompt has >= number of sentences than y_true
        # Join the sentences back into two parts
        prompt = ' '.join(words[:split_point])
        y_true = ' '.join(words[split_point:])
    return {'prompt': prompt, 'y_true': y_true}


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


@hydra.main(version_base=None, config_path="../configs", config_name="default_infer")
def main(cfg: DictConfig) -> None:
    ds = load_dataset_splits(cfg.dataset)  # expect DatasetDict object with a single key
    ds_split_names = list(ds)
    ds = ds[list(ds)[0]]  # extract the single Dataset object from the DatasetDict object
    ds = ds.remove_columns(list(set(ds.column_names) - {'id', 'abstract'}))  # keep only 'id' and 'abstract' columns

    ds = ds.map(split_abstracts,
                batched=False,
                desc='Splitting abstracts')

    if 'generation_cfg' in cfg.inference.keys():
        generation_cfg = OmegaConf.to_container(cfg.inference.generation_cfg, resolve=True)
        sampling_params = SamplingParams(**generation_cfg)
    else:
        sampling_params = SamplingParams()

    lora_adapter_name = cfg.inference.get('lora_adapter_name', None)
    model_name = lora_adapter_name if lora_adapter_name is not None else cfg.inference.base_model_name

    if lora_adapter_name is not None:
        enable_lora = True
        lora_request = LoRARequest("sql_adapter", 1, lora_path=lora_adapter_name)
    else:
        enable_lora = False
        lora_request = None

    # Create an LLM.
    if 'model_cfg' in cfg.inference.keys():
        llm = LLM(model=cfg.inference.base_model_name, enable_lora=enable_lora, **cfg.inference.model_cfg)
    else:
        llm = LLM(model=cfg.inference.base_model_name, enable_lora=enable_lora)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(ds['prompt'], sampling_params, lora_request=lora_request)

    vllm_prompts = []
    completions = []
    prompt_token_lengths = []
    completion_token_lengths = []

    for output in outputs:
        vllm_prompts.append(output.prompt)
        prompt_token_lengths.append(len(output.prompt_token_ids))
        completions.append(output.outputs[0].text)
        completion_token_lengths.append(len(output.outputs[0].token_ids))

    output_dict = {'model': model_name,
                   'dataset': {'name': cfg.dataset.name, 'splits': ds_split_names[0]},
                   'sampling_params': sampling_params.__repr__(),
                   'enable_lora': enable_lora,
                   'prompts': vllm_prompts,
                   'y_pred': completions,
                   'y_true': ds['y_true'],
                   'prompt_token_lengths': prompt_token_lengths,
                   'completion_token_lengths': completion_token_lengths}

    save_dict_to_json(data=output_dict,
                      directory=cfg.inference.output_dir,
                      filename=cfg.inference.output_fname)


if __name__ == "__main__":
    main()
