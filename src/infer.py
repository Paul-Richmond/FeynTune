import logging
import os
import re

import huggingface_hub
import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

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
