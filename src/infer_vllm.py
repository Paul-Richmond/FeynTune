import gc
import logging
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

    Args:
        example (dict): A dictionary containing the 'abstract' text to be split.

    Returns:
        dict: A dictionary with 'prompt' and 'y_true' keys containing the split abstract parts.
    """
    # Split the abstract into sentences (i.e. text sequences which end with any of .!? and a space)
    sentences = re.split(r'(?<=[.!?])\s+', example['abstract'])
    # Calculate the split point
    total_sentences = len(sentences)
    split_point = (total_sentences + 1) // 2  # Ensures the prompt has >= number of sentences than y_true
    # Join the sentences back into two parts
    prompt = ' '.join(sentences[:split_point])
    y_true = ' '.join(sentences[split_point:])
    return {'prompt': prompt, 'y_true': y_true}


def parse_y_pred(example):
    """
    Extracts the predicted text from the generated predictions.

    Args:
        example (dict): A dictionary containing 'prompt' and 'predictions' keys.

    Returns:
        dict: A dictionary with 'y_pred' key containing the generated output without the prompt.
    """
    len_prompt = len(example['prompt'])
    y_pred = example['predictions'][len_prompt:]
    return {'y_pred': y_pred}


@hydra.main(version_base=None, config_path="../configs", config_name="default_infer")
def main(cfg: DictConfig) -> None:
    try:
        wandb_runpath = cfg.inference.wandb_runpath
        if len(wandb_runpath.split('/')) != 3:
            logger.error(f"wandb_runpath should be in the form '<entity>/<project>/<run_id>' but \n"
                         f"wandb_runpath='{wandb_runpath}' was set instead.")
            raise ValueError
        else:
            entity, project, run_id = wandb_runpath.split('/')
            run = wandb.init(entity=entity, project=project, id=run_id, resume="allow")
    except MissingMandatoryValue:
        logger.info(f"You have not set a value for wandb_runpath, initializing using wandb.init() instead.")
        run = wandb.init()

    ds = load_dataset_splits(cfg.dataset)  # expect DatasetDict object with a single key
    ds = ds[list(ds)[0]]  # extract the single Dataset object from the DatasetDict object
    ds = ds.remove_columns(list(set(ds.column_names) - {'id', 'abstract'}))  # keep only 'id' and 'abstract' columns

    ds = ds.map(split_abstracts,
                batched=False,
                desc='Splitting abstracts')

    if 'generation_cfg' in cfg.inference.keys():
        generation_cfg = OmegaConf.to_container(cfg.inference.generation_cfg, resolve=True)
    else:
        generation_cfg = {"temperature": 0.7,
                          "top_p": 0.1,
                          "repetition_penalty": 1.18,
                          "top_k": 40,
                          "max_tokens": 1024}

    sampling_params = SamplingParams(**generation_cfg)

    # sql_lora_path = huggingface_hub.snapshot_download(repo_id="LLMsForHepth/test_llama_3.1_batch48")

    # Create an LLM.
    llm = LLM(model="LLMsForHepth/Llama-3.1-8B_lower_lr_merged",
              trust_remote_code=True)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(ds['prompt'], sampling_params)

    predictions = [output.outputs[0].text for output in outputs]
    ds_with_completions = ds.add_column(name='predictions', column=predictions)

    ds_with_completions = ds_with_completions.map(parse_y_pred,
                                                  batched=False,
                                                  desc='Parsing y_pred')
    # Tidy up before computing SemScore
    del llm, ds
    torch.cuda.empty_cache()
    gc.collect()

    # Score the predictions
    semscorer = SemsScore()
    sem_scores = semscorer.get_similarities(ds_with_completions['y_true'],
                                            ds_with_completions['y_pred'])
    ds_with_completions = ds_with_completions.add_column(name='SemScore', column=sem_scores)

    # Log to wandb
    inference_table = wandb.Table(dataframe=ds_with_completions.to_pandas())
    gen_cfg_table = wandb.Table(columns=list(generation_cfg.keys()),
                                data=[list(generation_cfg.values())])
    run.log({"inference": inference_table, "inference_gen_cfg": gen_cfg_table})


if __name__ == "__main__":
    main()
