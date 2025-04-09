import argparse
import gc
import logging
import os
from math import ceil, floor
from typing import Optional, List

import huggingface_hub
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.callbacks import SimilarityScorer

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)

# Create a logger and set level to INFO.
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

BASE_MODEL_REPO = 'meta-llama/Meta-Llama-3.1-8B'

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
                   's3_qkv': 'LLMsForHepth/s3-L-3.1-8B-qkv',
                   's4_qkv': 'LLMsForHepth/s4-L-3.1-8B-qkv',
                   's5_qkv': 'LLMsForHepth/s5-L-3.1-8B-qkv',
                   's6_qkv': 'LLMsForHepth/s6-L-3.1-8B-qkv',
                   's7_qkv': 'LLMsForHepth/s7-L-3.1-8B-qkv2',
                   's8_qkv': 'LLMsForHepth/s8-L-3.1-8B-qkv',
                   's9_qkv': 'LLMsForHepth/s9-L-3.1-8B-qkv',
                   's10_qkv': 'LLMsForHepth/s10-L-3.1-8B-qkv',
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
                        Choose from llama, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, 
                        s1_qkv, s2_qkv, s3_qkv, s4_qkv, s5_qkv, s6_qkv, s7_qkv, s8_qkv, s9_qkv, s10_qkv,''')
    parser.add_argument('-c', '--column', type=str, default='abstract',
                        help='Dataset column to tokenize (default: abstract)')
    parser.add_argument('-b', '--batch_size', type=int, default=12,
                        help='Batch size (default: 12)')
    parser.add_argument('-nb', '--no_base', action='store_true',
                        help='Do not evaluate using base model')
    # Parse the arguments
    args = parser.parse_args()
    return args


# I found this solution at https://github.com/pytorch/pytorch/issues/64947
# Using torch.quantile fails because it has a maximum input size of 2**24 which
# is smaller than the tensors we are dealing with here
def torch_quantile(
        input: torch.Tensor,
        q: float | torch.Tensor,
        dim: int | None = None,
        keepdim: bool = False,
        *,
        interpolation: str = "nearest",
        out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


def get_statistics(data):
    min_ = torch.min(data)
    mean_ = torch.mean(data)
    std = torch.std(data)
    quartiles = torch.tensor([torch_quantile(data, q, interpolation='nearest') for q in [0.25, 0.50, 0.75]])

    return {'min': min_,
            'mean': mean_,
            'std': std,
            'quartiles': quartiles}


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    # Get the parsed arguments
    args = parse_arguments()

    ds_repo = args.dataset_repo
    ds_name = ds_repo.split('/')[-1]
    adapters = validate_adapters(args.adapters)  # this is either None or a list
    ds_col = args.column
    batch_size = args.batch_size
    no_base = args.no_base

    if (adapters is None) and (no_base is True):
        raise ValueError('Must specify adapters or no_base')

    ds = load_dataset(ds_repo, split='test')

    if ds_col not in ds.column_names:
        raise ValueError(f'Column {ds_col} not found in dataset {ds_repo}')

    y_true = ds[ds_col]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_REPO)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_REPO,
                                                 device_map='auto',
                                                 trust_remote_code=True,
                                                 attn_implementation='flash_attention_2',
                                                 torch_dtype='float16')

    if no_base is False:
        logger.info('-' * 50)
        logger.info(f'Evaluating using model {BASE_MODEL_REPO}')
        logger.info('-' * 50)

        file = f"pwcs_{ds_name}_llama_{ds_col[:3]}.pt"

        scorer = SimilarityScorer(model, tokenizer, batch_size=batch_size)
        y_true_batches = scorer.get_batches(y_true)
        y_true_embeddings = torch.tensor([], device=scorer.device)

        with torch.inference_mode():
            for idx in tqdm(range(len(y_true_batches)), desc=f'Computing embeddings'):
                y_true_batch_embeddings = scorer.get_embeddings(y_true_batches[idx])
                y_true_embeddings = torch.cat((y_true_embeddings, y_true_batch_embeddings), dim=0)

        del scorer
        clear_cache()

        logger.info('Computing pairwise cosine similarities')
        pwcs = pairwise_cosine_similarity(y_true_embeddings, y_true_embeddings)
        logger.info('Saving pairwise cosine similarities')
        torch.save(pwcs, file)

        pwcs_stats = get_statistics(pwcs)
        logger.info(f'Statistics for pairwise cosine similarity of {file} are: \n\n {pwcs_stats} \n')

    if adapters is not None:
        for adapter in adapters:
            model.load_adapter(peft_model_id=ADAPTER_TO_REPO[adapter],
                               adapter_name=adapter)

    for adapter in adapters:
        model_repo = ADAPTER_TO_REPO[adapter]
        model.set_adapter(adapter)

        logger.info('-' * 50)
        logger.info(f'Active adapter is {model.active_adapter}')
        logger.info(f'Evaluating using model {ADAPTER_TO_REPO[adapter]}')
        logger.info('-' * 50)

        file = f"pwcs_{ds_name}_{adapter}_{ds_col[:3]}.pt"

        scorer = SimilarityScorer(model, tokenizer, batch_size=batch_size)
        y_true_batches = scorer.get_batches(y_true)
        y_true_embeddings = torch.tensor([], device=scorer.device)

        with torch.inference_mode():
            for idx in tqdm(range(len(y_true_batches)), desc=f'Computing embeddings'):
                y_true_batch_embeddings = scorer.get_embeddings(y_true_batches[idx])
                y_true_embeddings = torch.cat((y_true_embeddings, y_true_batch_embeddings), dim=0)

        del scorer
        clear_cache()

        logger.info('Computing pairwise cosine similarities')
        pwcs = pairwise_cosine_similarity(y_true_embeddings, y_true_embeddings)
        logger.info('Saving pairwise cosine similarities')
        torch.save(pwcs, file)

        pwcs_stats = get_statistics(pwcs)
        logger.info(f'Statistics for pairwise cosine similarity of {file} are: \n\n {pwcs_stats} \n')

huggingface_hub.logout()
