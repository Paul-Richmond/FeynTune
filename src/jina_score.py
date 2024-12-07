import sys
import logging
import json
import os
import gc
import torch
from dotenv import load_dotenv
import huggingface_hub
from transformers import (AutoTokenizer,
                          AutoModel,
                          AutoModelForCausalLM)
from datasets import Dataset, load_dataset
from statistics import mean
from utils.callbacks import SimilarityScorer

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)

# Create a logger and set level to INFO.
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

# model_cfg_big = {'device_map': 'auto',
#                  'trust_remote_code': True,
#                  'attn_implementation': 'flash_attention_2',
#                  'torch_dtype': torch.bfloat16}
#
# model_cfg = {"attn_implementation": "eager",
#              "trust_remote_code": True,
#              "quantization_config": None,
#              }
#
# scorers = [{'model_repo': None, 'model_cls': AutoModelForCausalLM,
#             'model_cfg': model_cfg_big, 'batch_size': 24, 'max_length': None},
#            {'model_repo': "sentence-transformers/all-mpnet-base-v2",
#             'model_cls': AutoModel, 'model_cfg': model_cfg, 'batch_size': 512, 'max_length': 384},
#            {'model_repo': "jinaai/jina-embeddings-v3",
#             'model_cls': AutoModel, 'model_cfg': model_cfg, 'batch_size': 512, 'max_length': 8192}
#            ]


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    # choose a dataset from LLMsForHepth/infer_hep_th or LLMsForHepth/infer_hep-ph_gr-qc
    ds = load_dataset('LLMsForHepth/infer_hep_th', split='test')
    y_true = ds['abstract']
    comp_columns = [col_name for col_name in ds.column_names if 'comp' in col_name]

    model_repo = 'jinaai/jina-embeddings-v3'
    model = AutoModel.from_pretrained(model_repo,
                                      attn_implementation='eager',
                                      trust_remote_code=True,
                                      quantization_config=None)
    tokenizer = AutoTokenizer.from_pretrained(model_repo)

    scorer = SimilarityScorer(model, tokenizer, batch_size=512, max_length=8192)  # 2500 fits on 80Gb H100 GPU

    for col in comp_columns:
        logger.info(f"Scoring using y_pred={col}")
        y_pred = ds[col]

        scores = scorer.get_similarities(y_true, y_pred)
        ds = ds.add_column(name=f"score_{col[5:]}", column=scores)

        clear_cache()

    ds.push_to_hub('paulrichmond/jina_scores_hep_th', split='test')
    huggingface_hub.logout()
