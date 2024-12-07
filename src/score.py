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

model_cfg_big = {'device_map': 'auto',
                 'trust_remote_code': True,
                 'attn_implementation': 'flash_attention_2',
                 'torch_dtype': torch.bfloat16}

model_cfg = {"attn_implementation": "eager",
             "trust_remote_code": True,
             "quantization_config": None,
             }

scorers = [{'model_repo': None, 'model_cls': AutoModelForCausalLM,
            'model_cfg': model_cfg_big, 'batch_size': 24, 'max_length': None},
           {'model_repo': "sentence-transformers/all-mpnet-base-v2",
            'model_cls': AutoModel, 'model_cfg': model_cfg, 'batch_size': 512, 'max_length': 384},
           {'model_repo': "jinaai/jina-embeddings-v3",
            'model_cls': AutoModel, 'model_cfg': model_cfg, 'batch_size': 512, 'max_length': 8192}
           ]

# scorers = [{'model_repo': "sentence-transformers/all-mpnet-base-v2",
#             'model_cls': AutoModel, 'model_cfg': model_cfg, 'batch_size': 512, 'max_length': 384},
#            {'model_repo': "jinaai/jina-embeddings-v3",
#             'model_cls': AutoModel, 'model_cfg': model_cfg, 'batch_size': 512, 'max_length': 8192}
#            ]

# scorers = [{'model_repo': "jinaai/jina-embeddings-v3",
#             'model_cls': AutoModel, 'model_cfg': model_cfg, 'batch_size': 256, 'max_length': 8192}
#            ]


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()


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
    # choose an initial dataset from LLMsForHepth/infer_hep_th or LLMsForHepth/infer_hep-ph_gr-qc
    # subsequent runs need LLMsForHepth/llm_scores_hep_th etc
    ds = load_dataset('LLMsForHepth/llm_scores_hep-ph_gr-qc', split='test')
    y_true = ds['abstract']
    comp_columns = [col_name for col_name in ds.column_names if 'comp_Llama-3.1-8B' in col_name]

    model_repo = 'meta-llama/Llama-3.1-8B'
    model = AutoModel.from_pretrained(model_repo,
                                      device_map='auto',
                                      attn_implementation='flash_attention_2',
                                      trust_remote_code=True,
                                      torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_repo)

    scorer = SimilarityScorer(model, tokenizer, batch_size=24)

    for col in comp_columns:
        logger.info(f"Scoring using y_pred={col}")
        y_pred = ds[col]

        scores = scorer.get_similarities(y_true, y_pred)
        ds = ds.add_column(name=f"score_{col[5:]}", column=scores)

        clear_cache()

    ds.push_to_hub('LLMsForHepth/llm_scores_hep-ph_gr-qc', split='test')
    huggingface_hub.logout()
