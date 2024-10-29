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
from datasets import Dataset
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
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <json_file_path>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    json_data = load_json_file(json_file_path)
    fdirectory, fname = os.path.split(json_file_path)

    if json_data is not None:
        logger.info("JSON data loaded successfully:")
    else:
        logger.error("JSON data could not be loaded")
        sys.exit(1)

    if scorers[0].get('model_repo') is None:
        scorers[0]['model_repo'] = json_data['model']

    ds_dict = {k: v for k, v in json_data.items() if k in ['prompts', 'y_pred', 'y_true']}
    ds = Dataset.from_dict(ds_dict)
    ds = ds.map(lambda x: {'abstract_pred': x['prompts'] + ' ' + x['y_pred'],
                           'abstract_true': x['prompts'] + ' ' + x['y_true']},
                batched=False)

    for model_dict in scorers:
        model_repo = model_dict['model_repo']
        model_name = model_repo.split('/')[-1]
        model_name = 'SemScore' if model_name == 'all-mpnet-base-v2' else model_name
        model_cls = model_dict['model_cls']
        model_cfg = model_dict['model_cfg']
        batch_size = model_dict['batch_size']
        max_length = model_dict['max_length']

        model = model_cls.from_pretrained(model_repo, **model_cfg)
        tokenizer = AutoTokenizer.from_pretrained(model_repo)

        scorer = SimilarityScorer(model, tokenizer, batch_size, max_length)

        logger.info(f"Scoring using {model_name} model")
        model_scores = scorer.get_similarities(ds['abstract_true'], ds['abstract_pred'])
        mean_model_scores = mean(model_scores)
        json_data.update({model_name: {'cos_similarities': model_scores,
                                       'mean_cos_similarities': mean_model_scores}
                          })
        save_dict_to_json(json_data, fdirectory, fname)

        clear_cache()
