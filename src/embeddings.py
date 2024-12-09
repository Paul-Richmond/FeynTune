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
from peft import PeftModel, PeftConfig
from datasets import Dataset, load_dataset
from tqdm import tqdm
from utils.callbacks import SimilarityScorer

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)

# Create a logger and set level to INFO.
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    ds_repo_download = 'LLMsForHepth/infer_hep_th'
    col_to_embed = 'abstract'
    base_model_repo = 'meta-llama/Llama-3.1-8B'
    ds_repo_upload = 'LLMsForHepth/emb_hep_th_abstract'

    ds = load_dataset(ds_repo_download, split='test')
    upload_ds = Dataset.from_dict({col_to_embed: ds[col_to_embed]})

    base_model = AutoModelForCausalLM.from_pretrained(base_model_repo,
                                                      device_map='auto',
                                                      attn_implementation='flash_attention_2',
                                                      trust_remote_code=True,
                                                      torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_repo)

    lora_adapters = [None,
                     'LLMsForHepth/s1-L-3.1-8B-base',
                     'LLMsForHepth/s2-L-3.1-8B-base',
                     'LLMsForHepth/s3-L-3.1-8B-base_v3']

    for lora_adapter in lora_adapters:

        if lora_adapter is None:
            new_col_name = 'emb_' + base_model_repo.split('/')[-1]
            model = base_model
        else:
            new_col_name = 'emb_' + lora_adapter.split('/')[-1]
            model = PeftModel.from_pretrained(base_model, lora_adapter)

        scorer = SimilarityScorer(model, tokenizer, batch_size=24)

        batches = scorer.get_batches(ds[col_to_embed])
        all_embeddings = []

        with torch.inference_mode():
            for idx in tqdm(range(len(batches)), desc=f'Computing embeddings'):
                embeddings = scorer.get_embeddings(batches[idx]).tolist()
                all_embeddings.extend(embeddings)
                clear_cache()

        upload_ds = upload_ds.add_column(name=new_col_name, column=all_embeddings)
        upload_ds.push_to_hub(repo_id=ds_repo_upload, split='test')

    huggingface_hub.logout()
