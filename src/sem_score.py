import gc
import logging
import os

import huggingface_hub
import torch
from datasets import load_dataset
from dotenv import load_dotenv

from utils.callbacks import SemsScore

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
    ds = load_dataset('LLMsForHepth/infer_hep_th', split='test')
    y_true = ds['abstract']
    comp_columns = [col_name for col_name in ds.column_names if 'comp' in col_name]

    scorer = SemsScore(batch_size=1000)

    for col in comp_columns:
        logger.info(f"Scoring using y_pred={col}")
        y_pred = ds[col]

        scores = scorer.get_similarities(y_true, y_pred)
        ds = ds.add_column(name=f"score_{col[5:]}", column=scores)

        clear_cache()

    ds.push_to_hub('LLMsForHepth/sem_scores_hep_th', split='test')
    huggingface_hub.logout()
