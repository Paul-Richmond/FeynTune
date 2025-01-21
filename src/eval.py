import logging
import os
from dotenv import load_dotenv
import huggingface_hub
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForSeq2Seq)
from datasets import load_dataset

from utils.metrics import compute_perplexities, metric_perplexity

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")
huggingface_hub.login(token=hf_token)

# Create a logger and set level to INFO.
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    ds_repo = 'LLMsForHepth/hep-ph_gr-qc_primary'
    ds_col = 'abstract'
    model_repo = 'meta-llama/Meta-Llama-3.1-8B'
    # 'meta-llama/Meta-Llama-3.1-8B'
    # 'LLMsForHepth/s1-L-3.1-8B-base'
    # 'LLMsForHepth/s2-L-3.1-8B-base'
    # 'LLMsForHepth/s3-L-3.1-8B-base_v3'
    # 'LLMsForHepth/s4-L-3.1-8B-base'
    # 'LLMsForHepth/s5-L-3.1-8B-base'
    # 'LLMsForHepth/s6-L-3.1-8B-base'

    ds = load_dataset(ds_repo, split='test')

    model = AutoModelForCausalLM.from_pretrained(model_repo,
                                                 device_map='auto',
                                                 trust_remote_code=True,
                                                 attn_implementation='flash_attention_2',
                                                 torch_dtype='bfloat16')

    tokenizer = AutoTokenizer.from_pretrained(model_repo)
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

    training_args = TrainingArguments(output_dir="tmp_trainer",
                                      bf16=True,
                                      per_device_eval_batch_size=16,
                                      report_to=None,
                                      logging_steps=1)

    # why pad_to_multiple_of=8? see https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#opt-tensor-cores
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           pad_to_multiple_of=8)  # dynamically pads a batch to all have same tensor shapes
    callbacks = None
    optimizer = None

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=None,
                      eval_dataset=tokenised_ds,  # will evaluate all datasets within tokenised_ds
                      data_collator=data_collator,
                      tokenizer=tokenizer,
                      optimizers=(optimizer, None),
                      callbacks=callbacks,
                      compute_metrics=metric_perplexity,
                      preprocess_logits_for_metrics=compute_perplexities
                      )

    logger.info(f"Evaluating model {model_repo}")
    res = trainer.evaluate()
    print(res)
