import json
from datasets import load_dataset
from transformers import TrainerCallback, pipeline, logging
from transformers.integrations import WandbCallback
import torch

logger = logging.get_logger(__name__)


class AbstractCompleter:
    def __init__(self, model, tokenizer, dataset, batch_size=None, generation_config=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = 16 if batch_size is None else batch_size
        if generation_config is None:
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.1,
                "repetition_penalty": 1.18,
                "top_k": 40,
                "do_sample": True,
                "max_new_tokens": 1024,
                "pad_token_id": self.tokenizer.pad_token_id,
                "max_time": 30
            }
        else:
            self.generation_config = generation_config

        self.tokenizer.pad_token = tokenizer.eos_token
        self.original_pad_side = tokenizer.padding_side
        self.label = 'abstract'

    def _predict(self, batch):
        texts = batch['prompt']
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(texts,
                                      padding='longest',
                                      truncation=True,
                                      max_length=min(2048, self.tokenizer.model_max_length),
                                      pad_to_multiple_of=8,
                                      add_special_tokens=False,
                                      return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**model_inputs, **self.generation_config)
        batch["predictions"] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.tokenizer.padding_side = self.original_pad_side
        return batch

    def get_predictions(self):
        self.dataset = self.dataset.map(lambda example: {'prompt': example[self.label][:len(example[self.label]) // 2]},
                                        desc='Generating prompts')
        self.dataset = self.dataset.map(self._predict,
                                        batched=True,
                                        batch_size=self.batch_size,
                                        desc='Generating model output')
        return self.dataset


class GenCallback(WandbCallback):
    def __init__(self, generation_config=None):
        super().__init__()
        self.generation_config = generation_config
        self.dataset = load_dataset("LLMsForHepth/arxiv_hepth_first_overfit").get('train')
        self.add_prompts = True

    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        self._gen_and_log(state, model, tokenizer)

    def _gen_and_log(self, state, model, tokenizer):
        switch_back_to_train = False
        if model.training:
            model.eval()
            switch_back_to_train = True
        with torch.no_grad():
            completer = AbstractCompleter(model, tokenizer, dataset=self.dataset, batch_size=5,
                                          generation_config=self.generation_config)
            completions = completer.get_predictions()
            new_table = self._wandb.Table(columns=['global_step', 'abstract 1', 'abstract 2',
                                                   'abstract 3', 'abstract 4', 'abstract 5'])
            if self.add_prompts:
                new_table.add_data("Prompt", *completions['prompt'])
                self.add_prompts = False
            new_table.add_data(str(state.global_step), *completions['predictions'])
            self._wandb.log({f"predictions": new_table}, commit=False)
        if switch_back_to_train:
            model.train()
