from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          )
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, DatasetDict


def load_optimizer(optimizer_cfg, model):
    optimizer = instantiate(optimizer_cfg, params=model.parameters(), **optimizer_cfg)
    return optimizer


def load_automodelforcausallm(cfg):
    # need to do this to get mixture of positional and keyword arguments
    model_cfg = OmegaConf.to_container(cfg.model_cfg, resolve=True)
    model_name = model_cfg.pop('name')

    if 'bnb_cfg' in cfg.keys():
        # we need to use OmegaConf.to_container to avoid json serialization errors when saving model
        quant_config = BitsAndBytesConfig(**OmegaConf.to_container(cfg.bnb_cfg, resolve=True))
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_cfg, quantization_config=quant_config)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_cfg)

    if 'lora_cfg' in cfg.keys():
        # we need to use OmegaConf.to_container to avoid json serialization errors when saving model
        lora_config = LoraConfig(**OmegaConf.to_container(cfg.lora_cfg, resolve=True))
        model = get_peft_model(model, lora_config)

    return model


def load_tokenizer(tokenizer_cfg):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg.name)
    if 'Llama-3' in tokenizer_cfg.name:
        from tokenizers import processors
        # found this on
        # https://github.com/huggingface/transformers/issues/30947
        bos = "<|begin_of_text|>"
        eos = "<|end_of_text|>"
        tokenizer._tokenizer.post_processor = processors.Sequence(
            [
                processors.ByteLevel(trim_offsets=False),
                processors.TemplateProcessing(
                    single=f"{bos}:0 $A:0 {eos}:0",
                    pair=f"{bos}:0 $A:0 {bos}:1 $B:1 {eos}:1",
                    special_tokens=[
                        (bos, tokenizer.bos_token_id),
                        (eos, tokenizer.eos_token_id),
                    ],
                ),
            ]
        )
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def instantiate_callbacks(callbacks_cfg):
    callbacks = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(instantiate(cb_conf))

    return callbacks


def load_train_and_eval_datasets(datasets_cfg):
    datasets_ = {}
    for split in datasets_cfg.splits:
        datasets_[split] = load_dataset(datasets_cfg.name, split=split)
    return DatasetDict(datasets_)
