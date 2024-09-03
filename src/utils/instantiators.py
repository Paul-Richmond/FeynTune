from datasets import DatasetDict, load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import Optimizer
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          PreTrainedModel,
                          PreTrainedTokenizerFast,
                          TrainerCallback
                          )
from typing import Any, Dict, List, Optional, Union


def load_optimizer(optimizer_cfg: DictConfig, model: PreTrainedModel) -> Optimizer:
    """
    Instantiate and return an optimizer based on the provided configuration.

    Args:
        optimizer_cfg (DictConfig): Configuration for the optimizer.
        model (PreTrainedModel): The model whose parameters will be optimized.

    Returns:
        Optimizer: The instantiated optimizer.
    """
    optimizer: Optimizer = instantiate(optimizer_cfg, params=model.parameters(), **optimizer_cfg)
    return optimizer


def load_automodelforcausallm(cfg: DictConfig) -> Union[PreTrainedModel, PeftModel]:
    """
    Load and configure an AutoModelForCausalLM based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration containing model, quantization, and LoRA settings.

    Returns:
        Union[PreTrainedModel, PeftModel]: The configured causal language model.
    """
    model_cfg: Dict[str, Any] = OmegaConf.to_container(cfg.model_cfg, resolve=True)
    model_name: str = model_cfg.pop('name')

    if 'bnb_cfg' in cfg.keys():
        quant_config: BitsAndBytesConfig = BitsAndBytesConfig(**OmegaConf.to_container(cfg.bnb_cfg, resolve=True))
        model: Union[PreTrainedModel, PeftModel] \
            = AutoModelForCausalLM.from_pretrained(model_name, **model_cfg, quantization_config=quant_config)
        model = prepare_model_for_kbit_training(model)
    else:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name, **model_cfg)

    if 'lora_cfg' in cfg.keys():
        lora_config: LoraConfig = LoraConfig(**OmegaConf.to_container(cfg.lora_cfg, resolve=True))
        model = get_peft_model(model, lora_config)

    return model


def load_tokenizer(tokenizer_cfg: DictConfig) -> Union[PreTrainedTokenizerFast, AutoTokenizer]:
    """
    Load and configure a tokenizer based on the provided configuration.

    Args:
        tokenizer_cfg (DictConfig): Configuration for the tokenizer.

    Returns:
        Union[PreTrainedTokenizerFast, AutoTokenizer]: The configured tokenizer.
    """
    tokenizer: Union[PreTrainedTokenizerFast, AutoTokenizer] = AutoTokenizer.from_pretrained(tokenizer_cfg.name)
    if 'Llama-3' in tokenizer_cfg.name:  # see https://github.com/huggingface/transformers/issues/30947
        from tokenizers import processors
        bos: str = "<|begin_of_text|>"
        eos: str = "<|end_of_text|>"
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
    else:
        tokenizer.add_eos_token = True
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    return tokenizer


def instantiate_callbacks(callbacks_cfg: Optional[DictConfig]) -> List[TrainerCallback]:
    """
    Instantiate callbacks based on the provided configuration.

    Args:
        callbacks_cfg (Optional[DictConfig]): Configuration for callbacks.

    Returns:
        List[TrainerCallback]: List of instantiated callbacks.

    Raises:
        TypeError: If callbacks_cfg is not a DictConfig.
    """
    callbacks: List[TrainerCallback] = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(instantiate(cb_conf))

    return callbacks


def load_dataset_splits(datasets_cfg: DictConfig) -> DatasetDict:
    """
    Load dataset splits based on the provided configuration.

    Args:
        datasets_cfg (DictConfig): Configuration for datasets.

    Returns:
        DatasetDict: A dictionary containing the loaded dataset splits.
    """
    datasets_: Dict[str, Any] = {}
    for split, split_value in datasets_cfg.splits.items():
        datasets_[split] = load_dataset(datasets_cfg.name, split=split_value)
    return DatasetDict(datasets_)


def instantiate_training(cfg: DictConfig) -> tuple:
    """
    Instantiate trainer and training arguments based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration containing trainer and training arguments.

    Returns:
        tuple: A tuple containing the instantiated trainer and training arguments.
    """
    partial_trainer: Any = instantiate(cfg.trainer_cfg)
    tr_args: Any = instantiate(cfg.training_args_cfg,
                               **OmegaConf.to_container(cfg.training_args_cfg, resolve=True))
    return partial_trainer, tr_args
