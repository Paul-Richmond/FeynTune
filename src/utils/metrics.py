"""
metrics.py - Perplexity Metrics for LLM Evaluation

This module provides utility functions for computing perplexity-based metrics
for evaluating language models. Perplexity is a common metric to assess how well
a language model predicts a sample of text.

The module includes functions for:
- Computing perplexity values from model logits and labels
- Converting perplexity scores to evaluation metrics
- Handling batch-level perplexity calculations

These functions are particularly useful for integration with evaluation pipelines
such as those provided by Hugging Face's Transformers library.
"""

import torch
import numpy as np


def compute_perplexities(logits, labels):
    """
    Compute perplexity scores from model logits and labels following logic from Evaluate's Perplexity metric:
    https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py

    This function calculates perplexity by:
    1. Creating a padding mask where -100 values in labels are ignored
    2. Shifting logits and labels for next-token prediction
    3. Computing cross-entropy loss with the padding mask applied
    4. Converting to perplexity via exponentiation

    Args:
        logits (torch.Tensor): Model output logits with shape (..., seq_len, vocab_size)
        labels (torch.Tensor): Ground truth labels with shape (..., seq_len)
                              where -100 indicates padding/ignored positions

    Returns:
        torch.Tensor: Perplexity scores for each sequence in the batch
    """
    padding_mask = torch.logical_not(labels.eq(-100))
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    padding_mask = padding_mask[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    perplexities = torch.exp(
                (loss_fct(logits.transpose(1, 2), labels) * padding_mask).sum(1)
                / padding_mask.sum(1)
            )
    return perplexities


def metric_perplexity(eval_pred):
    """
    Convert model predictions to a perplexity metric dictionary.

    This function is designed to be used with Hugging Face's evaluation pipeline.
    It expects an EvalPrediction object with pre-computed perplexity values.

    Args:
        eval_pred: An evaluation prediction object with a 'predictions' attribute
                  that contains pre-computed perplexity values

    Returns:
        dict: A dictionary containing the mean batch perplexity metric
    """
    processed_logits = eval_pred.predictions
    ppls = processed_logits.tolist()
    return {"mean_batch_perplexity": np.mean(ppls)}
