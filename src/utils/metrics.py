import torch
import numpy as np


def compute_perplexities(logits, labels):
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
    processed_logits = eval_pred.predictions
    ppls = processed_logits.tolist()
    return {"perplexities": ppls, "loss": np.mean(ppls)}


def perplexity_batch(logits, labels, compute_result=False):
    ppls = []
    mean_ppls = np.mean(ppls) if ppls else None
    return {"mean_perplexity_of_batch": mean_ppls}