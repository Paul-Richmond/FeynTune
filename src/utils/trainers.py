from transformers import Trainer

from .metrics import compute_perplexities


class PerplexityAsLossTrainer(Trainer):
    def __init__(
            self,
            model=None,
            args=None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=(None, None),
            preprocess_logits_for_metrics=None
    ):
        super().__init__(model,
                         args,
                         data_collator,
                         train_dataset,
                         eval_dataset,
                         tokenizer,
                         model_init,
                         compute_metrics,
                         callbacks,
                         optimizers,
                         preprocess_logits_for_metrics)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        loss = compute_perplexities(outputs['logits'], inputs['labels']).mean()
        return (loss, outputs) if return_outputs else loss


# class PerplexityTrainer(Trainer):
#     def __init__(
#             self,
#             model=None,
#             args=None,
#             data_collator=None,
#             train_dataset=None,
#             eval_dataset=None,
#             tokenizer=None,
#             model_init=None,
#             compute_metrics=None,
#             callbacks=None,
#             optimizers=(None, None),
#             preprocess_logits_for_metrics=None
#     ):
#         super().__init__(model,
#                          args,
#                          data_collator,
#                          train_dataset,
#                          eval_dataset,
#                          tokenizer,
#                          model_init,
#                          compute_metrics,
#                          callbacks,
#                          optimizers,
#                          preprocess_logits_for_metrics)
#
#         self.state.batch_perplexities = None
#         self.state.original_loss = None
#
#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         How the loss is computed by Trainer. By default, all models return the loss in the first element.
#
#         Subclass and override for custom behavior.
#         """
#         if self.label_smoother is not None and "labels" in inputs:
#             labels = inputs.pop("labels")
#         else:
#             labels = None
#         outputs = model(**inputs)
#         # Save past state if it exists
#         # TODO: this needs to be fixed and made cleaner later.
#         if self.args.past_index >= 0:
#             self._past = outputs[self.args.past_index]
#
#         if labels is not None:
#             unwrapped_model = self.accelerator.unwrap_model(model)
#             if _is_peft_model(unwrapped_model):
#                 model_name = unwrapped_model.base_model.model._get_name()
#             else:
#                 model_name = unwrapped_model._get_name()
#             if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#                 loss = self.label_smoother(outputs, labels, shift_labels=True)
#             else:
#                 loss = self.label_smoother(outputs, labels)
#         else:
#             if isinstance(outputs, dict) and "loss" not in outputs:
#                 raise ValueError(
#                     "The model did not return a loss from the inputs, only the following keys: "
#                     f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
#                 )
#             # We don't use .loss here since the model may return tuples instead of ModelOutput.
#             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
#
#         perplexities = compute_perplexities(outputs['logits'], inputs['labels'])
#         self.state.perplexity = perplexities.mean().detach().item()
#
#         return (loss, outputs) if return_outputs else loss
#
#     def log(self, logs):
#         if self.state.epoch is not None:
#             logs["epoch"] = self.state.epoch
#         if self.args.include_num_input_tokens_seen:
#             logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
#
#         if not self.control.should_evaluate:
#             logs["perplexity"] = self.state.perplexity
#         else:
#             logs["eval_perplexity"] = self.state.perplexity
#
#         output = {**logs, **{"step": self.state.global_step}}
#         self.state.log_history.append(output)
#         self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
#
#
# class BatchPerplexityTrainer(Trainer):
#     def __init__(
#             self,
#             model=None,
#             args=None,
#             data_collator=None,
#             train_dataset=None,
#             eval_dataset=None,
#             tokenizer=None,
#             model_init=None,
#             compute_metrics=None,
#             callbacks=None,
#             optimizers=(None, None),
#             preprocess_logits_for_metrics=None
#     ):
#         super().__init__(model,
#                          args,
#                          data_collator,
#                          train_dataset,
#                          eval_dataset,
#                          tokenizer,
#                          model_init,
#                          compute_metrics,
#                          callbacks,
#                          optimizers,
#                          preprocess_logits_for_metrics)
#
#         self.train_perplexities_history = []
#         self.eval_perplexities_history = []
#         self.batch_perplexities = None
#         self.state.perplexity = None
#
#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         How the loss is computed by Trainer. By default, all models return the loss in the first element.
#
#         Subclass and override for custom behavior.
#         """
#         if self.label_smoother is not None and "labels" in inputs:
#             labels = inputs.pop("labels")
#         else:
#             labels = None
#         outputs = model(**inputs)
#         # Save past state if it exists
#         # TODO: this needs to be fixed and made cleaner later.
#         if self.args.past_index >= 0:
#             self._past = outputs[self.args.past_index]
#
#         if labels is not None:
#             unwrapped_model = self.accelerator.unwrap_model(model)
#             if _is_peft_model(unwrapped_model):
#                 model_name = unwrapped_model.base_model.model._get_name()
#             else:
#                 model_name = unwrapped_model._get_name()
#             if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#                 loss = self.label_smoother(outputs, labels, shift_labels=True)
#             else:
#                 loss = self.label_smoother(outputs, labels)
#         else:
#             if isinstance(outputs, dict) and "loss" not in outputs:
#                 raise ValueError(
#                     "The model did not return a loss from the inputs, only the following keys: "
#                     f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
#                 )
#             # We don't use .loss here since the model may return tuples instead of ModelOutput.
#             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
#
#         perplexities = compute_perplexities(outputs['logits'], inputs['labels'])
#         self.batch_perplexities.append(perplexities.detach().tolist())
#         self.state.perplexity = perplexities.mean().detach().item()
#
#         return (loss, outputs) if return_outputs else loss
#
#     def log(self, logs):
#         if self.state.epoch is not None:
#             logs["epoch"] = self.state.epoch
#         if self.args.include_num_input_tokens_seen:
#             logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
#
#         # ------- New code -------
#         if "eval_loss" not in logs.keys():
#             eval_prefix = ""
#             history = self.train_perplexities_history
#         else:
#             eval_prefix = "eval_"
#             history = self.eval_perplexities_history
#
#         logs[f"{eval_prefix}perplexity"] = self.state.perplexity
#         history.append({"step": self.state.global_step,
#                         f"{eval_prefix}batch_perplexities": self.batch_perplexities})
#         # ------- New code ends -------
#
#         output = {**logs, **{"step": self.state.global_step}}
#         self.state.log_history.append(output)
#         self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
