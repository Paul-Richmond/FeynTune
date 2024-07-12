from transformers import Trainer

from .metrics import compute_perplexities


class PerplexityTrainer(Trainer):
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

        self.state.batch_perplexities = None
        self.state.original_loss = None

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        perplexities = compute_perplexities(outputs['logits'], inputs['labels'])

        self.state.batch_perplexities = perplexities.tolist()
        self.state.original_loss = outputs.loss.item()

        loss = perplexities.mean()

        return (loss, outputs) if return_outputs else loss

    def log(self, logs):
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        if not self.control.should_evaluate:
            logs = {"original_loss": self.state.original_loss, "perplexities": self.state.batch_perplexities}
            self.log(logs)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
