import json
from datasets import load_dataset
from transformers import TrainerCallback, pipeline


class GenCallback(TrainerCallback):
    def __init__(self):
        self.dataset = load_dataset("LLMsForHepth/arxiv_hepth_first_overfit")

    def on_train_begin(self, args, state, control, model=None, tokenizer=None, **kwargs):
        self._gen_and_save(state, model, tokenizer)

    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
        self._gen_and_save(state, model, tokenizer)

    def _gen_and_save(self, state, model, tokenizer):
        if (model is not None) and (tokenizer is not None):
            generation_config = model.generation_config
            generation_config.pad_token_id = tokenizer.eos_token_id

            generator = pipeline(task="text-generation",
                                 # do I need to unwrap the model as it a PEFT model
                                 model=model,
                                 tokenizer=tokenizer,
                                 )

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                tokenizer.convert_tokens_to_ids("<|end_of_text|>")
            ]
            full_dict = {}
            for idx, abstract in enumerate(self.dataset['train']['abstract']):
                prompt = abstract[:len(abstract) // 2]
                # A full list of arguments that can be passed can be found in the
                # GenerationConfig class in transformers/generation/configuration_utils.py
                generated_text = generator(prompt,
                                           do_sample=True,
                                           max_length=720,
                                           max_time=120,
                                           eos_token_id=terminators)

                paper_dict = {"Full abstract": abstract,
                              "Prompted abstract": prompt,
                              "Generated abstract": generated_text[0]['generated_text']
                              }

                full_dict.update({f"Paper{idx + 1}": paper_dict})
            # TODO: this gets saved to the repo directory, change it to hydra run directory
            json_file = f"Generated_abstracts_step_{state.global_step}.json"
            with open(json_file, "w") as f:
                json.dump(full_dict, f)
        else:
            print("Skipping generation of abstract because no model is provided")
