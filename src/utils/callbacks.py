import gc
import re

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.integrations import WandbCallback


class AbstractCompleter:
    """
    A class to handle abstract completion using a language model.

    Attributes:
        model (transformers.PreTrainedModel): The pre-trained model used for generating completions.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
        dataset (datasets.Dataset): The dataset to be processed.
        batch_size (int, optional): The number of samples to process in a batch. Defaults to 16.
        generation_config (dict, optional): Configuration parameters for text generation.
                                            Defaults to a predefined configuration.
        col_to_tokenize (str, optional): The name of the column in `dataset` to tokenize.
                                         Defaults to 'abstract'.
    """

    def __init__(self, model, tokenizer, dataset, batch_size=None, generation_config=None, col_to_tokenize=None):
        """
        Initializes the AbstractCompleter with the provided model, tokenizer, dataset, and optional parameters.

        Args:
            model (transformers.PreTrainedModel): The pre-trained model used for generating completions.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
            dataset (datasets.Dataset): The dataset to be processed.
            batch_size (int, optional): The number of samples to process in a batch. Defaults to 16.
            generation_config (dict, optional): Configuration parameters for text generation.
                                                Defaults to a predefined configuration.
            col_to_tokenize (str, optional): The name of the column to tokenize. Defaults to 'abstract'.
        """
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.original_pad_side = tokenizer.padding_side

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
                "max_time": 60
            }
        else:
            self.generation_config = generation_config

        if self.generation_config.get("pad_token_id", None) is None:
            self.generation_config.update({"pad_token_id": self.tokenizer.pad_token_id})

        self.label = 'abstract' if col_to_tokenize is None else col_to_tokenize

    def predict(self, batch):
        """
        Generates model predictions for a batch of input texts.

        Args:
            batch (dict): A batch of data containing 'prompt' keys with texts to generate predictions for.

        Returns:
            dict: The updated batch with model predictions added under the 'predictions' key.
        """
        texts = batch['prompt']
        self.tokenizer.padding_side = 'left'  # inference needs left padding
        # clear some memory
        torch.cuda.empty_cache()
        gc.collect()
        model_inputs = self.tokenizer(texts,
                                      padding='longest',
                                      pad_to_multiple_of=8,
                                      add_special_tokens=False,
                                      return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**model_inputs, **self.generation_config).to("cpu")
        batch["predictions"] = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.tokenizer.padding_side = self.original_pad_side
        return batch

    def split_abstracts(self, example):
        """
        Splits an abstract into a prompt and ground truth.

        The prompt is created from the first half (or slightly more) of the sentences in the abstract,
        and the ground truth is the remaining sentences.
        If there is only a single sentence then we split instead on spaces to get, roughly, words
        and take the first half (or slightly more) of these.

        Args:
            example (dict): A dictionary containing the 'abstract' text to be split.

        Returns:
            dict: A dictionary with 'prompt' and 'y_true' keys containing the split abstract parts.
        """
        text = example[self.label]
        # Split the abstract into sentences (i.e. text sequences which end with any of .!? and a space)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Calculate the split point
        total_sentences = len(sentences)
        if total_sentences > 1:  # more than 1 sentence so can split
            split_point = (total_sentences + 1) // 2  # Ensures the prompt has >= number of sentences than y_true
            # Join the sentences back into two parts
            prompt = ' '.join(sentences[:split_point])
            y_true = ' '.join(sentences[split_point:])
        else:  # only a single sentence so split on words (latex commands between $$ might get split)
            words = text.split()
            total_words = len(words)
            split_point = (total_words + 1) // 2  # Ensures the prompt has >= number of sentences than y_true
            # Join the sentences back into two parts
            prompt = ' '.join(words[:split_point])
            y_true = ' '.join(words[split_point:])
        return {'prompt': prompt, 'y_true': y_true}

    def parse_y_pred(self, example):
        """
        Extracts the predicted text from the generated predictions.

        Args:
            example (dict): A dictionary containing 'prompt' and 'predictions' keys.

        Returns:
            dict: A dictionary with 'y_pred' key containing the generated output without the prompt.
        """
        len_prompt = len(example['prompt'])
        y_pred = example['predictions'][len_prompt:]
        return {'y_pred': y_pred}

    def get_predictions(self):
        """
        Processes the dataset to generate and parse predictions.

        The dataset is first split into prompts and ground truths, then predictions are generated
        for each prompt, and finally, the predictions are parsed.

        Returns:
            datasets.Dataset: The processed dataset with predictions included.
        """
        self.dataset = self.dataset.map(self.split_abstracts,
                                        batched=False,
                                        desc='Splitting abstracts')
        self.dataset = self.dataset.map(self.predict,
                                        batched=True,
                                        batch_size=self.batch_size,
                                        desc='Generating model output')
        self.dataset = self.dataset.map(self.parse_y_pred,
                                        batched=False,
                                        desc='Parsing y_pred')
        return self.dataset


class SimilarityScorer:
    """
    A class for computing similarity scores between pairs of text using a pre-trained model.

    This class uses a given model and tokenizer to compute embeddings for text inputs,
    and then calculates cosine similarity between pairs of embeddings.
    """

    def __init__(self, model, tokenizer, batch_size):
        """
        Initialize the SimilarityScorer.

        Args:
            model: The pre-trained model to use for computing embeddings.
            tokenizer: The tokenizer associated with the model.
            batch_size (int): The number of examples to process in each batch.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer.padding_side = 'right'  # pad right as we might truncate in get_embeddings
        self.tokenizer.truncation_side = 'right'  # make sure this is consistent with padding_side
        self.max_length = tokenizer.model_max_length

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def mean_pooling(self, model_output, attention_mask):
        """
        Compute the mean pooling of the model's output.

        Code taken from https://huggingface.co/sentence-transformers/all-mpnet-base-v2

        Args:
            model_output: The output of the model.
            attention_mask: The attention mask for the input.

        Returns:
            torch.Tensor: The mean-pooled embeddings.
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, batch):
        """
        Compute embeddings for a batch of text inputs.

        Args:
            batch (list): A list of text inputs.

        Returns:
            torch.Tensor: The computed embeddings for the batch.
        """
        encoded_input = self.tokenizer(batch,
                                       padding=True,
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        return batch_embeddings

    def get_batches(self, examples):
        """
        Split a list of examples into batches.

        Args:
            examples (list): The list of examples to split.

        Returns:
            list: A list of batches, where each batch is a sublist of examples.
        """
        return [examples[i:i + self.batch_size] for i in range(0, len(examples), self.batch_size)]

    def get_similarities(self, x, y):
        """
        Compute similarity scores between pairs of texts.

        Args:
            x (list): The first list of texts.
            y (list): The second list of texts.

        Returns:
            list: A list of similarity scores for each pair of texts in x and y.

        Raises:
            AssertionError: If the lengths of x and y are not equal.
        """
        assert len(x) == len(y)
        x_batches = self.get_batches(x)
        y_batches = self.get_batches(y)

        all_similarities = []

        with torch.no_grad():
            for idx in tqdm(range(len(x_batches)), desc=f'Computing similarity scores'):
                x_embeddings = self.get_embeddings(x_batches[idx])
                y_embeddings = self.get_embeddings(y_batches[idx])
                similarities = self.cos(x_embeddings, y_embeddings).tolist()
                all_similarities.extend(similarities)
                torch.cuda.empty_cache()
                gc.collect()

        return all_similarities


class SemsScore(SimilarityScorer):
    """
    A class for calculating the SemSCore metric described in https://arxiv.org/pdf/2401.17072.

    This class inherits from SimilarityScorer and initializes with a specific model
    for sentence embeddings.

    Attributes:
        max_length (int): The maximum sequence length for input tokenization.
    """
    def __init__(self):
        semscore_model = "sentence-transformers/all-mpnet-base-v2"
        model = AutoModel.from_pretrained(semscore_model)
        tokenizer = AutoTokenizer.from_pretrained(semscore_model)
        batch_size = 256  # fits on a single P100 GPU with 16Gb VRAM
        super().__init__(model=model, tokenizer=tokenizer, batch_size=batch_size)
        self.max_length = 384  # See the model card at https://huggingface.co/sentence-transformers/all-mpnet-base-v2


class GenCallback(WandbCallback):
    """
    A callback for logging model predictions on a fixed dataset containing 5 abstracts
    during evaluation with Weights & Biases.

    Attributes:
        generation_config (dict or None): Configuration for text generation.
        dataset (Dataset): Dataset used for generating predictions.
        add_prompts (bool): Flag indicating whether to include prompts in logs.
    """

    def __init__(self, generation_config=None):
        """
        Initializes the GenCallback instance.

        Args:
            generation_config (dict or None): Configuration for text generation. Defaults to None.
        """
        super().__init__()
        self.generation_config = generation_config
        self.dataset = load_dataset("LLMsForHepth/arxiv_hepth_first_overfit").get('train')
        self.add_prompts = True

    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """
        Handles the evaluation phase by generating and logging predictions.

        Args:
            args (Namespace): Arguments passed to the evaluation.
            state (TrainerState): Current state of the training process.
            control (TrainerControl): Controls for managing the training loop.
            model (transformers.PreTrainedModel or None): The model being evaluated. Defaults to None.
            tokenizer (PreTrainedTokenizer or None): The tokenizer used with the model. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self._gen_and_log(state, model, tokenizer)

    def _gen_and_log(self, state, model, tokenizer):
        """
        Generates predictions and logs them to Weights & Biases.

        Args:
            state (transformers.TrainerState): Current state of the training process.
            model (transformers.PreTrainedModel): The model used for generating predictions.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used with the model.
        """
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

            # If completing for 1st time add prompts to new_table
            if self.add_prompts:
                new_table.add_data("Prompt", *completions['prompt'])
                self.add_prompts = False

            new_table.add_data(str(state.global_step), *completions['predictions'])
            self._wandb.log({f"predictions": new_table}, commit=False)

        if switch_back_to_train:
            model.train()
