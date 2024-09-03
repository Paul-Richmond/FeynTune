# hepthLlama

---

A pipeline for finetuning LLMs on arXiv abstracts using 
the 🤗 Hugging Face Transformers library. The aim is to 
have a trained model that is able to complete unseen arXiv
abstracts. 

Uses 4-bit quantization to load model and LoRA (i.e. QLoRA) 
to unfreeze a small percentage of model weights.

During training, the loss that is optimzed is the usual cross-entropy loss, 
but we also evaluate perplexity on forward passes of both the train 
and test datasets. In addition, in order to qualitatively measure the 
model's improvement during training, we have included a callback which
generates abstract completions for 5 papers published by the 
hepthLlama creators. 





# Quickstart

---

You will need access to at least one Nvidia GPU with a reasonable amount of RAM.
We have used both QMUL's Apocrita HPC cluster (https://docs.hpc.qmul.ac.uk) and
the Sulis cluster (https://sulis-hpc.github.io), both of which have A100 GPUs 
with 40Gb of RAM.

The following packages are needed (the pip/conda install names are listed):

- transformers
- torch
- wandb
- huggingface_hub
- peft
- flash-attn
- datasets
- bitsandbytes
- accelerate
- hydra-core
- python-dotenv

See `/apocrita/create_env.sh` and `/sulis/llmenvcr.slurm` as examples of
creating environments with the necessary installs.

After creating your environment you will need to copy the file `.env_example`
and rename it to `.env`. Then paste your private Hugging Face and 
Weights & Biases API keys into the renamed file. Several of the models hosted 
on Hugging Face are gated and require access to be granted before use.

The training script is found in `src/finetune.py`. A default 
configuration is set but each time the script is executed a `run_name`
must be specified. This is done using hydra's override syntax e.g.
to set `run_name` to `"run1"` you would use

```bash
python3 src/finetune.py training.training_args_cfg.run_name=run1
```

# Setting a training configuration

---

The pipeline is designed to be flexible and allow for different 
models, tokenizers, datasets, optimizers and training hyperparameters
as well as custom trainers and callbacks.

A lot of this flexibility is enabled by using the [hydra package](https://hydra.cc)
whose key feature is
> the ability to dynamically create a hierarchical configuration
> by composition and override it through config files and the command line.

For example to change the model and tokenizer to be `gemma-2-9b` rather than 
the current default `Llama-3.1-8B` we would run
```bash
python3 src/finetune.py model.model_cfg.name=google/gemma-2-9b \
                        tokenizer.name=google/gemma-2-9b \
                        training.training_args_cfg.run_name=gemma2
```

# Recommencing training from a checkpoint

---

If training is cut short it can be restarted from a saved checkpoint. 
To enable this, the parameter `resume_from_checkpoint` in 
`configs/training/training_args.yaml` should be set to `true` or to 
a local path of a saved checkpoint (for full details, see the documentation for 
`Trainer.train` [here](https://huggingface.co/docs/transformers/main_classes/trainer)).
However, if you want the W&B logging to continue from where it left off, 
you have to in addition modify the `.env` file by adding 

```env
WANDB_RESUME="must"
WANDB_RUN_ID=<run ID>
```

as suggested [here](https://github.com/huggingface/transformers/issues/25032).
The W&B documentation on resuming runs can be found at this 
[link](https://docs.wandb.ai/guides/runs/resuming).

# A W&B hack

---

Unfortunately W&B does not allow for continuous updating of tables during training
which seems to be an oversight given the recent incredible uptake of generative AI.

One way to overcome this is to regularly log a new table using the same key
and then combine the tables in the W&B UI. 

This can be achieved as follows:
In a W&B run workspace go to the Tables panels and click on "Add panel",
choosing "Query panel" from the dropdown which appears. Then paste
```
runs[0].loggedArtifactVersions.map((row, index) => row.file("predictions.table.json"))
```
and hit enter. This will automatically concatenate the tables so that the LLM's generated output 
can be qualitatively assessed by subject experts.

# Inference

---

The script to run inference on a trained model is `src/inference.py`.
The code is designed to take abstracts, form prompts by truncating each abstract,
and then use the model's `generate` method to complete the prompts. 
The finished completions are logged to W&B as a table.
Running the inference script requires a single dataset of abstracts 
and a configuration file.
The defaults used are set in `configs/default_infer.yaml` 
and are currentlly `configs/dataset/full.yaml` and 
`configs/inference/infer.yaml`. 

Since we only need to complete abstracts in the test dataset we can use 
the following hydra override to remove the train dataset split;

`~dataset.splits.train=train`

The configuration file specifies the model, quantization setup, 
generation configuration passed to the model's `generate` method
and the batch size. You can also specify the parameter `wandb_runpath`,
which is where the table of completed abstracts gets logged. 
`wandb_runpath` can be an existing W&B run, say the run associated 
with training the model used for inference.

The following is an example of running the inference script:
```bash
python3 src/inference.py ~dataset.splits.train=train inference.wandb_runpath=llms-for-hepth/huggingface/llama3.1-8B
```

