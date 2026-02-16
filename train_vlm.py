import os
import torch
from datasets import load_from_disk, load_dataset, concatenate_datasets
from transformers import AutoModelForImageTextToText, AutoConfig, set_seed, AutoProcessor
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser, get_peft_config

if __name__ == "__main__":
    set_seed(41)
    
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True, **model_kwargs
    )
    
    if os.path.exists(script_args.dataset_name):
        dataset = load_from_disk(script_args.dataset_name)
    else:
        dataset = load_dataset(script_args.dataset_name, split="train")

    dataset = dataset.shuffle(41)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)