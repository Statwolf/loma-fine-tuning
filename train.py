import argparse
import json
import wandb
import torch
 
from datasets import Dataset
from pathlib import Path
from typing import Optional
from prompts import SYS_PROMPT_LOMA
from utils import get_dataset
from unsloth import FastLanguageModel
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AddedToken, AutoModelForCausalLM, EarlyStoppingCallback


def formatting_func(examples, tokenizer):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []

    for instruction, output in zip(instructions, outputs):
        messages = [
            {"role": "system", "content": SYS_PROMPT_LOMA},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        texts.append(text)
    return {"text": texts}

class TokenHandler:
    def __init__(self, tokens_path: str):
        self.tokens_path = tokens_path
    
    def load_tokens(self):
        with open(self.tokens_path, "r", encoding="utf-8") as f:
            keywords = [json.loads(line)["keyword"] for line in f]
        return [AddedToken(key, normalized=False, special=False) for key in keywords]
    
    def add_tokens(self, model, tokenizer):
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        if actual_tokenizer.eos_token == "<EOS_TOKEN>" or actual_tokenizer.eos_token not in actual_tokenizer.get_vocab():
            actual_tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})

        if actual_tokenizer.pad_token is None or actual_tokenizer.pad_token == "<EOS_TOKEN>":
            actual_tokenizer.pad_token = actual_tokenizer.eos_token
        actual_tokenizer.padding_side = "right"

        old_vocab_size = len(actual_tokenizer)
        n_tokens_added = actual_tokenizer.add_tokens(self.load_tokens())
        print(f"{n_tokens_added} NEW TOKENS ADDED.\n")

        model.resize_token_embeddings(len(actual_tokenizer))
        self._init_new_embeddings(model, old_vocab_size)
        tokenizer.save_pretrained(Path(self.tokens_path).parent)

        return list(range(old_vocab_size, old_vocab_size + n_tokens_added))
    
    @staticmethod
    def _init_new_embeddings(model, old_vocab_size: int):
        input_emb = model.get_input_embeddings().weight.data
        output_emb = model.get_output_embeddings().weight.data

        with torch.no_grad():
            input_emb[old_vocab_size:] = input_emb[:old_vocab_size].mean(dim=0).to(input_emb.dtype)
            output_emb[old_vocab_size:] = output_emb[:old_vocab_size].mean(dim=0).to(output_emb.dtype)

class ModelLoader:
    def __init__(self, args):
        self.args = args
    
    def load(self):
        if self.args.unsloth:
            return self._load_unsloth()
        return self._load_hf()

    def _load_unsloth(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.args.model,
            max_seq_length=self.args.max_length,
            dtype=torch.bfloat16,
            load_in_4bit=bool(self.args.quantize)
        )

        if self.args.add_tokens:
            TokenHandler(self.args.tokens_path).add_tokens(model, tokenizer)
        
        model = FastLanguageModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            # target_modules="all-linear",
            lora_dropout=self.args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            modules_to_save=["embed_tokens", "lm_head"] if self.args.add_tokens else None
        )

        return model, tokenizer, None

    def _load_hf(self):        
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        model.enable_input_require_grads()
        tokenizer = AutoTokenizer.from_pretrained(self.args.model)
 
        if self.args.add_tokens:
            TokenHandler(self.args.tokens_path).add_tokens(model, tokenizer)
 
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules="all-linear",
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["embed_tokens", "lm_head"] if self.args.add_tokens else None,
        )
 
        return model, tokenizer, lora_config

class FineTuning:
    def __init__(self, args):
        self.args = args

    def _prepare_dataset(self):
        train_dataset = get_dataset(self.args.dataset).shuffle(seed=42)
        test_dataset = get_dataset(self.args.test_dataset)

        return train_dataset, test_dataset

    def _build_training_args(self, run_name: str):
        return SFTConfig(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.epochs,
            learning_rate=self.args.lr,
            logging_steps=5,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.grad_acc,
            gradient_checkpointing=not self.args.unsloth,
            max_grad_norm=self.args.clipping,
            dataset_text_field="text" if not self.args.instruct else None,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            bf16=True,
            max_length=self.args.max_length,
            packing=not self.args.instruct,
            report_to="wandb",
            run_name=run_name
        )

    def _save(self, trainer, tokenizer):
        if self.args.unsloth:
            adapters_dir = self.args.output_dir + "_adapters"
            trainer.model.save_pretrained(adapters_dir)
            tokenizer.save_pretrained(adapters_dir)
            print(f"Adapters saved to: {adapters_dir}")
 
            if not self.args.quantize:
                merged = trainer.model.merge_and_unload()
                merged.save_pretrained(self.args.output_dir)
                tokenizer.save_pretrained(self.args.output_dir)
                print(f"Merged model saved to: {self.args.output_dir}")
            else:
                print("Training done in 4bit: to get a GGUF convertible model, reload the base model in fp16/bf16, load the adapters and merge separately.")
        else:
            trainer.save_model(self.args.output_dir)
            tokenizer.save_pretrained(self.args.output_dir)

    def run(self):
        run_name = f"{self.args.output_dir.split('/')[-1]}-loma-fine-tuning".replace("/", "-")
 
        wandb.init(
            project="loma-fine-tuning",
            entity="loma-statwolf",
            name=run_name,
            config=vars(self.args)
        )
 
        train_dataset, test_dataset = self._prepare_dataset()
        model, tokenizer, lora_config = ModelLoader(self.args).load()
 
        actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        if actual_tokenizer.eos_token == "<EOS_TOKEN>" or actual_tokenizer.eos_token not in actual_tokenizer.get_vocab():
            actual_tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
 
        if actual_tokenizer.pad_token is None or actual_tokenizer.pad_token == "<EOS_TOKEN>":
            actual_tokenizer.pad_token = actual_tokenizer.eos_token
        actual_tokenizer.padding_side = "right"
 
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=lora_config,
            processing_class=actual_tokenizer,
            args=self._build_training_args(run_name),
            formatting_func=lambda x: formatting_func(x, actual_tokenizer)["text"] if self.args.instruct else None,
            data_collator=None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
 
        trainer.train()
        self._save(trainer, actual_tokenizer)
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tuning")
    parser.add_argument("--model", type=str, help="Base Model to Fine-Tune")
    parser.add_argument("--dataset", type=str, help="Dataset Path (JSONL file)")
    parser.add_argument("--test_dataset", type=str, help="Test Dataset Path (JSONL file)")
    parser.add_argument("--output_dir", type=str, help="Model output directory")
    parser.add_argument("--unsloth", action="store_true", help="Activate Unsloth")
    parser.add_argument("--quantize", action="store_true", help="Activate Quanization")
    parser.add_argument("--add_tokens", action="store_true", help="Add tokens")
    parser.add_argument("--instruct", action="store_true", help="Instruction Dataset")
    parser.add_argument("--tokens_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--clipping", type=float, default=0.4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    args = parser.parse_args()

    if args.add_tokens and not args.tokens_path:
        raise ValueError("--tokens_path is mandatory when --add_tokens is enabled")
 
    if args.instruct and not args.test_dataset:
        raise ValueError("--test_dataset is mandatory when --instruct is enabled")
 
    FineTuning(args).run()