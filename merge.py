import argparse
import torch

from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import AutoTokenizer

def merge(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    try:
        print(f"Loading tokenizer from adapter dir: {args.adapter_dir}")
        adapter_tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    except Exception:
        print("Tokenizer not found in adapter dir, using base model tokenizer.")
        adapter_tokenizer = tokenizer

    model.resize_token_embeddings(len(adapter_tokenizer))

    model = PeftModel.from_pretrained(model, args.adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.merged_dir)
    adapter_tokenizer.save_pretrained(args.merged_dir)

    print(f"Merged model saved: {args.merged_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Adapters")
    parser.add_argument("--base_model", type=str, help="Base Model to Merge")
    parser.add_argument("--adapter_dir", type=str, help="Directory containing the adapters to merge")
    parser.add_argument("--merged_dir", type=str, help="Output directory for the merged model")
    args = parser.parse_args()
    merge(args)