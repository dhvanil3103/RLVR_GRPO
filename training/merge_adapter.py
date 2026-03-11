"""
training/merge_adapter.py

Merges a LoRA adapter into the base model weights and saves the result
as a single standalone model. Run this after SFT completes and before
starting GRPO, since GRPOTrainer expects a full merged model.

Also run it again after GRPO to produce the final model for evaluation
and serving.

Usage:
    # After SFT (run once in an interactive session before GRPO):
    python training/merge_adapter.py \
        --base_model_path ./models/qwen2.5-7b-instruct \
        --adapter_path ./checkpoints/sft \
        --output_path ./checkpoints/sft_merged

    # After GRPO (for final eval and serving):
    python training/merge_adapter.py \
        --base_model_path ./checkpoints/sft_merged \
        --adapter_path ./checkpoints/grpo_v1 \
        --output_path ./final_model
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", default="./models/qwen2.5-7b-instruct")
parser.add_argument("--adapter_path",    default="./checkpoints/sft")
parser.add_argument("--output_path",     default="./checkpoints/sft_merged")
args = parser.parse_args()

print(f"Loading base model from {args.base_model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

print(f"Loading adapter from {args.adapter_path}...")
model = PeftModel.from_pretrained(model, args.adapter_path)

print("Merging adapter into base model weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {args.output_path}...")
model.save_pretrained(args.output_path)

# Save tokenizer alongside the model so the output directory is self-contained
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, local_files_only=True)
tokenizer.save_pretrained(args.output_path)

print(f"\nDone. Merged model saved to {args.output_path}")
print("You can now point grpo_train.py --model_path to this directory.")