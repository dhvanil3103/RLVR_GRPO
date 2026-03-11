# merge_checkpoint.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# This is your original SFT merged model - the base for run 15
BASE_PATH = "./checkpoints/grpo_v14_merged"

# This is the final saved LoRA from run 15
LORA_PATH = "./checkpoints/grpo_v15"

# This is where the new merged model will be saved for final evaluation
NEW_BASE_PATH = "./checkpoints/grpo_final_merged"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)

print("Merging LoRA into base weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {NEW_BASE_PATH}...")
model.save_pretrained(NEW_BASE_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
tokenizer.save_pretrained(NEW_BASE_PATH)

print("Done.")