"""
training/sft_smoketest.py

Runs  tiny SFT job (50 steps, 50 examples) to verify the full training
pipeline works before committing to a multi-hour full run.

Written for TRL 0.28.0. Key differences from older TRL:
  - `processing_class` instead of `tokenizer` in SFTTrainer
  - `warmup_steps` instead of `warmup_ratio` in SFTConfig
  - `max_seq_length` passed to SFTTrainer, not SFTConfig
  - `local_files_only=True` needed when loading from a local path

Submit via: sbatch slurm/sft_smoketest.sh
"""

import argparse
import os

from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",     default="./models/qwen2.5-14b-instruct")
parser.add_argument("--data_dir",       default="./data/processed/sft")
parser.add_argument("--output_dir",     default="./checkpoints/sft_smoketest")
parser.add_argument("--batch_size",     type=int, default=2)
parser.add_argument("--grad_accum",     type=int, default=4)
parser.add_argument("--max_seq_length", type=int, default=8192)
parser.add_argument("--max_steps",      type=int, default=50)
parser.add_argument("--n_examples",     type=int, default=50)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
print(f"Loading SFT dataset from {args.data_dir}...")
dataset = load_from_disk(args.data_dir)
train_dataset = dataset["train"].select(range(args.n_examples))
print(f"  Using {len(train_dataset)} examples for smoketest")

# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ---------------------------------------------------------------------------
# SFT config (TRL 0.28.0)
# warmup_ratio is removed — use warmup_steps instead.
# max_seq_length does NOT go here, it goes in SFTTrainer.
# ---------------------------------------------------------------------------
sft_config = SFTConfig(
    output_dir=args.output_dir,
    run_name="sft-smoketest",
    max_steps=args.max_steps,
    max_length=args.max_seq_length,       
    model_init_kwargs={"local_files_only": True},
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

# ---------------------------------------------------------------------------
# Tokenizer — TRL 0.28.0 calls this processing_class in SFTTrainer
# local_files_only=True tells transformers not to try HuggingFace Hub
# when given a local directory path
# ---------------------------------------------------------------------------
processing_class = AutoTokenizer.from_pretrained(
    args.model_path,
    local_files_only=True,
)
processing_class.padding_side = "right"

# ---------------------------------------------------------------------------
# Trainer — TRL 0.28.0
# tokenizer is renamed to processing_class
# max_seq_length goes here, not in SFTConfig
# model_init_kwargs passes extra args to from_pretrained for the model
# ---------------------------------------------------------------------------
trainer = SFTTrainer(
    model=args.model_path,
    args=sft_config,
    train_dataset=train_dataset,
    peft_config=lora_config,
    processing_class=processing_class,
)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print("\n===== SFT Smoketest =====")
print(f"  Model       : {args.model_path}")
print(f"  Examples    : {args.n_examples}")
print(f"  Max steps   : {args.max_steps}")
print(f"  Seq length  : {args.max_seq_length}")
print(f"  Eff. batch  : {args.batch_size} x 4 GPUs x {args.grad_accum} grad_accum = {args.batch_size * 4 * args.grad_accum}")
print("=========================\n")

trainer.train()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
log_history = trainer.state.log_history
losses = [entry["loss"] for entry in log_history if "loss" in entry]

print(f"\n===== Smoketest Results =====")
if len(losses) >= 2:
    print(f"  First loss : {losses[0]:.4f}")
    print(f"  Final loss : {losses[-1]:.4f}")
    if losses[-1] < losses[0]:
        print("  Loss is decreasing. Pipeline is working correctly.")
        print("  You are safe to submit the full sft_job.sh.")
    else:
        print("  WARNING: Loss did not decrease. Check dataset formatting and learning rate.")
else:
    print("  Not enough log entries to compare. Check the output above for errors.")

trainer.save_model(args.output_dir)
print(f"\n  Checkpoint saved to {args.output_dir}")
print("  You can delete this directory after confirming the smoketest passed.")a