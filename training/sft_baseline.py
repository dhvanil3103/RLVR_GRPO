"""
training/sft_baseline.py

Supervised fine-tuning baseline on DeepMath-103K using R1 solutions.
Written for TRL 0.28.0.

Why SFT first?
  GRPO needs some correct completions per batch to compute advantages. A fresh
  Qwen2.5-14B-Instruct won't produce our <think>/<answer> format, so rewards
  are always 0 and gradients are useless. Two epochs of SFT gives the model a
  warm start in the right output format before GRPO.

Submit via: sbatch slurm/sft_job.sh
"""

import argparse
import os

from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path",     default="./models/qwen2.5-7b-instruct")
parser.add_argument("--data_dir",       default="./data/processed/sft")
parser.add_argument("--output_dir",     default="./checkpoints/sft")
parser.add_argument("--run_name",       default="deepmath-sft-14b")
parser.add_argument("--epochs",         type=int,   default=1)
parser.add_argument("--batch_size",     type=int,   default=1)
parser.add_argument("--grad_accum",     type=int,   default=8)
parser.add_argument("--lr",             type=float, default=2e-4)
parser.add_argument("--max_seq_length", type=int,   default=8192)
parser.add_argument("--lora_r",         type=int,   default=16)
parser.add_argument("--lora_alpha",     type=int,   default=32)
parser.add_argument("--lora_dropout",   type=float, default=0.05)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
print(f"Loading SFT dataset from {args.data_dir}...")
dataset = load_from_disk(args.data_dir)
train_dataset = dataset["train"]
print(f"  Train: {len(train_dataset):,} examples")

# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

# ---------------------------------------------------------------------------
# SFT config (TRL 0.28.0)
# Warmup steps = ~5% of total steps. With 249K examples, 2 epochs,
# effective batch 32: ~15,600 steps total, so 780 warmup steps.
# ---------------------------------------------------------------------------
sft_config = SFTConfig(
    output_dir=args.output_dir,
    run_name=args.run_name,
    num_train_epochs=args.epochs,
    model_init_kwargs={
        "local_files_only": True,
    },
    per_device_train_batch_size=args.batch_size,
    max_length=args.max_seq_length,
    packing=True,
    gradient_accumulation_steps=args.grad_accum,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    gradient_checkpointing=True,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=143,
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    eval_strategy="no",
    report_to="wandb",
    dataloader_num_workers=4,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
processing_class = AutoTokenizer.from_pretrained(
    args.model_path,
    local_files_only=True,
)
processing_class.padding_side = "right"

# ---------------------------------------------------------------------------
# Trainer (TRL 0.28.0)
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
print("\nStarting SFT training...")
print(f"  Model     : {args.model_path}")
print(f"  Epochs    : {args.epochs}")
print(f"  Eff. batch: {args.batch_size * args.grad_accum * 4} "
      f"(per_device={args.batch_size} x grad_accum={args.grad_accum} x 4 GPUs)")
print(f"  LR        : {args.lr}")
print(f"  Max seq   : {args.max_seq_length}")

# Auto-detect latest checkpoint if it exists
checkpoint_dir = None
if os.path.isdir(args.output_dir):
    checkpoints = [
        os.path.join(args.output_dir, d)
        for d in os.listdir(args.output_dir)
        if d.startswith("checkpoint-")
    ]
    if checkpoints:
        checkpoint_dir = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        print(f"  Resuming from: {checkpoint_dir}")
    else:
        print("  No checkpoint found, starting from scratch.")

trainer.train(resume_from_checkpoint=checkpoint_dir)
trainer.save_model(args.output_dir)
print(f"\nSFT complete. Adapter saved to {args.output_dir}")
print("Next: run sft_sanity_check.py in an interactive session, then submit grpo_job.sh")