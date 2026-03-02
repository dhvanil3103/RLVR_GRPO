"""
training/grpo_train.py

GRPO (Group Relative Policy Optimization) training on DeepMath-103K.
Trains on top of the merged SFT checkpoint produced by merge_adapter.py.

Why start from the merged SFT model?
  GRPO computes a KL penalty against a frozen reference model. TRL handles
  this internally but needs a single merged model, not a base + adapter split.
  The merge_adapter.py script handles this — run it once before this job.

Submit via: sbatch slurm/grpo_smoketest.sh  (always run smoketest first)
            sbatch slurm/grpo_job.sh         (full run after smoketest passes)
"""

import argparse
import os
import sys

from datasets import load_from_disk
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers.trainer_utils import get_last_checkpoint
# ---------------------------------------------------------------------------
# Reward functions — must be imported from the same directory
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reward_functions import accuracy_reward, format_reward, efficiency_reward

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path",            default="./checkpoints/sft_merged")
parser.add_argument("--data_dir",              default="./data/processed/grpo_20k")
parser.add_argument("--output_dir",            default="./checkpoints/grpo_v1")
parser.add_argument("--run_name",              default="deepmath-grpo-7b-v1")
parser.add_argument("--max_steps",             type=int,   default=-1)
parser.add_argument("--per_device_batch_size", type=int,   default=1)
parser.add_argument("--grad_accum",            type=int,   default=4)
parser.add_argument("--num_generations",       type=int,   default=4)
parser.add_argument("--lr",                    type=float, default=5e-6)
parser.add_argument("--max_completion_length", type=int,   default=12288)
parser.add_argument("--warmup_steps",          type=int,   default=10)
parser.add_argument("--beta",                  type=float, default=0.04)
parser.add_argument("--lora_r",                type=int,   default=16)
parser.add_argument("--lora_alpha",            type=int,   default=32)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
print(f"Loading GRPO dataset from {args.data_dir}...")
dataset    = load_from_disk(args.data_dir)
train_data = dataset
print(f"  Train: {len(train_data):,} examples")

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
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ---------------------------------------------------------------------------
# GRPO config
#
# Key hyperparameter notes:
#
#   learning_rate: 5e-6 — much lower than SFT (2e-4). GRPO is very sensitive
#     to LR. The most common cause of early training collapse is using an
#     SFT-scale LR here.
#
#   num_generations: 8 — number of completions sampled per prompt to compute
#     group-relative advantages. Higher G gives better advantage estimation
#     and more stable gradients. With 4x A100 80GB and the 7B model, G=8 is
#     well within VRAM budget. If you hit OOM, drop to G=4.
#
#   max_completion_length: 4096 — DeepMath requires long reasoning chains.
#     1024 caused silent truncation in the sanity check. 4096 gives the model
#     room to finish its <think> block and write <answer>. If you see reward
#     being 0 for all completions, check this first.
#
#   beta: 0.04 — KL penalty coefficient. Controls how far the model is allowed
#     to drift from the reference (frozen SFT) model. If KL spikes above 10
#     in W&B, increase beta to 0.1. If rewards plateau early, try reducing to
#     0.01.
#
#   max_grad_norm: 0.1 — tighter than SFT (1.0). GRPO gradients can be large
#     and spiky. 0.1 keeps training stable. If loss diverges, try 0.01.
#
# Effective batch size = per_device_batch_size * num_gpus * grad_accum
#                      = 2 * 4 * 4 = 32 prompts per optimizer step
#                      = 32 * 8 = 256 completions evaluated per step
# ---------------------------------------------------------------------------
grpo_config = GRPOConfig(
    output_dir=args.output_dir,
    run_name=args.run_name,
    ddp_find_unused_parameters=False,
    # Optimizer
    use_liger_kernel=True,
    optim="paged_adamw_8bit",
    learning_rate=args.lr,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    max_grad_norm=0.1,

    # Schedule — warmup_ratio is deprecated, use warmup_steps
    lr_scheduler_type="cosine",
    warmup_steps=args.warmup_steps,

    # Batch
    per_device_train_batch_size=args.per_device_batch_size,
    gradient_accumulation_steps=args.grad_accum,
    num_generations=args.num_generations,

    # Sequence lengths
    max_completion_length=args.max_completion_length,

    # KL penalty
    beta=args.beta,

    # Duration
    ignore_data_skip=False,
    num_train_epochs=1,
    max_steps=args.max_steps,   # -1 = full epoch. Set to 50 for smoketest.

    # Precision
    bf16=True,

    # gradient_checkpointing_kwargs is critical for LoRA compatibility.
    # use_reentrant=False preserves the computation graph through LoRA layers.
    # Without this, the backward pass crashes with "does not require grad".
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Logging and saving
    logging_steps=1,
    save_strategy="steps",
    save_steps=5,
    save_total_limit=5,

    # W&B
    report_to="wandb",

    # remove_unused_columns=False is required — the dataset has a ground_truth
    # column that is not a model input but IS needed by the reward functions.
    # If this is True, TRL will drop ground_truth before calling reward funcs.
    remove_unused_columns=False,

    dataloader_num_workers=4,
    model_init_kwargs={"local_files_only": True},
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
trainer = GRPOTrainer(
    model=args.model_path,
    reward_funcs=[accuracy_reward, format_reward, efficiency_reward],
    args=grpo_config,
    train_dataset=train_data,
    peft_config=lora_config,
)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print("\nStarting GRPO training...")
print(f"  Model              : {args.model_path}")
print(f"  LR                 : {args.lr}")
print(f"  Num generations    : {args.num_generations}")
print(f"  Max completion len : {args.max_completion_length}")
print(f"  Beta (KL penalty)  : {args.beta}")
print(f"  Warmup steps       : {args.warmup_steps}")
print(f"  Effective batch    : {args.per_device_batch_size} x 4 GPUs x {args.grad_accum} grad_accum = "
      f"{args.per_device_batch_size * 4 * args.grad_accum} prompts/step "
      f"({args.per_device_batch_size * 4 * args.grad_accum * args.num_generations} completions/step)")

# Resume from checkpoint if one exists

checkpoint_dir = None
if os.path.isdir(args.output_dir):
    checkpoint_dir = get_last_checkpoint(args.output_dir)
    if checkpoint_dir:
        print(f"  Resuming from      : {checkpoint_dir}")
    else:
        print("  No checkpoint found, starting from scratch.")

trainer.train(resume_from_checkpoint=checkpoint_dir)
trainer.save_model(args.output_dir)

print(f"\nGRPO training complete. Adapter saved to {args.output_dir}")
print("Next steps:")
print("  1. Check W&B for reward curves.")
print("  2. Run eval/run_eval.py to compare SFT vs GRPO accuracy.")
print("  3. Run training/merge_adapter.py pointing to this output for the final merged model.")