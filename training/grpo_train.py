"""
training/grpo_train.py

GRPO (Group Relative Policy Optimization) training on DeepMath-103K.
Trains on top of a fully merged base model (SFT merged, or a previous GRPO
run merged via merge_checkpoint.py).

WHY FRESH LORA EVERY RUN (IMPORTANT — READ THIS):
  TRL's GRPOTrainer does NOT keep a separate reference model in memory when
  you use PEFT/LoRA. Instead it calls disable_adapter() on your policy model
  to get reference logprobs. This means:

    reference logprobs = base model weights (LoRA disabled)
    policy logprobs    = base model weights + LoRA weights (LoRA enabled)

  If you load a pre-trained LoRA from a previous run, the policy and reference
  are DIFFERENT from step 1. KL divergence is huge immediately, beta * KL
  crushes the reward signal, and training produces 0.01-0.1 rewards with
  accuracy stuck at zero. This is NOT the model learning slowly — it is
  broken training.

  The correct workflow is:
    1. After each run, merge the LoRA into the base using merge_checkpoint.py
    2. Pass the merged model as --model_path for the next run
    3. Always start with a FRESH LoRA (zero weights)
    4. Now reference (adapter disabled) == policy at step 0. KL = 0. Works.

  Never load a pre-trained LoRA adapter into this script. The checkpoint
  detection logic below only resumes within the SAME run (same output_dir).

CHECKPOINT RESUME (within the same run):
  TRL's resume_from_checkpoint has persistent bugs with optimizer state format
  mismatches across torch versions. Instead this script loads the adapter
  directly via PeftModel.from_pretrained and slices the dataset to skip
  already-consumed examples. The optimizer starts fresh but model weights and
  step position are correctly restored.

  This ONLY applies when resuming a run that was interrupted mid-way.
  The checkpoint LoRA being loaded here is from the SAME run, so the base
  model it was trained on top of is identical to args.model_path. Reference
  and policy divergence is expected and manageable because it accumulated
  gradually, not all at once from a previous run.

Submit via: sbatch slurm/grpo_smoketest.sh  (always run smoketest first)
            sbatch slurm/grpo_job.sh         (full run after smoketest passes)

Workflow across multiple runs:
  Run 1:  --model_path sft_merged        --output_dir grpo_v1
          → trains, saves grpo_v1/checkpoint-N
          → after run: python merge_checkpoint.py --base sft_merged
                                                  --lora grpo_v1
                                                  --out  grpo_v1_merged

  Run 2:  --model_path grpo_v1_merged    --output_dir grpo_v2
          → fresh LoRA on grpo_v1_merged, trains cleanly
          → after run: merge again → grpo_v2_merged

  Run 3:  --model_path grpo_v2_merged    --output_dir grpo_v3
  ...and so on.
"""

import argparse
import os
import sys
import json
import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# Reward functions — must be imported from the same directory
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reward_functions import accuracy_reward, format_reward, efficiency_reward

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path",            default="./checkpoints/grpo_v14_merged",
                    help="Path to a FULLY MERGED base model. Never pass a LoRA adapter here.")
parser.add_argument("--data_dir",              default="./data/processed/grpo_20k")
parser.add_argument("--output_dir",            default="./checkpoints/grpo_v15")
parser.add_argument("--run_name",              default="deepmath-grpo-7b-v2")
parser.add_argument("--max_steps",             type=int,   default=-1)
parser.add_argument("--per_device_batch_size", type=int,   default=1)
parser.add_argument("--grad_accum",            type=int,   default=2)
parser.add_argument("--num_generations",       type=int,   default=4)
parser.add_argument("--lr",                    type=float, default=5e-6)
parser.add_argument("--max_completion_length", type=int,   default=4096)
parser.add_argument("--warmup_steps",          type=int,   default=10)
parser.add_argument("--beta",                  type=float, default=0.04)
parser.add_argument("--lora_r",                type=int,   default=16)
parser.add_argument("--lora_alpha",            type=int,   default=32)
parser.add_argument("--examples_to_skip", type=int, default=9380,
                    help="Number of dataset examples to skip. Set manually "
                         "based on how many examples previous runs consumed. "
                         "Formula: sum of (steps * per_device_batch * 4 GPUs * grad_accum) "
                         "across all previous runs.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Detect checkpoint WITHIN THIS RUN ONLY
# This is for resuming an interrupted run, not for loading a previous run's LoRA.
# If output_dir has a checkpoint from this same run, we resume from it.
# If you are starting a NEW run, output_dir should be a fresh/empty directory.
# ---------------------------------------------------------------------------
steps_completed = 0
checkpoint_path = None

if os.path.isdir(args.output_dir):
    checkpoints = [
        os.path.join(args.output_dir, d)
        for d in os.listdir(args.output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))
    ]
    if checkpoints:
        checkpoint_path = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        state_file = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(state_file):
            with open(state_file) as f:
                steps_completed = json.load(f).get("global_step", 0)
        print(f"  Found checkpoint   : {checkpoint_path}")
        print(f"  Steps completed    : {steps_completed}")
        print(f"  Resuming this run from step {steps_completed}")

# ---------------------------------------------------------------------------
# Dataset — slice to skip already-consumed examples
# ---------------------------------------------------------------------------
print(f"\nLoading GRPO dataset from {args.data_dir}...")
dataset    = load_from_disk(args.data_dir)
train_data = dataset
print(f"  Total examples     : {len(train_data):,}")

if args.examples_to_skip > 0:
    if args.examples_to_skip >= len(train_data):
        print("  WARNING: examples_to_skip >= dataset size. Starting from beginning.")
    else:
        train_data = train_data.select(range(args.examples_to_skip, len(train_data)))
        print(f"  Skipped first      : {args.examples_to_skip} examples (previous runs)")
        print(f"  Remaining          : {len(train_data):,} examples")
else:
    print("  No examples skipped (fresh start)")
# ---------------------------------------------------------------------------
# Load base model
# args.model_path must always be a fully merged model (no adapter).
# ---------------------------------------------------------------------------
print(f"\nLoading base model from {args.model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

# ---------------------------------------------------------------------------
# Apply LoRA
#
# TWO CASES:
#
# Case 1 — Resuming an interrupted run (checkpoint_path is not None):
#   A checkpoint exists in output_dir from this same run. We load that LoRA
#   to restore the model state. This is safe because the checkpoint was trained
#   on top of args.model_path, so base weights match. TRL's disable_adapter()
#   will return args.model_path weights as reference, which is the correct
#   reference for this run.
#
# Case 2 — Starting fresh (checkpoint_path is None):
#   No checkpoint in output_dir. We apply a zero-initialized fresh LoRA.
#   Reference (adapter disabled) == Policy (adapter enabled with zero weights).
#   KL = 0 at step 0. This is the correct and expected starting state.
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

if checkpoint_path is not None:
    print(f"  Resuming: loading LoRA from checkpoint: {checkpoint_path}")
    model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
        is_trainable=True,
    )
    print("  Adapter loaded with gradients enabled (resume within same run)")
else:
    print("  Fresh start: applying zero-initialized LoRA")
    model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# GRPO config
#
# Key hyperparameter notes:
#
#   learning_rate: 5e-6 — much lower than SFT (2e-4). GRPO is very sensitive
#     to LR. The most common cause of early training collapse is an SFT-scale LR.
#
#   warmup_steps: 10 — protects against erratic early updates from a cold
#     optimizer. Always keep this at minimum 10.
#
#   num_generations: 4 — completions sampled per prompt to compute group-relative
#     advantages. Minimum useful value. If VRAM allows, 8 gives more stable
#     gradient estimates.
#
#   max_completion_length: 4096 — DeepMath requires long reasoning chains.
#     This is the max allowed generation length. Actual completions will often
#     be shorter. If steps are very slow, try 2048 as a speed tradeoff.
#
#   beta: 0.04 — KL penalty coefficient. Controls how far the policy is allowed
#     to drift from the reference. If KL spikes above 10 in W&B, increase to
#     0.1. If rewards plateau early, try reducing to 0.01.
#
#   max_grad_norm: 0.1 — tighter than SFT (1.0). GRPO gradients can be large
#     and spiky. 0.1 keeps training stable.
#
# Effective batch size = per_device_batch_size * num_gpus * grad_accum
#                      = 1 * 4 * 2 = 8 prompts per optimizer step
#                      = 8 * 4 = 32 completions evaluated per step
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

    # Schedule
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
    max_steps=args.max_steps,

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
    # If True, TRL drops ground_truth before calling reward funcs.
    remove_unused_columns=False,

    dataloader_num_workers=4,
    #vllm
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.4,
)

# ---------------------------------------------------------------------------
# Trainer
# No ref_model passed — TRL uses disable_adapter() on the PeftModel to get
# reference logprobs. This is correct ONLY when the LoRA starts from zero
# (fresh run) or when resuming within the same run. Never load a LoRA from
# a different run without first merging it into the base model.
# ---------------------------------------------------------------------------
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[accuracy_reward, format_reward, efficiency_reward],
    args=grpo_config,
    train_dataset=train_data,
)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print("\nStarting GRPO training...")
print(f"  Base model         : {args.model_path}")
print(f"  Checkpoint loaded  : {checkpoint_path or 'None (fresh start)'}")
print(f"  Steps already done : {steps_completed}")
print(f"  LR                 : {args.lr}")
print(f"  Warmup steps       : {args.warmup_steps}")
print(f"  Num generations    : {args.num_generations}")
print(f"  Max completion len : {args.max_completion_length}")
print(f"  Beta (KL penalty)  : {args.beta}")
print(f"  Effective batch    : {args.per_device_batch_size} x 4 GPUs x {args.grad_accum} grad_accum = "
      f"{args.per_device_batch_size * 4 * args.grad_accum} prompts/step, "
      f"{args.per_device_batch_size * 4 * args.grad_accum * args.num_generations} completions/step")

trainer.train()
trainer.save_model(args.output_dir)

print(f"\nGRPO training complete. Adapter saved to {args.output_dir}")
print("\nNEXT STEPS (do not skip):")
print("  1. Check W&B for reward curves. Accuracy reward should trend upward.")
print("  2. Merge this run's LoRA before starting the next run:")
print(f"       python merge_checkpoint.py \\")
print(f"         --base  {args.model_path} \\")
print(f"         --lora  {args.output_dir} \\")
print(f"         --out   {args.output_dir}_merged")
print(f"  3. For the next run, pass --model_path {args.output_dir}_merged")
print("  4. Run eval/run_eval.py to compare SFT vs GRPO accuracy.")