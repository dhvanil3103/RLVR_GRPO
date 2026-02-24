"""
data/prepare_data.py

Loads zwhe99/DeepMath-103K from HuggingFace, performs a train/test split,
formats examples for both SFT and GRPO, and saves the results to disk.

DeepMath-103K columns:
  - question      : the problem text
  - final_answer  : the verifiable ground-truth answer (may be LaTeX)
  - difficulty    : float (some labeled -1.0 or 0, filter with --min_difficulty)
  - topic         : hierarchical subject label
  - r1_solution_1 : full reasoning chain from DeepSeek-R1 (use for SFT)
  - r1_solution_2 : alternate R1 solution
  - r1_solution_3 : alternate R1 solution

Run on a compute node (cpucluster interactive session recommended):
    python data/prepare_data.py \
        --save_dir ./data/processed \
        --model_path /scratch/YOUR_USERNAME/models/qwen2.5-14b-instruct \
        --use_all_r1_solutions \
        --min_difficulty 0 \
        --max_tokens 8192
"""

import argparse
import os
import random

from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer

# ----------------------------------------------------------------------------------------------
# System prompt — must be identical in prepare_data.py, sft_baseline.py, and reward_functions.py
# ----------------------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert mathematician. When solving problems, first think through "
    "your reasoning step by step inside <think> tags. Then provide your final "
    "answer inside <answer> tags. Your answer should be concise and match the "
    "expected format (number, expression, or LaTeX).\n\n"
    "Example format:\n"
    "<think>\n[Your step-by-step reasoning here]\n</think>\n"
    "<answer>42</answer>"
)


def format_for_grpo(example):
    """
    Format for GRPO: prompt only, no assistant turn.
    The model generates completions at training time.
    ground_truth is stored separately for the reward function.
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": example["question"]},
        ],
        "ground_truth": example["final_answer"],
        "difficulty":   example["difficulty"],
        "topic":        example["topic"],
    }


def format_for_sft(example, solution_key: str = "r1_solution_1"):
    """
    Format for SFT: full messages list including the assistant turn.
    The assistant turn wraps the R1 solution in <think> tags and appends
    the ground-truth answer in <answer> tags.

    This is what teaches the model the output format before GRPO runs.
    """
    solution = example[solution_key].strip()
    answer   = example["final_answer"].strip()

    assistant_content = (
        f"<think>\n{solution}\n</think>\n"
        f"<answer>{answer}</answer>"
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": example["question"]},
            {"role": "assistant", "content": assistant_content},
        ],
        "ground_truth": answer,
        "difficulty":   example["difficulty"],
        "topic":        example["topic"],
    }


def make_token_filter(tokenizer, max_tokens: int):
    """
    Returns a filter function that returns True if the FULL conversation
    (system + user + assistant) fits within max_tokens when tokenized.

    Why tokenize the full conversation and not just the assistant turn?
    Because max_seq_length in SFTTrainer applies to the entire sequence,
    not just the assistant portion. The system prompt and question also
    consume tokens, so we need to measure everything together.

    Why not use word count?
    Math LaTeX is token-heavy. A single expression like \frac{a+b}{c-d}
    can be 10+ tokens but only 1 "word". Word count systematically
    underestimates true token length for this dataset.
    """
    def filter_fn(example):
        # Build the full text the way SFTTrainer will see it,
        # using the model's own chat template.
        full_text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        token_count = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
        return token_count <= max_tokens

    return filter_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", default="./data/processed",
        help="Directory to save the processed splits."
    )
    parser.add_argument(
        "--model_path",
        default="/Users/924235556/RLVR/models/qwen2.5-14b-instruct",
        help="Path to Qwen2.5-14B-Instruct. Needed to load the tokenizer for "
             "accurate token counting."
    )
    parser.add_argument(
        "--test_size", type=int, default=2000,
        help="Number of examples held out as test set."
    )
    parser.add_argument(
        "--min_difficulty", type=float, default=0.0,
        help="Drop examples with difficulty below this. Use 0 to drop the "
             "mislabeled -1.0 examples while keeping everything valid."
    )
    parser.add_argument(
        "--max_difficulty", type=float, default=10.0,
    )
    parser.add_argument(
        "--max_tokens", type=int, default=8192,
        help="Drop SFT examples whose full tokenized sequence exceeds this. "
             "8192 covers ~97 percent of DeepMath. Setting higher risks OOM "
             "at training time since Qwen2.5-14B weights alone take ~28GB per "
             "GPU, leaving limited room for long-sequence activations."
    )
    parser.add_argument(
        "--use_all_r1_solutions", action="store_true",
        help="Use all three R1 solutions per problem, tripling SFT training data."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load tokenizer — needed for accurate token counting
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("Loading zwhe99/DeepMath-103K from HuggingFace Hub...")
    raw = load_dataset("zwhe99/DeepMath-103K", split="train")
    print(f"  Total examples: {len(raw):,}")

    # ------------------------------------------------------------------
    # Difficulty filter
    # This removes the mislabeled -1.0 examples and anything outside
    # the requested range.
    # ------------------------------------------------------------------
    before = len(raw)
    raw = raw.filter(
        lambda x: args.min_difficulty <= x["difficulty"] <= args.max_difficulty
    )
    print(
        f"  After difficulty filter [{args.min_difficulty}, {args.max_difficulty}]: "
        f"{len(raw):,} examples (removed {before - len(raw):,})"
    )

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    raw = raw.shuffle(seed=args.seed)
    test_split  = raw.select(range(args.test_size))
    train_split = raw.select(range(args.test_size, len(raw)))
    print(f"  Train: {len(train_split):,}  |  Test: {len(test_split):,}")

    # ------------------------------------------------------------------
    # GRPO format
    # No token filtering here — GRPO generates completions at runtime
    # and sequence length is controlled by the GRPO config, not the data.
    # ------------------------------------------------------------------
    print("\nFormatting for GRPO...")
    grpo_train = train_split.map(
        format_for_grpo,
        remove_columns=raw.column_names,
    )
    grpo_test = test_split.map(
        format_for_grpo,
        remove_columns=raw.column_names,
    )
    grpo_dataset = DatasetDict({"train": grpo_train, "test": grpo_test})
    grpo_path = os.path.join(args.save_dir, "grpo")
    grpo_dataset.save_to_disk(grpo_path)
    print(f"  Saved GRPO dataset to {grpo_path}")

    # ------------------------------------------------------------------
    # SFT format
    # We DO filter by token length here because SFTTrainer applies
    # max_seq_length to the full sequence. Examples over the limit would
    # be silently truncated mid-reasoning-chain, teaching the model
    # incomplete thoughts. Better to drop them entirely.
    # ------------------------------------------------------------------
    print("\nFormatting for SFT...")

    if args.use_all_r1_solutions:
        # Build a list manually using all three solutions.
        # Each original problem yields up to 3 SFT examples.
        sft_examples = []
        for ex in train_split:
            for key in ["r1_solution_1", "r1_solution_2", "r1_solution_3"]:
                if ex.get(key) and ex[key].strip():
                    sft_examples.append(format_for_sft(ex, solution_key=key))
        sft_train_unfiltered = Dataset.from_list(sft_examples)
    else:
        sft_train_unfiltered = train_split.map(
            lambda ex: format_for_sft(ex, solution_key="r1_solution_1"),
            remove_columns=raw.column_names,
        )

    sft_test_unfiltered = test_split.map(
        lambda ex: format_for_sft(ex, solution_key="r1_solution_1"),
        remove_columns=raw.column_names,
    )

    # ------------------------------------------------------------------
    # Token length filter
    #
    # What this does:
    #   1. Takes each SFT example (system + user + assistant messages).
    #   2. Applies the model's chat template to get the exact text
    #      SFTTrainer will tokenize.
    #   3. Tokenizes it with Qwen's tokenizer.
    #   4. Keeps only examples where total tokens <= max_tokens.
    #
    # Why this matters:
    #   If you skip this and just set max_seq_length in SFTTrainer,
    #   the trainer silently truncates long examples. For a DeepMath R1
    #   solution that's 15,000 tokens, it would cut off most of the
    #   reasoning chain. The model then trains on an incomplete thought
    #   which teaches it bad habits. The reward functions will penalise
    #   this during GRPO.
    # ------------------------------------------------------------------
    print(f"  Filtering SFT examples with more than {args.max_tokens} tokens...")
    token_filter = make_token_filter(tokenizer, args.max_tokens)

    before_train = len(sft_train_unfiltered)
    sft_train = sft_train_unfiltered.filter(token_filter, num_proc=8)
    before_test = len(sft_test_unfiltered)
    sft_test  = sft_test_unfiltered.filter(token_filter, num_proc=8)

    print(
        f"  SFT train: {before_train:,} -> {len(sft_train):,} "
        f"(removed {before_train - len(sft_train):,} over {args.max_tokens} tokens)"
    )
    print(
        f"  SFT test : {before_test:,} -> {len(sft_test):,} "
        f"(removed {before_test - len(sft_test):,} over {args.max_tokens} tokens)"
    )

    sft_dataset = DatasetDict({"train": sft_train, "test": sft_test})
    sft_path = os.path.join(args.save_dir, "sft")
    sft_dataset.save_to_disk(sft_path)
    print(f"  Saved SFT dataset to {sft_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n===== Done. Summary =====")
    print(f"  GRPO train : {len(grpo_train):,} examples")
    print(f"  GRPO test  : {len(grpo_test):,} examples")
    print(f"  SFT train  : {len(sft_train):,} examples")
    print(f"  SFT test   : {len(sft_test):,} examples")
    pct_kept = len(sft_train) / before_train * 100
    print(f"  Token filter kept {pct_kept:.1f}% of SFT train examples")


if __name__ == "__main__":
    main()