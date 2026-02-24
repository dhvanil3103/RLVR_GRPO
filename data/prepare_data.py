"""
data/prepare_data.py

Loads zwhe99/DeepMath-103K from HuggingFace, performs a train/test split,
formats examples for both SFT and GRPO, and saves the results to disk.

DeepMath-103K columns:
  - question      : the problem text
  - final_answer  : the verifiable ground-truth answer (may be LaTeX)
  - difficulty    : float in roughly 5-9 (higher = harder)
  - topic         : hierarchical subject label
  - r1_solution_1 : full reasoning chain from DeepSeek-R1 (use for SFT)
  - r1_solution_2 : alternate R1 solution
  - r1_solution_3 : alternate R1 solution

Run on the LOGIN node:
    python data/prepare_data.py --save_dir ./data/processed
"""

import argparse
import os
import random

from datasets import load_dataset, DatasetDict

# System prompt used for both SFT and GRPO
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
    Format a DeepMath example as a GRPO prompt.

    GRPO receives the prompt only; the model generates completions at training
    time. We store ground_truth separately so the reward function can use it.
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
    Format a DeepMath example as an SFT training example.

    We wrap the R1 solution in <think> tags and append the ground-truth
    answer in <answer> tags.  This teaches the model the output format
    before GRPO is applied.

    Args:
        solution_key: which of the three R1 solutions to use.
                      Pass 'all' to return three examples per input
                      (caller must handle the list).
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", default="./data/processed",
        help="Directory to save the processed splits."
    )
    parser.add_argument(
        "--test_size", type=int, default=2000,
        help="Number of examples to hold out as a test set."
    )
    parser.add_argument(
        "--min_difficulty", type=float, default=0.0,
        help="Only keep examples with difficulty >= this value (0 = keep all)."
    )
    parser.add_argument(
        "--max_difficulty", type=float, default=10.0,
        help="Only keep examples with difficulty <= this value."
    )
    parser.add_argument(
        "--use_all_r1_solutions", action="store_true",
        help="Triple the SFT dataset by using all three R1 solutions per problem."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)

    print("Loading zwhe99/DeepMath-103K from HuggingFace Hub...")
    raw = load_dataset("zwhe99/DeepMath-103K", split="train")
    print(f"  Total examples: {len(raw)}")

    # Optional difficulty filter
    if args.min_difficulty > 0.0 or args.max_difficulty < 10.0:
        before = len(raw)
        raw = raw.filter(
            lambda x: args.min_difficulty <= x["difficulty"] <= args.max_difficulty
        )
        print(
            f"  After difficulty filter [{args.min_difficulty}, {args.max_difficulty}]: "
            f"{len(raw)} examples (removed {before - len(raw)})"
        )

    # Train / test split
    # Shuffle then split so the test set is representative.
    raw = raw.shuffle(seed=args.seed)
    test_split  = raw.select(range(args.test_size))
    train_split = raw.select(range(args.test_size, len(raw)))
    print(f"  Train: {len(train_split)}  |  Test: {len(test_split)}")

    # GRPO format  (prompt + ground_truth, no assistant turn)
    print("Formatting for GRPO...")
    grpo_train = train_split.map(
        format_for_grpo,
        remove_columns=raw.column_names,
    )
    grpo_test  = test_split.map(
        format_for_grpo,
        remove_columns=raw.column_names,
    )
    grpo_dataset = DatasetDict({"train": grpo_train, "test": grpo_test})
    grpo_path = os.path.join(args.save_dir, "grpo")
    grpo_dataset.save_to_disk(grpo_path)
    print(f"  Saved GRPO dataset to {grpo_path}")

    # SFT format  (messages list with assistant turn containing R1 solution)
    print("Formatting for SFT...")
    if args.use_all_r1_solutions:
        # Use all three solutions: each problem yields 3 training examples.
        sft_examples = []
        for ex in train_split:
            for key in ["r1_solution_1", "r1_solution_2", "r1_solution_3"]:
                if ex[key] and ex[key].strip():
                    sft_examples.append(format_for_sft(ex, solution_key=key))
        from datasets import Dataset
        sft_train = Dataset.from_list(sft_examples)
    else:
        sft_train = train_split.map(
            lambda ex: format_for_sft(ex, solution_key="r1_solution_1"),
            remove_columns=raw.column_names,
        )

    sft_test = test_split.map(
        lambda ex: format_for_sft(ex, solution_key="r1_solution_1"),
        remove_columns=raw.column_names,
    )
    sft_dataset = DatasetDict({"train": sft_train, "test": sft_test})
    sft_path = os.path.join(args.save_dir, "sft")
    sft_dataset.save_to_disk(sft_path)
    print(f"  Saved SFT dataset to {sft_path}")
    if args.use_all_r1_solutions:
        print(f"  SFT train size (3x solutions): {len(sft_train)}")

    print("\nDone. Summary:")
    print(f"  GRPO train : {len(grpo_train):,} examples")
    print(f"  GRPO test  : {len(grpo_test):,} examples")
    print(f"  SFT train  : {len(sft_train):,} examples")
    print(f"  SFT test   : {len(sft_test):,} examples")


if __name__ == "__main__":
    main()