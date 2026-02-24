"""
data/inspect_data.py

Quick sanity-check script.  Run this on the login node after prepare_data.py
to verify the splits look correct before submitting any training jobs.

Usage:
    python data/inspect_data.py --data_dir ./data/processed
"""

import argparse
import os
import textwrap

from datasets import load_from_disk


def print_separator(title=""):
    width = 70
    if title:
        print(f"\n{'=' * 10} {title} {'=' * (width - 12 - len(title))}")
    else:
        print("=" * width)


def truncate(text, max_chars=300):
    if len(text) > max_chars:
        return text[:max_chars] + " ... [truncated]"
    return text


def inspect_grpo(data_dir):
    path = os.path.join(data_dir, "grpo")
    dataset = load_from_disk(path)

    print_separator("GRPO Dataset")
    for split_name, split in dataset.items():
        print(f"\n  Split: {split_name}  |  {len(split):,} examples")
        print(f"  Columns: {split.column_names}")

        # Difficulty distribution
        diffs = split["difficulty"]
        print(f"  Difficulty  min={min(diffs):.1f}  max={max(diffs):.1f}  "
              f"mean={sum(diffs)/len(diffs):.2f}")

        # Topic distribution (top 5)
        from collections import Counter
        topic_counts = Counter(split["topic"])
        print("  Top 5 topics:")
        for topic, count in topic_counts.most_common(5):
            print(f"    {topic:<50} {count:>6}")

    # Show a single GRPO example in full
    print_separator("GRPO Example (train[0])")
    ex = dataset["train"][0]
    print(f"  difficulty : {ex['difficulty']}")
    print(f"  topic      : {ex['topic']}")
    print(f"  ground_truth: {ex['ground_truth']}")
    print("\n  prompt messages:")
    for msg in ex["prompt"]:
        role = msg["role"].upper()
        content = truncate(msg["content"], 200)
        print(f"\n  [{role}]\n{textwrap.indent(content, '    ')}")


def inspect_sft(data_dir):
    path = os.path.join(data_dir, "sft")
    dataset = load_from_disk(path)

    print_separator("SFT Dataset")
    for split_name, split in dataset.items():
        print(f"\n  Split: {split_name}  |  {len(split):,} examples")
        print(f"  Columns: {split.column_names}")

    # Show a single SFT example
    print_separator("SFT Example (train[0])")
    ex = dataset["train"][0]
    print(f"  difficulty   : {ex['difficulty']}")
    print(f"  topic        : {ex['topic']}")
    print(f"  ground_truth : {ex['ground_truth']}")
    print("\n  messages:")
    for msg in ex["messages"]:
        role = msg["role"].upper()
        content = truncate(msg["content"], 400)
        print(f"\n  [{role}]\n{textwrap.indent(content, '    ')}")

    # Check assistant turn has <think> and <answer> tags
    assistant_content = dataset["train"][0]["messages"][-1]["content"]
    has_think  = "<think>" in assistant_content and "</think>" in assistant_content
    has_answer = "<answer>" in assistant_content and "</answer>" in assistant_content
    print(f"\n  Format check  <think>: {has_think}  |  <answer>: {has_answer}")
    if not (has_think and has_answer):
        print("  WARNING: example is missing expected tags. Check prepare_data.py.")

    # Average assistant response length
    lengths = [
        len(ex["messages"][-1]["content"].split())
        for ex in dataset["train"]
    ]
    print(f"\n  Avg assistant token count (words): {sum(lengths)/len(lengths):.0f}")
    print(f"  Max assistant token count (words): {max(lengths)}")
    print(f"  95th pct                         : {sorted(lengths)[int(0.95*len(lengths))]}")
    print(
        "\n  NOTE: If the 95th-percentile word count is above ~1500, consider "
        "setting max_seq_length >= 3072 in sft_baseline.py to avoid heavy truncation."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/processed")
    args = parser.parse_args()

    inspect_grpo(args.data_dir)
    inspect_sft(args.data_dir)
    print_separator()
    print("Inspection complete.")


if __name__ == "__main__":
    main()