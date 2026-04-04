"""
eval/aggregate_results.py

Reads the four result JSONs and prints a comparison table.

Usage:
  python eval/aggregate_results.py --results_dir ./results
"""

import argparse
import json
import os

FILES = [
    ("base",         "base.json"),
    ("sft",          "sft.json"),
    ("grpo_ckpt5000","grpo_ckpt5000.json"),
    ("grpo_final",   "grpo_final.json"),
]


def load_summary(path):
    with open(path) as f:
        data = json.load(f)
    return data["summary"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    rows = []
    for label, filename in FILES:
        fpath = os.path.join(args.results_dir, filename)
        if not os.path.exists(fpath):
            print(f"  [MISSING] {fpath}")
            continue
        s = load_summary(fpath)
        rows.append({
            "label":        label,
            "n_total":      s["n_total"],
            "n_scored":     s["n_scored"],
            "accuracy":     s["accuracy"] * 100,
            "format_rate":  s["format_rate"] * 100,
            "discard_rate": s["discard_rate"] * 100,
        })

    if not rows:
        print("No result files found.")
        return

    # Print table
    col_w = 20
    header = f"{'Model':<{col_w}}  {'Scored':>7}  {'Accuracy':>10}  {'Format':>8}  {'Discarded':>10}"
    sep    = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['label']:<{col_w}}  {r['n_scored']:>7}  "
            f"{r['accuracy']:>9.2f}%  {r['format_rate']:>7.2f}%  {r['discard_rate']:>9.2f}%"
        )
    print(sep)

    # Delta vs base
    if len(rows) > 1:
        base = rows[0]
        print(f"\n  Delta vs base (accuracy / format / discard):")
        for r in rows[1:]:
            d_acc  = r["accuracy"]     - base["accuracy"]
            d_fmt  = r["format_rate"]  - base["format_rate"]
            d_disc = r["discard_rate"] - base["discard_rate"]
            sa = "+" if d_acc  >= 0 else ""
            sf = "+" if d_fmt  >= 0 else ""
            sd = "+" if d_disc >= 0 else ""
            print(f"    {r['label']:<{col_w-4}}  {sa}{d_acc:.2f}%  /  {sf}{d_fmt:.2f}%  /  {sd}{d_disc:.2f}%")
    print()


if __name__ == "__main__":
    main()