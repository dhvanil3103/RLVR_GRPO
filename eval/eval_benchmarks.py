"""
eval/eval_benchmarks.py

Evaluates a model on MATH-500 or AIME 2024 benchmarks loaded directly from HuggingFace.
Use this for base model and GRPO final model comparison on standard benchmarks.

Datasets:
  MATH-500 : HuggingFaceH4/MATH-500   (500 problems, test split)
  AIME 2024: HuggingFaceH4/aime_2024  (30 problems, train split)

Usage:
  # GRPO model on MATH-500
  python eval/eval_benchmarks.py \
    --model_path ./checkpoints/grpo_final_merged \
    --dataset    math500 \
    --output     results/grpo_final_math500.json \
    --run_name   grpo_final

  # Base model on AIME 2024 (uses boxed fallback)
  python eval/eval_benchmarks.py \
    --model_path ./models/qwen2.5-7b-instruct \
    --dataset    aime2024 \
    --output     results/base_aime2024.json \
    --run_name   base \
    --fallback_boxed
"""

import argparse
import json
import re
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from math_verify import verify, parse
from tqdm import tqdm


# ---------------------------------------------------------------------------
# System prompt — must match exactly what was used during GRPO training
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Think through the problem step by step inside <think> tags. "
    "Then provide your final answer inside <answer> tags using LaTeX where needed. "
    "Example format:\n"
    "<think>\n[your reasoning here]\n</think>\n"
    "<answer>42</answer>"
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "math500": {
        "hf_name":      "HuggingFaceH4/MATH-500",
        "split":        "test",
        "problem_col":  "problem",
        "answer_col":   "answer",
        "description":  "MATH-500",
    },
    "aime2024": {
        "hf_name":      "HuggingFaceH4/aime_2024",
        "split":        "train",
        "problem_col":  "problem",
        "answer_col":   "answer",
        "description":  "AIME 2024",
    },
}


def load_benchmark(dataset_name: str):
    cfg = DATASET_CONFIGS[dataset_name]
    print(f"Loading {cfg['description']} from {cfg['hf_name']}...")
    ds = load_dataset(cfg["hf_name"], split=cfg["split"])
    print(f"  {len(ds)} problems loaded.")
    return ds, cfg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True,
                        help="Path to the merged model checkpoint to evaluate.")
    parser.add_argument("--dataset",     required=True, choices=["math500", "aime2024"],
                        help="Which benchmark to evaluate on.")
    parser.add_argument("--output",      default="results.json",
                        help="Path to write results JSON.")
    parser.add_argument("--log_file",    default=None,
                        help="Path for human-readable log. Defaults to --output with .log extension.")
    parser.add_argument("--run_name",    default="",
                        help="Label for this run in the output files.")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max tokens to generate.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Evaluate only the first N problems (for quick smoke tests).")
    parser.add_argument("--preview_count", type=int, default=10,
                        help="Number of full responses to include in the log file (default 10).")
    parser.add_argument("--fallback_boxed", action="store_true",
                        help="Extract answer from \\boxed{} when <answer> tag is missing. "
                             "Use this for the base model.")
    parser.add_argument("--save_every",  type=int, default=5,
                        help="Save incremental results every N examples (default 5).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer_tag(completion: str):
    """Returns content inside <answer>...</answer>, stripping LaTeX delimiters."""
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        answer = re.sub(r"^\\\(", "", answer).strip()
        answer = re.sub(r"\\\)$", "", answer).strip()
        answer = re.sub(r"^\\\[", "", answer).strip()
        answer = re.sub(r"\\\]$", "", answer).strip()
        return answer
    return None


def extract_boxed(completion: str):
    """Returns content inside \\boxed{}. Used as fallback for base model."""
    match = re.search(r"\\boxed\{([^}]+)\}", completion)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Answer verification
# ---------------------------------------------------------------------------

def check_correct(answer_content: str, ground_truth: str) -> bool:
    """
    Tries math-verify first (handles numeric/LaTeX/symbolic equivalence).
    Falls back to normalized string match for yes/no/true/false style answers.
    """
    try:
        if verify(parse(answer_content), parse(ground_truth)):
            return True
    except Exception:
        pass

    predicted = answer_content.strip().lower()
    expected  = ground_truth.strip().lower()

    if predicted == expected:
        return True

    bool_map = {"yes": True, "no": False, "true": True, "false": False}
    if predicted in bool_map and expected in bool_map:
        return bool_map[predicted] == bool_map[expected]

    return False


def check_format(completion: str) -> bool:
    has_think  = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
    return has_think and has_answer


# ---------------------------------------------------------------------------
# Incremental save helpers
# ---------------------------------------------------------------------------

def save_json(path: str, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"  [WARNING] Failed to save JSON: {e}")


def write_log(log_path: str, run_name: str, summary: dict, results: list, preview_count: int):
    divider  = "=" * 80
    thin_div = "-" * 80
    try:
        with open(log_path, "w", encoding="utf-8") as f:

            # Header
            f.write(f"{divider}\n")
            f.write(f"  BENCHMARK EVALUATION LOG\n")
            f.write(f"  Run        : {run_name}\n")
            f.write(f"  Model      : {summary['model_path']}\n")
            f.write(f"  Dataset    : {summary['dataset']}\n")
            f.write(f"  Status     : {summary['status']}\n")
            f.write(f"  Processed  : {summary.get('n_processed', summary.get('n_total', '?'))}"
                    f" / {summary['n_total']}\n")
            f.write(f"  Scored     : {summary['n_scored']}  "
                    f"(discarded {summary['n_discarded']}, {summary['discard_rate']*100:.1f}%)\n")
            f.write(f"  Accuracy   : {summary['accuracy']*100:.2f}%\n")
            f.write(f"  Format     : {summary['format_rate']*100:.2f}%\n")
            f.write(f"{divider}\n\n")

            # Full response preview
            f.write(f"{divider}\n")
            f.write(f"  FULL RESPONSE PREVIEW  (first {preview_count} non-discarded examples)\n")
            f.write(f"{divider}\n\n")

            shown = 0
            for r in results:
                if shown >= preview_count:
                    break
                if r["discarded"]:
                    continue
                status = "CORRECT" if r["correct"] else "WRONG"
                f.write(f"[Example {r['index']}]  Status: {status}\n")
                f.write(f"{thin_div}\n")
                f.write(f"QUESTION:\n{r['question']}\n\n")
                f.write(f"GROUND TRUTH : {r['ground_truth']}\n\n")
                f.write(f"FULL RESPONSE:\n{r['completion']}\n")
                f.write(f"\n{thin_div}\n\n")
                shown += 1

            # Results table
            f.write(f"{divider}\n")
            f.write(f"  ALL RESULTS  (ground truth vs predicted answer)\n")
            f.write(f"{divider}\n\n")

            col = 40
            f.write(f"  {'#':<6}  {'Status':<10}  {'Ground Truth':<{col}}  Predicted\n")
            f.write(f"  {thin_div}\n")

            for r in results:
                if r["discarded"]:
                    status    = "DISCARDED"
                    predicted = "(no answer found)"
                else:
                    status    = "CORRECT" if r["correct"] else "WRONG"
                    match = re.search(r"<answer>(.*?)</answer>", r["completion"], re.DOTALL)
                    if match:
                        predicted = match.group(1).strip()
                    else:
                        bm = re.search(r"\\boxed\{([^}]+)\}", r["completion"])
                        predicted = bm.group(1).strip() if bm else "(parse error)"

                f.write(f"  {r['index']:<6}  {status:<10}  "
                        f"{r['ground_truth'][:col]:<{col}}  {predicted[:col]}\n")

    except Exception as e:
        print(f"  [WARNING] Failed to write log: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    run_name = args.run_name or args.model_path
    log_path = args.log_file if args.log_file else os.path.splitext(args.output)[0] + ".log"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model   : {args.model_path}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Output  : {args.output}")
    print(f"{'='*60}\n")

    # Load benchmark dataset
    ds, cfg = load_benchmark(args.dataset)
    if args.num_samples:
        ds = ds.select(range(args.num_samples))
    total = len(ds)

    # Load model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model loaded across {len(set(str(p.device) for p in model.parameters()))} device(s).\n")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Eval loop
    results         = []
    correct_count   = 0
    format_count    = 0
    discarded_count = 0

    for i, example in enumerate(tqdm(ds, desc="Evaluating")):
        question     = example[cfg["problem_col"]]
        ground_truth = str(example[cfg["answer_col"]])

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        completion = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        answer_content = extract_answer_tag(completion)

        if answer_content is None:
            if args.fallback_boxed:
                answer_content = extract_boxed(completion)
            if answer_content is None:
                discarded_count += 1
                results.append({
                    "index":        i,
                    "question":     question,
                    "ground_truth": ground_truth,
                    "completion":   completion,
                    "correct":      None,
                    "format_ok":    False,
                    "discarded":    True,
                })
                continue

        correct = check_correct(answer_content, ground_truth)
        fmt_ok  = check_format(completion)

        correct_count += int(correct)
        format_count  += int(fmt_ok)

        results.append({
            "index":        i,
            "question":     question,
            "ground_truth": ground_truth,
            "completion":   completion,
            "correct":      correct,
            "format_ok":    fmt_ok,
            "discarded":    False,
        })

        # Incremental save every N examples
        if (i + 1) % args.save_every == 0:
            scored_so_far = (i + 1) - discarded_count
            acc_so_far    = correct_count / scored_so_far * 100 if scored_so_far else 0
            disc_so_far   = discarded_count / (i + 1) * 100
            print(f"  [{i+1}/{total}]  acc={acc_so_far:.1f}%  discarded={disc_so_far:.1f}%")

            partial_summary = {
                "run_name":     run_name,
                "model_path":   args.model_path,
                "dataset":      cfg["description"],
                "status":       "in_progress",
                "n_total":      total,
                "n_processed":  i + 1,
                "n_scored":     scored_so_far,
                "n_discarded":  discarded_count,
                "accuracy":     round(correct_count / scored_so_far, 4) if scored_so_far else 0,
                "format_rate":  round(format_count  / scored_so_far, 4) if scored_so_far else 0,
                "discard_rate": round(discarded_count / (i + 1), 4),
            }
            save_json(args.output, {"summary": partial_summary, "examples": results})
            write_log(log_path, run_name, partial_summary, results, args.preview_count)

    # Final summary
    n        = len(results)
    scored_n = n - discarded_count
    summary = {
        "run_name":     run_name,
        "model_path":   args.model_path,
        "dataset":      cfg["description"],
        "status":       "complete",
        "n_total":      n,
        "n_scored":     scored_n,
        "n_discarded":  discarded_count,
        "accuracy":     round(correct_count / scored_n, 4) if scored_n else 0,
        "format_rate":  round(format_count  / scored_n, 4) if scored_n else 0,
        "discard_rate": round(discarded_count / n, 4),
    }

    print(f"\n{'='*60}")
    print(f"  Run       : {run_name}")
    print(f"  Dataset   : {cfg['description']}")
    print(f"  Scored    : {scored_n}  (discarded {discarded_count}, {summary['discard_rate']*100:.1f}%)")
    print(f"  Accuracy  : {summary['accuracy']*100:.2f}%")
    print(f"  Format    : {summary['format_rate']*100:.2f}%")
    print(f"{'='*60}\n")

    save_json(args.output, {"summary": summary, "examples": results})
    print(f"Results written to {args.output}")
    write_log(log_path, run_name, summary, results, args.preview_count)


if __name__ == "__main__":
    main()