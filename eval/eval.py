"""
eval/eval.py

Evaluates a model checkpoint on the held-out DeepMath test split (1600 examples).
Measures:
  - accuracy_rate  : answer correctness via math-verify + string fallback
  - format_rate    : fraction of scored outputs that have <think>...</think> and <answer>...</answer>
  - discard_rate   : fraction of outputs with no <answer> tag (excluded from accuracy/format)

Usage:
  python eval.py \
    --model_path ./checkpoints/grpo_final_merged \
    --data_dir   /scratch/your_username/data/grpo \
    --output     results_final.json \
    --run_name   grpo_final
"""

import argparse
import json
import re
import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from math_verify import verify, parse
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True,
                        help="Path to the merged model checkpoint to evaluate.")
    parser.add_argument("--data_dir",    required=True,
                        help="Path to the GRPO dataset folder saved via save_to_disk "
                             "(must contain a 'test' split with 'prompt' and 'ground_truth').")
    parser.add_argument("--output",      default="results.json",
                        help="Path to write per-example results JSON.")
    parser.add_argument("--run_name",    default="",
                        help="Label for this run, written into the output JSON summary.")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max tokens to generate. Match what was used during training.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="If set, only evaluate on the first N examples (for quick smoke tests).")
    parser.add_argument("--log_file", default=None,
                        help="Path for the human-readable predictions log. Defaults to --output with .log extension.")
    parser.add_argument("--preview_count", type=int, default=10,
                        help="Number of full responses printed at the end for manual inspection (default 10).")
    parser.add_argument("--fallback_boxed", action="store_true",
                        help="If set, extract answer from \\boxed{} when <answer> tag is missing. "
                             "Use this for the base model which was not trained on answer tags.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Answer verification  (math-verify with string fallback)
# ---------------------------------------------------------------------------

def extract_answer_tag(completion: str):
    """Returns content inside <answer>...</answer>, or None if the tag is missing."""
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        # Strip LaTeX inline/display math delimiters the model sometimes wraps around the answer
        answer = re.sub(r"^\\(|\\)$", "", answer).strip()
        answer = re.sub(r"^\\[|\\]$", "", answer).strip()
        return answer
    return None


def extract_boxed(completion: str):
    """Returns content inside \\boxed{}, or None if not found. Used as fallback for base model."""
    match = re.search(r"\\boxed\{([^}]+)\}", completion)
    if match:
        return match.group(1).strip()
    return None


def check_correct(answer_content: str, ground_truth: str) -> bool:
    """
    Receives the text already extracted from <answer>...</answer>.
    Tries math-verify first (handles numeric/LaTeX/symbolic equivalence).
    Falls back to normalized string match for yes/no/true/false style answers.
    """
    # -- math-verify pass --
    try:
        if verify(parse(answer_content), parse(ground_truth)):
            return True
    except Exception:
        pass

    # -- string fallback --
    predicted = answer_content.strip().lower()
    expected  = ground_truth.strip().lower()

    if predicted == expected:
        return True

    # yes/no/true/false equivalence
    bool_map = {"yes": True, "no": False, "true": True, "false": False}
    if predicted in bool_map and expected in bool_map:
        return bool_map[predicted] == bool_map[expected]

    return False


# ---------------------------------------------------------------------------
# Format check  (mirrors format_reward used during GRPO training)
# ---------------------------------------------------------------------------

def check_format(completion: str) -> bool:
    has_think  = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
    return has_think and has_answer



# ---------------------------------------------------------------------------
# Log file writer
# ---------------------------------------------------------------------------

def write_log(log_path: str, run_name: str, summary: dict, results: list, preview_count: int):
    """
    Writes a human-readable log file with:
      - Summary stats at the top
      - Full response text for the first `preview_count` examples
      - Ground truth vs predicted answer for every example
    """
    divider  = "=" * 80
    thin_div = "-" * 80

    with open(log_path, "w", encoding="utf-8") as f:

        # -- Header --
        f.write(f"{divider}\n")
        f.write(f"  EVALUATION LOG\n")
        f.write(f"  Run        : {run_name}\n")
        f.write(f"  Model      : {summary['model_path']}\n")
        f.write(f"  Total      : {summary['n_total']}\n")
        f.write(f"  Scored     : {summary['n_scored']}  (discarded {summary['n_discarded']}, "
                f"{summary['discard_rate']*100:.1f}%)\n")
        f.write(f"  Accuracy   : {summary['accuracy']*100:.2f}%\n")
        f.write(f"  Format     : {summary['format_rate']*100:.2f}%\n")
        f.write(f"{divider}\n\n")

        # -- Full response preview for first N non-discarded examples --
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

        # -- Per-example ground truth vs predicted table --
        f.write(f"{divider}\n")
        f.write(f"  ALL RESULTS  (ground truth vs predicted answer)\n")
        f.write(f"{divider}\n\n")

        col = 40
        f.write(f"  {'#':<6}  {'Status':<10}  {'Ground Truth':<{col}}  Predicted\n")
        f.write(f"  {thin_div}\n")

        for r in results:
            if r["discarded"]:
                status    = "DISCARDED"
                predicted = "(no <answer> tag)"
            else:
                status    = "CORRECT" if r["correct"] else "WRONG"
                match     = re.search(r"<answer>(.*?)</answer>", r["completion"], re.DOTALL)
                predicted = match.group(1).strip() if match else "(parse error)"

            gt_short   = r["ground_truth"][:col]
            pred_short = predicted[:col]
            f.write(f"  {r['index']:<6}  {status:<10}  {gt_short:<{col}}  {pred_short}\n")

    print(f"Log written to {log_path}")

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    print(f"\n{'='*60}")
    print(f"  Model   : {args.model_path}")
    print(f"  Data    : {args.data_dir}")
    print(f"  Output  : {args.output}")
    print(f"{'='*60}\n")

    # -- Load dataset --
    dataset = load_from_disk(args.data_dir)
    test_data = dataset["test"]
    if args.num_samples:
        test_data = test_data.select(range(args.num_samples))
    print(f"Loaded {len(test_data)} test examples.\n")

    # -- Load model and tokenizer --
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # shards across all visible GPUs
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model loaded across {len(set(str(p.device) for p in model.parameters()))} device(s).\n")

    # -- Generation config (greedy, reproducible) --
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )

    # -- Resolve output paths and create directories up front --
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    log_path = args.log_file if args.log_file else os.path.splitext(args.output)[0] + ".log"

    # -- Eval loop --
    results        = []
    correct_count  = 0
    format_count   = 0
    discarded_count = 0

    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        # example["prompt"] is already a list of message dicts from prepare_data.py
        prompt_text = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        completion = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        ground_truth   = example["ground_truth"]
        answer_content = extract_answer_tag(completion)

        # If no <answer> tag, try boxed fallback for base model, otherwise discard
        if answer_content is None:
            if args.fallback_boxed:
                answer_content = extract_boxed(completion)
            if answer_content is None:
                discarded_count += 1
                results.append({
                    "index":        i,
                    "question":     example["prompt"][-1]["content"],
                    "ground_truth": ground_truth,
                    "completion":   completion,
                    "correct":      None,   # None = discarded, not counted as wrong
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
            "question":     example["prompt"][-1]["content"],
            "ground_truth": ground_truth,
            "completion":   completion,
            "correct":      correct,
            "format_ok":    fmt_ok,
            "discarded":    False,
        })

        # Live progress + incremental save every 50 examples
        if (i + 1) % 5 == 0:
            scored_so_far = (i + 1) - discarded_count
            acc_so_far    = correct_count / scored_so_far * 100 if scored_so_far else 0
            fmt_so_far    = format_count  / scored_so_far * 100 if scored_so_far else 0
            disc_so_far   = discarded_count / (i + 1) * 100
            print(f"  [{i+1}/{len(test_data)}]  acc={acc_so_far:.1f}%  fmt={fmt_so_far:.1f}%  discarded={disc_so_far:.1f}%")

            # Write partial results to disk so progress is not lost if job is killed
            partial_summary = {
                "run_name":     args.run_name,
                "model_path":   args.model_path,
                "status":       "in_progress",
                "n_total":      len(test_data),
                "n_processed":  i + 1,
                "n_scored":     scored_so_far,
                "n_discarded":  discarded_count,
                "accuracy":     round(correct_count / scored_so_far, 4) if scored_so_far else 0,
                "format_rate":  round(format_count  / scored_so_far, 4) if scored_so_far else 0,
                "discard_rate": round(discarded_count / (i + 1), 4),
            }
            with open(args.output, "w") as f:
                json.dump({"summary": partial_summary, "examples": results}, f, indent=2)

            # Write incremental log alongside JSON
            write_log(
                log_path      = log_path,
                run_name      = args.run_name or args.model_path,
                summary       = partial_summary,
                results       = results,
                preview_count = args.preview_count,
            )

    # -- Summary --
    n        = len(results)
    scored_n = n - discarded_count
    summary = {
        "run_name":     args.run_name,
        "model_path":   args.model_path,
        "status":       "complete",
        "n_total":      n,
        "n_scored":     scored_n,
        "n_discarded":  discarded_count,
        "accuracy":     round(correct_count / scored_n, 4) if scored_n else 0,
        "format_rate":  round(format_count  / scored_n, 4) if scored_n else 0,
        "discard_rate": round(discarded_count / n, 4),
    }

    print(f"\n{'='*60}")
    print(f"  Run       : {args.run_name or args.model_path}")
    print(f"  Total     : {n}")
    print(f"  Scored    : {scored_n}  (discarded {discarded_count}, {summary['discard_rate']*100:.1f}%)")
    print(f"  Accuracy  : {summary['accuracy']*100:.2f}%")
    print(f"  Format    : {summary['format_rate']*100:.2f}%")
    print(f"{'='*60}\n")

    # -- Save final JSON results --
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "examples": results}, f, indent=2)
    print(f"Results written to {args.output}")

    # -- Write final human-readable log --
    write_log(
        log_path     = log_path,
        run_name     = args.run_name or args.model_path,
        summary      = summary,
        results      = results,
        preview_count= args.preview_count,
    )

if __name__ == "__main__":
    main()