"""
training/reward_functions.py

Reward functions for GRPO training on DeepMath-103K.

Three signals, in order of importance:
  1. accuracy_reward    (weight 1.0) — did the model get the right answer?
  2. format_reward      (weight 0.5) — did the model use the correct tag structure?
  3. efficiency_reward  (weight 0.3) — did the model reason efficiently?

A correct, well-formatted, efficiently-reasoned answer scores ~1.8.
An incorrect but well-formatted answer scores ~0.5.
That gap is the learning signal GRPO uses.

These functions must all share the same signature:
    fn(completions, **kwargs) -> list[float]
where completions is a list of strings (one per generation in the batch).
Ground truth is passed via kwargs by GRPOTrainer when it's in the dataset.
"""

import re

from math_verify import parse, verify


# ---------------------------------------------------------------------------
# Helper: robust correctness check
# ---------------------------------------------------------------------------

def _is_correct(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted matches ground_truth.

    Tries math_verify first for symbolic equivalence (handles cases like
    '1/2' matching '\\frac{1}{2}'). Falls back to case-insensitive string
    comparison for non-math answers like 'Yes', 'No', or single words that
    math_verify cannot parse.

    The fallback is intentional: math_verify returns False rather than
    raising on unparseable strings, so we always need the string fallback
    as a second attempt rather than only catching exceptions.
    """
    try:
        if verify(parse(predicted), parse(ground_truth)):
            return True
    except Exception:
        pass
    # Fallback for Yes/No, single-word answers, etc.
    return predicted.strip().lower() == ground_truth.strip().lower()


# ---------------------------------------------------------------------------
# Reward 1: Accuracy (weight 1.0)
# ---------------------------------------------------------------------------

def accuracy_reward(completions, ground_truth, **kwargs):
    """
    Core signal. Returns 1.0 if the answer inside <answer> tags matches
    ground_truth, 0.0 otherwise.

    ground_truth values are bare LaTeX strings from DeepMath's final_answer
    column (e.g. '\\frac{3}{2}', '-7', '2\\sqrt{3}', 'Yes').
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        try:
            match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            if match:
                predicted = match.group(1).strip()
                rewards.append(1.0 if _is_correct(predicted, gt) else 0.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Reward 2: Format compliance (weight 0.5)
# ---------------------------------------------------------------------------

def format_reward(completions, **kwargs):
    """
    Structural reward. Checks that the model uses both tag pairs correctly
    and in the right order (<think> before <answer>).

    Breakdown:
      +0.2  for opening and closing <think> tags
      +0.2  for opening and closing <answer> tags
      +0.1  for <think> appearing before <answer> (correct order)

    Max score: 0.5. This reward jumps quickly in the first 50-100 steps
    and then plateaus — that's expected and healthy.
    """
    rewards = []
    for completion in completions:
        score = 0.0
        has_think  = "<think>" in completion and "</think>" in completion
        has_answer = "<answer>" in completion and "</answer>" in completion

        if has_think:
            score += 0.2
        if has_answer:
            score += 0.2
        if has_think and has_answer:
            think_pos  = completion.find("<think>")
            answer_pos = completion.find("<answer>")
            if think_pos < answer_pos:
                score += 0.1

        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# Reward 3: Reasoning efficiency (weight 0.3)
# ---------------------------------------------------------------------------

def efficiency_reward(completions, **kwargs):
    """
    Encourages efficient reasoning chains — neither too short nor too long.

    Three zones based on character count inside <think> tags:

      Growth  (0 to 9,000 chars, ~1500 words):
        Linear scale from 0.0 to 0.3. Rewards the model for doing
        non-trivial reasoning rather than jumping straight to an answer.

      Plateau (9,000 to 24,000 chars, ~6,000 tokens):
        Constant reward of 0.3. The model is thinking deeply and that's
        fine — no incentive to cut it short or pad it out.

      Penalty (> 24,000 chars):
        Subtracts 0.1 per 4,000 excess chars (~1,000 tokens), floored at
        -0.5. Discourages rambling, repetitive reasoning loops, and
        unnecessary rechecking that adds latency without improving accuracy.

    Why this replaces length_reward:
        length_reward rewarded longer thinking unconditionally, which would
        conflict with this function's penalty zone. Using both simultaneously
        would send contradictory signals past 24k chars.

    On negative rewards in GRPO:
        GRPO computes group-relative advantages, so the absolute value of a
        reward matters less than how one completion compares to others in the
        same group. A -0.5 on a rambling completion vs +0.3 on an efficient
        one is valid and useful gradient signal.
    """
    TARGET_START = 9000    # ~1500 words: minimum quality threshold
    CLIFF_START  = 24000   # ~6000 tokens: maximum efficiency threshold

    rewards = []
    for completion in completions:
        match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        length = len(match.group(1).strip())

        if length < TARGET_START:
            # Linear ramp: reward effort to reach ~1500 words
            score = 0.3 * (length / TARGET_START)
        elif length <= CLIFF_START:
            # Plateau: deep thinking is fine, no further reward or penalty
            score = 0.3
        else:
            # Penalty: subtract 0.1 per ~1000 excess tokens, floor at -0.5
            excess_penalty = 0.1 * ((length - CLIFF_START) / 4000)
            score = max(-0.5, 0.3 - excess_penalty)

        rewards.append(score)
    return rewards