    """
    training/reward_functions.py

    Reward functions for GRPO training on DeepMath-103K.

    Three signals, in order of importance:
    1. accuracy_reward   (weight 1.0) — did the model get the right answer?
    2. format_reward     (weight 0.5) — did the model use the correct tag structure?
    3. length_reward     (weight 0.3) — did the model produce non-trivial reasoning?

    A correct, well-formatted, well-reasoned answer scores ~1.8.
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
    # Reward 3: Reasoning length (weight 0.3)
    # ---------------------------------------------------------------------------

    def length_reward(completions, **kwargs):
        """
        Encourages non-trivial reasoning chains without rewarding bloat.

        The reward scales linearly from 0 at 50 chars to 0.3 at 1000+ chars
        of content inside <think> tags. The cap at 0.3 means the model can't
        offset a wrong answer by rambling. It just slightly rewards showing work.

        Why character count and not word count?
        Math LaTeX is character-heavy. A single expression like \\frac{a+b}{c}
        is one "word" but substantial reasoning. Characters are more faithful.
        """
        rewards = []
        for completion in completions:
            think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
            if think_match:
                length = len(think_match.group(1).strip())
                if length < 50:
                    rewards.append(0.0)
                elif length >= 1000:
                    rewards.append(0.3)
                else:
                    # Linear scale between 50 and 1000 chars
                    rewards.append(0.3 * (length - 50) / 950)
            else:
                rewards.append(0.0)
        return rewards