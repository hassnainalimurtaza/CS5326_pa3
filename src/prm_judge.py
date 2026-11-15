"""
src/prm_judge.py - PRM judge prompt construction, parsing, and step scoring.

Reference key used in comments:
  [L<n>]   = lecture "008-Inference-time Improvement and Reasoning", slide n
  [PA3 §x] = assignment handout PA3.pdf, section x

The judge is the SYSTEM lever: a verifier placed around the frozen generator [L6-L7].
Using a bigger 7B judge for a 1.5B generator addresses the roadmap's warning that
"the evaluator is usually the same model that made the mistake" [L9 Search row; PA3 §3.1.1].
"""
import re  # regular expressions for verdict/rationale parsing
from typing import Any, Callable  # type hints only

JudgeFn = Callable[[str], dict]
# judge_fn(judge_prompt) -> {"text": str, "num_tokens": int}


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    """
    Builds the verification prompt for the PRM judge to classify current_step
    as VALID or INVALID given the problem question and preceding_steps.

    Judge Semantics (Local Validity):
    The judge evaluates current_step ASSUMING ALL PRECEDING STEPS ARE CORRECT.
    A step is VALID if its arithmetic is sound and it follows logically from the
    problem statement plus preceding steps as written. An error in an earlier step
    does NOT make subsequent steps invalid by itself.

    Autograder Literal Checks:
    - Preceding steps are rendered as 'Step 1: <text>', 'Step 2: <text>', ... in order.
    - The current step is rendered with the label 'Current Step:'.
    - The prompt contains the phrase 'assume all preceding steps are correct' (case-insensitive).
    - The prompt contains the literal strings 'Verdict: VALID', 'Verdict: INVALID', and 'Rationale:'.
    - With an empty preceding_steps list, the prompt states '(no preceding steps)'.
    """
    if preceding_steps:  # there is earlier work to show the judge
        preceding_block = "\n".join(  # one line per earlier step
            f"Step {i}: {step}" for i, step in enumerate(preceding_steps, start=1)  # 'Step 1: ...', 'Step 2: ...'
        )
    else:  # judging the very first step
        preceding_block = "(no preceding steps)"  # exact literal required by the autograder [PA3 §5.2]

    return (
        "You are a careful math teacher. Your job is to verify ONE step of a student's solution.\n"
        "\n"
        f"Problem:\n{question}\n"  # the original word problem
        "\n"
        f"Preceding steps:\n{preceding_block}\n"  # context the step may rely on
        "\n"
        f"Current Step: {current_step}\n"  # the single step being judged
        "\n"
        "Instructions:\n"
        # Local validity semantics [PA3 §2.2]: judge the step, not the whole solution.
        "- Assume all preceding steps are correct, even if some of them contain mistakes.\n"
        "- Judge ONLY the current step. It is valid if every calculation in it is arithmetically correct "
        "and it follows logically from the problem statement and the preceding steps as written.\n"
        "- Recompute every number in the current step yourself before you decide.\n"
        "- A step that correctly restates information from the problem is valid.\n"
        "\n"
        "Reply with exactly two lines in this format and nothing else:\n"
        "Verdict: <VALID or INVALID>\n"
        "Rationale: <one short sentence>\n"
        "\n"
        # Binary decision, not a 1-5 scale: soundness is all-or-nothing [PA3 §2.3].
        "Write \"Verdict: VALID\" if the current step is correct, or \"Verdict: INVALID\" "
        "if it contains an arithmetic or logical error.\n"
    )


# Rule 1: an explicit verdict line, e.g. 'Verdict: VALID' or 'verdict: **invalid**' [PA3 §5.2 Deliv 2.3].
_VERDICT_RE = re.compile(r"verdict\s*:\s*\**\s*(VALID|INVALID)\b", re.IGNORECASE)
# Rule 2: the standalone word INVALID or the phrase NOT VALID. \b avoids the "INVALID contains VALID" trap.
_NEGATIVE_RE = re.compile(r"\b(?:INVALID|NOT\s+VALID)\b", re.IGNORECASE)
# Rule 3: the standalone word VALID.
_POSITIVE_RE = re.compile(r"\bVALID\b", re.IGNORECASE)
# Rationale: whatever follows 'Rationale:' on the same line (bold markers tolerated).
_RATIONALE_RE = re.compile(r"rationale\s*\**\s*:\s*\**\s*(.*)", re.IGNORECASE)
# Sentence end: '.', '!' or '?' followed by whitespace or end of text, so the '.' in '2.5' is not an end.
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")


def _extract_rationale(judge_output: str) -> str:
    """Returns the first sentence after 'Rationale:' (or up to the newline), stripped; '' if absent."""
    match = _RATIONALE_RE.search(judge_output)  # find the rationale label
    if match is None:  # judge wrote no rationale
        return ""
    line = match.group(1)  # rest of that line ('.' in the regex never crosses a newline)
    end = _SENTENCE_END_RE.search(line)  # first real sentence end
    if end is not None:  # there is a sentence terminator on this line
        line = line[: end.end()]  # keep up to and including the '.', '!' or '?'
    return line.strip()  # trim surrounding whitespace


def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    r"""
    Parses the judge model's raw text completion into (score, rationale).

    Rules:
      1. If a line matches /verdict\s*:\s*\**\s*(VALID|INVALID)\b/i, use it.
         If several exist, use the first.
      2. Otherwise, if standalone word INVALID or phrase 'NOT VALID' appears
         (case-insensitive, word boundaries) -> -1.0.
      3. Otherwise, if standalone word VALID appears -> +1.0.
      4. Otherwise -> -1.0 (unparseable defaults to INVALID).

    Warning: A naive '"VALID" in text' check is wrong because "INVALID" contains "VALID".

    Rationale: Text after 'Rationale:' up to the first sentence end (".", "!", or "?"
    followed by whitespace or end of string, so decimals like "2.5" are not split),
    or a newline, whichever comes first, stripped. If absent, returns empty string "".
    """
    if not isinstance(judge_output, str) or not judge_output:  # None / empty judge output
        return -1.0, ""  # rule 4: unparseable defaults to INVALID

    rationale = _extract_rationale(judge_output)  # one-sentence justification (may be "")

    verdict = _VERDICT_RE.search(judge_output)  # rule 1: first explicit 'Verdict:' line
    if verdict is not None:  # the judge followed the format
        score = 1.0 if verdict.group(1).upper() == "VALID" else -1.0  # VALID -> +1, INVALID -> -1 (eq. 3-4)
        return score, rationale
    if _NEGATIVE_RE.search(judge_output):  # rule 2: free text says INVALID / NOT VALID
        return -1.0, rationale
    if _POSITIVE_RE.search(judge_output):  # rule 3: free text says VALID (as a whole word)
        return 1.0, rationale
    return -1.0, rationale  # rule 4: no decision found -> be conservative, treat as INVALID


def score_path(
    judge_model: Any = None,
    judge_tokenizer: Any = None,
    question: str = "",
    candidate: dict[str, Any] | None = None,
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Sequentially scores each step in candidate["steps"] using judge_fn.
    Step t is judged with preceding_steps = steps[:t].

    The student's code MUST call build_prm_judge_prompt to build the string passed to judge_fn,
    and MUST call parse_judge_decision on the returned text.

    This turns the judge into a Process Reward Model: one reward r_t in {-1,+1} per step
    (eq. 3, [PA3 §2.2]) instead of one reward for the final answer (an ORM).

    Returns:
      step_scores, rationales, judge_calls, judge_tokens
    """
    if judge_fn is None:  # no mock injected -> use the real 7B judge
        from .utils import make_hf_judge_fn  # lazy import: only needed with a real model
        judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)  # greedy, max_new_tokens=96 [PA3 §3.4 box]

    steps = list((candidate or {}).get("steps") or [])  # the parsed reasoning steps of this chain
    step_scores: list[float] = []  # r_1 ... r_T
    rationales: list[str] = []  # judge's one-sentence reason for each step
    judge_tokens = 0  # judge OUTPUT tokens only [PA3 §5.3 bookkeeping]

    for t, current_step in enumerate(steps):  # every step is judged (no early stopping)
        prompt = build_prm_judge_prompt(question, steps[:t], current_step)  # context = steps before t
        out = judge_fn(prompt) or {}  # one judge call per step
        score, rationale = parse_judge_decision(out.get("text", "") or "")  # text -> (+1/-1, reason)
        step_scores.append(score)  # record the step reward
        rationales.append(rationale)  # record why
        judge_tokens += int(out.get("num_tokens", 0) or 0)  # accumulate judge cost

    return {
        "step_scores": step_scores,  # per-step rewards
        "rationales": rationales,  # per-step reasons (used in the judge audit, report §3)
        "judge_calls": len(steps),  # K_i judge calls for this candidate [PA3 §5.3]
        "judge_tokens": judge_tokens,  # total judge output tokens
    }
