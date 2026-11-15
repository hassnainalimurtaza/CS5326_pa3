"""
src/generation.py - Prompt formatting and baseline generation routines.

Reference key used in comments:
  [L<n>]   = lecture "008-Inference-time Improvement and Reasoning", slide n
  [PA3 §x] = assignment handout PA3.pdf, section x
"""
import time  # wall-clock timing for the "time_seconds" field [PA3 §5.4]
from typing import Any, Callable  # type hints only

from .extraction import extract_answer, parse_steps  # step splitter + final-integer parser [PA3 §4.3]
from .utils import GREEDY_GEN_KWARGS, SAMPLING_GEN_KWARGS  # course-mandated decoding settings [PA3 §3.4 box]

GenerateFn = Callable[[str, int, dict], list[dict]]
# generate_fn(raw_prompt, num_samples, gen_kwargs) -> list of {"text": str, "num_tokens": int}
# gen_kwargs carries do_sample/temperature/top_p/top_k/repetition_penalty/max_new_tokens


# The zero-shot instruction. This is the INPUT lever: we change what the model is asked,
# not its weights [L6-L7]. Asking for written steps = "produce more tokens", i.e. give the
# model more computation per problem, because every written token is another forward pass [L8].
_STEP_PROMPT_TEMPLATE = (
    "Solve the following math word problem by reasoning step by step.\n"
    "\n"
    "Follow these formatting rules exactly:\n"
    "1. Write every reasoning step on its own line, starting with \"Step 1:\", then \"Step 2:\", and so on.\n"
    "2. Each step must contain exactly one deduction or one arithmetic calculation, and must show the numbers used.\n"
    "3. Use plain text only. Do NOT use markdown, headings, bold text, bullet points, LaTeX, or \\boxed{{}}.\n"
    "4. After the last step, write one final line in exactly this form:\n"
    "Final Answer: <integer>\n"
    "   The final line must contain only the integer: no units, no dollar sign, no words.\n"
    "\n"
    "Example of the required format:\n"
    "Step 1: Janet starts with 16 eggs per day.\n"
    "Step 2: She uses 3 + 4 = 7 eggs.\n"
    "Step 3: She sells 16 - 7 = 9 eggs.\n"
    "Step 4: She earns 9 * 2 = 18 dollars.\n"
    "Final Answer: 18\n"
    "\n"
    "Problem: {question}\n"
)


def format_step_prompt(question: str) -> str:
    """
    Formats the raw problem question into a prompt that instructs the
    model to emit numbered steps ('Step 1: ...') and 'Final Answer: <int>'.

    Returns a raw instruction string (no chat template applied).
    The chat template is added later by utils.make_hf_generate_fn.
    """
    return _STEP_PROMPT_TEMPLATE.format(question=str(question).strip())  # put the question into the template


def compute_majority_vote(answers: list[int | None]) -> int | None:
    """
    Given a list of extracted answers, returns the majority answer.
    Ignores None values. Breaks ties deterministically in favor of the answer
    that appeared earliest in the list.

    This is the "aggregate" half of "sample and aggregate" [L9]: wrong chains tend to be
    wrong in different ways, right chains agree, so the most common answer is a good pick
    ("generate more, then select well" [L8]; Wang et al. 2023, eq. 8 in [PA3 §5.1]).
    """
    counts: dict[int, int] = {}  # answer -> how many samples produced it
    first_seen: dict[int, int] = {}  # answer -> index of its first appearance (for tie-breaking)
    for idx, ans in enumerate(answers or []):  # walk the answers in sampling order
        if ans is None:  # format failures do not get a vote
            continue
        counts[ans] = counts.get(ans, 0) + 1  # one more vote for this answer
        first_seen.setdefault(ans, idx)  # remember only the FIRST position of this answer
    if not counts:  # empty list or all None
        return None  # nobody to vote for
    # Highest count wins; among equal counts the smaller first index wins (hence the minus sign).
    return max(counts, key=lambda a: (counts[a], -first_seen[a]))


def _resolve_generate_fn(model: Any, tokenizer: Any, generate_fn: GenerateFn | None) -> GenerateFn:
    """Uses the injected generate_fn (mock in tests) or builds the real Hugging Face one."""
    if generate_fn is not None:  # tests and the pipeline pass a callable directly
        return generate_fn
    from .utils import make_hf_generate_fn  # lazy import: only needed with a real model
    return make_hf_generate_fn(model, tokenizer)  # wraps model.generate() with the Qwen chat template


def sample_candidates(
    prompt: str,
    n: int,
    temperature: float,
    generate_fn: GenerateFn,
) -> tuple[list[dict[str, Any]], int]:
    """
    Draws n stochastic completions in ONE batched call and turns each into a candidate dict.
    Shared by self-consistency, PRM Best-of-N and compute-matched SC so all use identical settings.
    Returns (candidates, total_generated_tokens).
    """
    if n <= 0:  # nothing to sample (e.g. compute-matched SC with no deficit)
        return [], 0
    gen_kwargs = dict(SAMPLING_GEN_KWARGS)  # copy so we never modify the shared constant
    gen_kwargs["temperature"] = temperature  # T=0.7 by default: sharper than T=1, still diverse [L16-L17]
    # SAMPLING_GEN_KWARGS also sets top_p=1.0 and top_k=0, i.e. truncation is switched OFF [L18-L19],
    # so temperature is the only control on randomness ("pure temperature sampling" [PA3 §3.4 box]).
    outputs = generate_fn(prompt, n, gen_kwargs)  # one batched call (num_return_sequences=n inside)
    candidates = []  # one dict per sampled reasoning chain
    total_tokens = 0  # output tokens across all samples (prompt tokens excluded) [PA3 §5.3 bookkeeping]
    for out in outputs:  # each sampled chain of thought
        text = out.get("text", "") or ""  # raw completion text
        num_tokens = int(out.get("num_tokens", 0) or 0)  # tokens counted from token IDs, not words
        candidates.append({
            "completion": text,  # raw model output (kept for the report / judge audit)
            "steps": parse_steps(text),  # list of step strings for the PRM judge
            "answer": extract_answer(text),  # final integer or None
            "num_tokens": num_tokens,  # cost of this sample
        })
        total_tokens += num_tokens  # accumulate generation cost
    return candidates, total_tokens


def run_greedy(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes greedy decoding for a single question.
    Calls generate_fn with num_samples=1 and gen_kwargs containing do_sample=False,
    temperature=None, top_p=None, top_k=None.

    Greedy = always take the highest-probability token [L14]. Deterministic, so it is
    "one chain of reasoning, one sample, and nothing checks it" [L9 Prompting row].

    Returns:
      prompt, completion, steps, answer, generation_calls, generated_tokens, time_seconds
    """
    start = time.perf_counter()  # start the clock
    gen = _resolve_generate_fn(model, tokenizer, generate_fn)  # real HF generate or test mock
    prompt = format_step_prompt(question)  # step-structured instruction
    gen_kwargs = dict(GREEDY_GEN_KWARGS)  # do_sample=False, max_new_tokens=512
    gen_kwargs.update({"do_sample": False, "temperature": None, "top_p": None, "top_k": None})  # override Qwen's sampling defaults [PA3 §3.4 box]
    outputs = gen(prompt, 1, gen_kwargs)  # exactly one deterministic completion
    first = outputs[0] if outputs else {"text": "", "num_tokens": 0}  # be safe if nothing came back
    completion = first.get("text", "") or ""  # raw text
    return {
        "prompt": prompt,  # the instruction we sent
        "completion": completion,  # what the model wrote
        "steps": parse_steps(completion),  # split into steps
        "answer": extract_answer(completion),  # final integer or None
        "generation_calls": 1,  # one generated sequence = one call [PA3 §5.3 bookkeeping]
        "generated_tokens": int(first.get("num_tokens", 0) or 0),  # output tokens only
        "time_seconds": time.perf_counter() - start,  # elapsed wall-clock time
    }


def run_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    temperature: float = 0.7,
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Samples n diverse reasoning trajectories and aggregates candidate answers via majority vote.

    This is the COMPUTATION lever: "make more calls ... and combine the results" [L8],
    the "Sample and aggregate" row of the roadmap [L9]. Sampling (not greedy) is needed
    so the n chains actually differ [L15].

    Returns:
      candidates [each: completion, steps, answer, num_tokens], answer,
      generation_calls, generated_tokens, time_seconds
    """
    start = time.perf_counter()  # start the clock
    gen = _resolve_generate_fn(model, tokenizer, generate_fn)  # real HF generate or test mock
    prompt = format_step_prompt(question)  # same prompt as greedy
    candidates, total_tokens = sample_candidates(prompt, n, temperature, gen)  # n stochastic chains
    answer = compute_majority_vote([c["answer"] for c in candidates])  # most common final answer
    return {
        "candidates": candidates,  # all n chains (reused later by PRM Best-of-N) [PA3 §5.3 pool sharing]
        "answer": answer,  # majority-vote prediction
        "generation_calls": len(candidates),  # n sequences = n calls, even though batched [PA3 §5.3]
        "generated_tokens": total_tokens,  # sum of output tokens of all n samples
        "time_seconds": time.perf_counter() - start,  # elapsed wall-clock time
    }
