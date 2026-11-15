"""
src/search.py - Path aggregation, Best-of-N search, and compute-matched search.

Reference key used in comments:
  [L<n>]   = lecture "008-Inference-time Improvement and Reasoning", slide n
  [PA3 §x] = assignment handout PA3.pdf, section x

Everything here is the "select well" half of "generate more, then select well" [L8].
"""
import math  # math.prod for the product aggregation
import time  # wall-clock timing
from typing import Any, Callable  # type hints only

from .generation import (  # reuse the baseline building blocks
    _resolve_generate_fn,  # real HF generate or test mock
    compute_majority_vote,  # tie-breaking and SC voting
    format_step_prompt,  # same prompt for every method
    run_self_consistency,  # samples the N candidates for standalone Best-of-N
    sample_candidates,  # batched sampling used by compute-matched SC
)
from .prm_judge import score_path  # step-level PRM scoring

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]

_VALID_STRATEGIES = ("min", "product", "step_penalty")  # the three rules of [PA3 §2.4]
_TIE_TOL = 1e-9  # scores closer than this count as tied [PA3 §5.3 Deliv 3.1]


def aggregate_path_score(
    step_scores: list[float],
    strategy: str = "min",
    penalty_lambda: float = 2.0
) -> float:
    """
    Aggregates a list of step scores (+1.0 or -1.0) into a scalar trajectory score.

    Rules:
      - An empty list returns -1.0 for every strategy.
      - 'min': min(step_scores).
      - 'product': p_t = clamp((r_t + 1)/2, 0.01, 0.99); returns the raw product.
      - 'step_penalty': mean(step_scores) - penalty_lambda * count(r_t == -1.0).
      - An unknown strategy raises ValueError.
    """
    if strategy not in _VALID_STRATEGIES:  # typo or unsupported rule
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")  # required by the contract
    if not step_scores:  # a path with no parsable steps cannot be trusted
        return -1.0

    if strategy == "min":  # weakest link, eq. 5: one INVALID step sinks the whole path
        return float(min(step_scores))

    if strategy == "product":  # joint validity, eq. 6
        probs = [min(max((r + 1.0) / 2.0, 0.01), 0.99) for r in step_scores]  # +1 -> 0.99, -1 -> 0.01
        return float(math.prod(probs))  # raw product (not the log); each error multiplies by 0.01

    # strategy == "step_penalty", eq. 7: average reward minus lambda per invalid step
    mean_score = sum(step_scores) / len(step_scores)  # 1.0 for a fully valid path at any length
    num_invalid = sum(1 for r in step_scores if r == -1.0)  # how many steps the judge rejected
    return float(mean_score - penalty_lambda * num_invalid)  # bigger lambda = harsher per error


def select_best_path(
    candidate_paths: list[dict[str, Any]],
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0
) -> dict[str, Any]:
    """
    Selects the winning candidate path from candidate_paths.

    Input Dictionary Contract:
      Each candidate dictionary in candidate_paths must contain 'step_scores' (list[float])
      and 'answer' (int | None). Any other keys present in the candidate dictionary
      are passed through unchanged in the returned copy.

    Selection & Tie-Breaking Rules:
      1. Compute path_score for each candidate with aggregate_path_score.
         A candidate with an empty step_scores list gets -1.0.
      2. Eligibility: if at least one candidate has a non-None answer, candidates
         with answer=None are excluded. If all answers are None, all remain eligible.
      3. Find maximum path_score among eligible candidates (tolerance 1e-9).
         The tied set is all eligible candidates within tolerance of the maximum.
      4. If the tied set has more than one candidate, compute compute_majority_vote
         over their answers. Return the lowest-index candidate in the tied set whose
         answer equals the vote. If the vote is None, return the lowest-index tied candidate.
      5. Returns a shallow COPY of the chosen dict with two added keys:
         'path_score' (float) and 'num_tied' (int, size of the tied set).
         Does not mutate the inputs.
    """
    if not candidate_paths:  # nothing to choose from
        raise ValueError("select_best_path needs at least one candidate")

    scores = [  # step 1: one scalar per candidate
        aggregate_path_score(list(c.get("step_scores") or []), aggregation_strategy, penalty_lambda)
        for c in candidate_paths
    ]

    any_answer = any(c.get("answer") is not None for c in candidate_paths)  # step 2: is there any real answer?
    eligible = [  # indices allowed to win
        i for i, c in enumerate(candidate_paths)
        if c.get("answer") is not None or not any_answer  # drop answer=None unless everyone is None
    ]

    best = max(scores[i] for i in eligible)  # step 3: best score among eligible paths
    tied = [i for i in eligible if abs(scores[i] - best) <= _TIE_TOL]  # all paths at that best score

    chosen = tied[0]  # default: lowest-index tied candidate
    if len(tied) > 1:  # step 4: filter-then-vote [PA3 §5.3 "Why this matters"]
        vote = compute_majority_vote([candidate_paths[i].get("answer") for i in tied])  # vote among survivors
        if vote is not None:  # some tied candidate has an answer
            chosen = next(i for i in tied if candidate_paths[i].get("answer") == vote)  # first one with it

    result = dict(candidate_paths[chosen])  # step 5: shallow copy, input left untouched
    result["path_score"] = float(scores[chosen])  # the winner's aggregated score
    result["num_tied"] = len(tied)  # how many candidates shared the best score (tie rate in the ablation)
    return result


def run_prm_best_of_n(
    model: Any = None,
    judge_model: Any = None,
    gen_tokenizer: Any = None,
    judge_tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0,
    candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None,
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Executes PRM-guided Best-of-N search over reasoning trajectories.

    If candidates is provided (shared candidate pool), does NOT sample new candidates
    (generation_calls = 0, generated_tokens = 0). Provided candidates use the
    run_self_consistency candidate format (at minimum 'steps' and 'answer').
    Otherwise, samples n candidates using run_self_consistency.

    Scores each candidate's steps using score_path.
    Selects the winning candidate via select_best_path.

    Lecture link: "Sample and aggregate: draw many chains and vote OR SCORE" [L9];
    here we score with an external verifier (system lever [L6-L7]).

    Returns:
      selected, answer, candidates [each augmented with step_scores, rationales],
      generation_calls, judge_calls, generated_tokens, judge_tokens, time_seconds
    """
    start = time.perf_counter()  # start the clock
    generation_calls = 0  # stays 0 when a shared pool is supplied
    generated_tokens = 0  # stays 0 when a shared pool is supplied

    if candidates is None:  # standalone mode: draw our own N samples
        sc = run_self_consistency(model, gen_tokenizer, question, n=n, generate_fn=generate_fn)  # N sampled chains
        candidates = sc["candidates"]  # the pool to be judged
        generation_calls = sc["generation_calls"]  # N calls
        generated_tokens = sc["generated_tokens"]  # their output tokens

    if judge_fn is None:  # no mock injected -> build the real 7B judge once (not once per candidate)
        from .utils import make_hf_judge_fn  # lazy import
        judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)

    scored: list[dict[str, Any]] = []  # candidates augmented with step scores
    judge_calls = 0  # sum of K_i over candidates [PA3 §5.3]
    judge_tokens = 0  # judge output tokens
    for cand in candidates:  # judge every path fully (no early stopping)
        res = score_path(judge_model, judge_tokenizer, question, cand, judge_fn)  # per-step verdicts
        augmented = dict(cand)  # copy so the caller's dicts are not mutated
        augmented["step_scores"] = res["step_scores"]  # r_1..r_T
        augmented["rationales"] = res["rationales"]  # reasons
        scored.append(augmented)  # keep in original order
        judge_calls += res["judge_calls"]  # bookkeeping
        judge_tokens += res["judge_tokens"]  # bookkeeping

    selected = select_best_path(scored, aggregation_strategy, penalty_lambda)  # PRM ranking + vote tie-break

    return {
        "selected": selected,  # winning candidate (with path_score, num_tied)
        "answer": selected.get("answer"),  # the PRM prediction
        "candidates": scored,  # all candidates with step_scores/rationales (reused for other strategies)
        "generation_calls": generation_calls,  # new samples drawn here only
        "judge_calls": judge_calls,  # total step evaluations
        "generated_tokens": generated_tokens,  # new samples only
        "judge_tokens": judge_tokens,  # total judge output tokens
        "time_seconds": time.perf_counter() - start,  # elapsed wall-clock time
    }


def run_compute_matched_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    target_calls: int = 15,
    temperature: float = 0.7,
    existing_candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes compute-matched Self-Consistency aggregating target_calls paths.
    If existing_candidates are supplied, samples only the deficit:
    target_calls - len(existing_candidates). Returned candidates = existing_candidates followed by new samples, in sampling order.

    Why: PRM Best-of-5 costs ~5 generation + ~25 judge calls, so the fair comparison is
    "spend the same budget on more samples instead" (N_SC = 15) [PA3 §5.3 Deliv 3.3] -
    the computation lever pulled harder instead of the system lever [L6-L8].

    Returns:
      candidates, answer, generation_calls (new samples only), total_samples,
      generated_tokens (new samples only), time_seconds
    """
    start = time.perf_counter()  # start the clock
    existing = list(existing_candidates or [])[:target_calls]  # reuse the shared SC@5 pool (indices 0-4)
    deficit = max(0, target_calls - len(existing))  # how many new samples are still needed (10 for 5 -> 15)

    new_candidates: list[dict[str, Any]] = []  # newly drawn chains (indices 5-14)
    new_tokens = 0  # their output tokens
    if deficit > 0:  # only touch the generator if we actually need more samples
        gen = _resolve_generate_fn(model, tokenizer, generate_fn)  # real HF generate or test mock
        prompt = format_step_prompt(question)  # identical prompt to SC@5
        new_candidates, new_tokens = sample_candidates(prompt, deficit, temperature, gen)  # batched sampling

    all_candidates = existing + new_candidates  # sampling order preserved: old first, then new
    answer = compute_majority_vote([c.get("answer") for c in all_candidates])  # vote over all 15

    return {
        "candidates": all_candidates,  # 15 chains
        "answer": answer,  # SC@15 prediction
        "generation_calls": len(new_candidates),  # new samples only
        "total_samples": len(all_candidates),  # = target_calls
        "generated_tokens": new_tokens,  # new samples only
        "time_seconds": time.perf_counter() - start,  # time to sample the extra candidates
    }
