"""
run_all.py - End-to-end PA3 evaluation pipeline (run top to bottom on Kaggle GPU T4x2).

Usage (from the repository root):
  python run_all.py --check-format --limit 50   # greedy only: check "Final Answer:" compliance first [PA3 §4.2]
  python run_all.py --limit 2                   # quick smoke run on 2 problems
  python run_all.py                             # full 50-problem benchmark (~1-2 h)
  python run_all.py --artifacts-only            # rebuild outputs/*.json from outputs/per_problem.jsonl

Reference key used in comments:
  [L<n>]   = lecture "008-Inference-time Improvement and Reasoning", slide n
  [PA3 §x] = assignment handout PA3.pdf, section x
"""
import argparse  # command-line flags
import csv  # judge-audit spreadsheet for report §3
import json  # reading the dataset and writing artifacts
import os  # paths
import platform  # Python version for the reproducibility section
import random  # bootstrap resampling
import sys  # sys.path tweak so "src" imports work from any cwd
import time  # timing

ROOT = os.path.dirname(os.path.abspath(__file__))  # repository root (folder containing this file)
sys.path.insert(0, ROOT)  # make "import src..." work no matter where python was launched
os.chdir(ROOT)  # all relative paths (data/, outputs/) are relative to the repo root [PA3 §3.4]

from src.generation import compute_majority_vote, run_greedy, run_self_consistency  # Problem 1
from src.search import (  # Problems 2-3
    run_compute_matched_self_consistency,
    run_prm_best_of_n,
    select_best_path,
)
from src.utils import (  # starter helpers (model loading, seeds, artifact writers)
    GLOBAL_SEED,
    load_models,
    make_hf_generate_fn,
    make_hf_judge_fn,
    save_aggregation_ablation,
    save_results_summary,
    save_sc_scaling_curve,
    set_seed_for_problem,
)

DATA_PATH = "data/gsm8k_test_50.jsonl"  # the fixed 50-problem benchmark [PA3 §4.1]
OUT_DIR = "outputs"  # where every artifact goes
PER_PROBLEM_PATH = os.path.join(OUT_DIR, "per_problem.jsonl")  # one line per problem, written incrementally
N_SHARED = 5  # size of the shared candidate pool (SC@5 and all PRM Best-of-5 variants) [PA3 §5.3 pool sharing]
N_MATCHED = 15  # compute-matched SC budget [PA3 §5.3 Deliv 3.3]
TEMPERATURE = 0.7  # sampling temperature [L16-L17; PA3 §3.4 box]
MAIN_LAMBDA = 2.0  # default step-penalty lambda [PA3 §2.4 eq. 7]
ABLATION_LAMBDAS = (0.0, 0.5, 2.0)  # lambda sweep for the ablation [PA3 §5.4]

# Exact method strings required in results_summary.json [PA3 §5.4 table].
M_GREEDY = "Greedy (N=1)"
M_SC5 = "Self-Consistency (N=5)"
M_SC15 = "Compute-Matched SC (N=15)"
M_PRM_MIN = "PRM Best-of-5 (min)"
M_PRM_PROD = "PRM Best-of-5 (product)"
M_PRM_PEN = "PRM Best-of-5 (step_penalty, lambda=2.0)"
METHODS = [M_GREEDY, M_SC5, M_SC15, M_PRM_MIN, M_PRM_PROD, M_PRM_PEN]  # row order of the summary table
PRED_KEY = {  # method name -> key inside record["predictions"]
    M_GREEDY: "greedy", M_SC5: "sc5", M_SC15: "sc15",
    M_PRM_MIN: "prm_min", M_PRM_PROD: "prm_product", M_PRM_PEN: "prm_step_penalty",
}


# ----------------------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------------------
def load_problems(path: str) -> list[dict]:
    """Reads the JSONL benchmark: one {"id", "question", "ground_truth_answer"} per line."""
    with open(path, encoding="utf-8") as f:  # open the dataset
        return [json.loads(line) for line in f if line.strip()]  # skip blank lines


def load_done_records(path: str) -> dict[str, dict]:
    """Loads already-finished problems so a disconnected Kaggle session can resume [PA3 §3.4 box]."""
    done: dict[str, dict] = {}  # problem id -> record
    if not os.path.exists(path):  # first run: nothing finished yet
        return done
    with open(path, encoding="utf-8") as f:  # read finished records
        for line in f:  # one problem per line
            line = line.strip()  # ignore trailing newline
            if not line:  # skip blank lines
                continue
            try:
                rec = json.loads(line)  # parse the record
            except json.JSONDecodeError:  # a half-written last line after a crash
                continue  # that problem will simply be redone
            done[rec["id"]] = rec  # remember it
    return done


def append_record(path: str, record: dict) -> None:
    """Appends one finished problem and flushes immediately (checkpointing) [PA3 §3.4 box]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # make sure outputs/ exists
    with open(path, "a", encoding="utf-8") as f:  # append mode keeps earlier problems
        f.write(json.dumps(record, ensure_ascii=False) + "\n")  # one JSON object per line
        f.flush()  # push to disk now, not at exit
        os.fsync(f.fileno())  # survive a sudden kernel restart


# ----------------------------------------------------------------------------------------
# Per-problem experiment
# ----------------------------------------------------------------------------------------
def solve_problem(index: int, problem: dict, gen_fn, judge_fn) -> dict:
    """Runs all six configurations on one problem, sharing samples and judge scores."""
    question = problem["question"]  # the word problem
    set_seed_for_problem(index)  # seed 5326 + index -> reproducible and resumable [PA3 §3.4 box]

    # 1) Greedy baseline: readout lever, highest-probability token every step [L14].
    greedy = run_greedy(question=question, generate_fn=gen_fn)

    # 2) Shared pool of 5 sampled chains = SC@5 (computation lever: more calls [L8]).
    sc5 = run_self_consistency(question=question, n=N_SHARED, temperature=TEMPERATURE, generate_fn=gen_fn)

    # 3) Judge the SAME 5 chains once (system lever: external verifier [L6-L7, L9]).
    prm = run_prm_best_of_n(
        question=question, n=N_SHARED, aggregation_strategy="min",
        candidates=sc5["candidates"], judge_fn=judge_fn,  # shared pool -> no new generation
    )
    scored = prm["candidates"]  # 5 candidates now carrying step_scores and rationales

    # 4) Re-use those step scores for all three aggregation rules (no new judge calls) [PA3 §5.3].
    selections, selection_time = {}, {}  # strategy -> selected candidate / seconds spent selecting
    for key, strategy, lam in (("prm_min", "min", MAIN_LAMBDA),
                               ("prm_product", "product", MAIN_LAMBDA),
                               ("prm_step_penalty", "step_penalty", MAIN_LAMBDA)):
        t0 = time.perf_counter()  # time only the aggregation + selection
        selections[key] = select_best_path(scored, aggregation_strategy=strategy, penalty_lambda=lam)
        selection_time[key] = time.perf_counter() - t0  # tiny, but reported per the time rules [PA3 §5.4 note]

    # 5) Compute-matched SC@15: keep the 5 shared samples, draw 10 more [PA3 §5.3 pool sharing].
    sc15 = run_compute_matched_self_consistency(
        question=question, target_calls=N_MATCHED, temperature=TEMPERATURE,
        existing_candidates=sc5["candidates"], generate_fn=gen_fn,
    )
    extra = sc15["candidates"][N_SHARED:]  # the 10 new candidates (indices 5-14)
    all_candidates = scored + extra  # 15 candidates in sampling order; first 5 carry judge info
    answers = [c.get("answer") for c in all_candidates]  # final integers (or None) in sampling order

    # 6) Scaling curve: majority vote over the first N samples, N = 1..15 [PA3 §5.3 pool sharing].
    curve_answers = [compute_majority_vote(answers[:n]) for n in range(1, N_MATCHED + 1)]

    return {
        "id": problem["id"],  # e.g. "gsm8k_test_01"
        "index": index,  # position in the benchmark (seed = 5326 + index)
        "question": question,  # problem text
        "ground_truth_answer": problem["ground_truth_answer"],  # gold integer
        "greedy": {  # greedy output
            "completion": greedy["completion"],
            "steps": greedy["steps"],
            "answer": greedy["answer"],
            "generated_tokens": greedy["generated_tokens"],
            "time_seconds": greedy["time_seconds"],
        },
        "candidates": all_candidates,  # 15 candidates; step_scores + rationales for the first 5
        "sc5_time_seconds": sc5["time_seconds"],  # time to sample the shared 5
        "sc15_extra_time_seconds": sc15["time_seconds"],  # time to sample the extra 10
        "judge_calls": prm["judge_calls"],  # sum of steps over the 5 candidates
        "judge_tokens": prm["judge_tokens"],  # judge output tokens
        "judge_time_seconds": prm["time_seconds"],  # time spent judging
        "selection_time_seconds": selection_time,  # per-strategy selection time
        "num_tied": {k: v["num_tied"] for k, v in selections.items()},  # tie sizes per strategy
        "path_scores": {k: v["path_score"] for k, v in selections.items()},  # winning scores
        "predictions": {  # each method's final answer
            "greedy": greedy["answer"],
            "sc5": sc5["answer"],
            "sc15": sc15["answer"],
            "prm_min": selections["prm_min"].get("answer"),
            "prm_product": selections["prm_product"].get("answer"),
            "prm_step_penalty": selections["prm_step_penalty"].get("answer"),
        },
        "sc_curve_answers": curve_answers,  # SC@N prediction for N = 1..15
    }


# ----------------------------------------------------------------------------------------
# Metrics and artifacts
# ----------------------------------------------------------------------------------------
def pct(x: float) -> float:
    """Rounds a fraction to a percentage with 2 decimals."""
    return round(100.0 * x, 2)


def is_correct(pred, gold) -> bool:
    """Exact-match accuracy against the gold integer [PA3 §5.4]."""
    return pred is not None and int(pred) == int(gold)


def bootstrap_ci(correct: list[int], n_resamples: int = 1000, seed: int = GLOBAL_SEED) -> tuple[float, float]:
    """95% bootstrap confidence interval for accuracy (1,000 resamples, seed 5326) [PA3 §7.1]."""
    rng = random.Random(seed)  # fixed seed -> reproducible CI
    n = len(correct)  # number of problems
    if n == 0:  # nothing to resample
        return 0.0, 0.0
    accs = sorted(  # accuracy of each resampled benchmark
        sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)
    )
    lo = accs[int(0.025 * n_resamples)]  # 2.5th percentile
    hi = accs[min(n_resamples - 1, int(0.975 * n_resamples))]  # 97.5th percentile
    return pct(lo), pct(hi)


def build_artifacts(records: list[dict]) -> dict:
    """Writes results_summary, aggregation_ablation, sc_scaling_curve and report helper files."""
    records = sorted(records, key=lambda r: r["index"])  # benchmark order
    n = len(records)  # number of solved problems
    gold = [r["ground_truth_answer"] for r in records]  # gold answers

    # ---------- results_summary.json (6 rows) [PA3 §5.4] ----------
    greedy_tok = sum(r["greedy"]["generated_tokens"] for r in records)  # total greedy output tokens
    sc5_tok = sum(sum(c["num_tokens"] for c in r["candidates"][:N_SHARED]) for r in records)  # shared pool tokens
    sc15_tok = sum(sum(c["num_tokens"] for c in r["candidates"][:N_MATCHED]) for r in records)  # all 15 samples
    judge_calls = sum(r["judge_calls"] for r in records)  # total step evaluations
    judge_tok = sum(r["judge_tokens"] for r in records)  # total judge output tokens
    t_greedy = sum(r["greedy"]["time_seconds"] for r in records)  # greedy time
    t_sc5 = sum(r["sc5_time_seconds"] for r in records)  # SC@5 time
    t_sc15 = t_sc5 + sum(r["sc15_extra_time_seconds"] for r in records)  # SC@15 = SC@5 + 10 extra
    t_judge = sum(r["judge_time_seconds"] for r in records)  # judging time (shared by PRM rows)

    def sel_time(key: str) -> float:  # total selection time for one strategy
        return sum(r["selection_time_seconds"][key] for r in records)

    cost = {  # method -> (gen calls, judge calls, gen tokens, judge tokens, time) per the time rules [PA3 §5.4 note]
        M_GREEDY: (n, 0, greedy_tok, 0, t_greedy),
        M_SC5: (N_SHARED * n, 0, sc5_tok, 0, t_sc5),
        M_SC15: (N_MATCHED * n, 0, sc15_tok, 0, t_sc15),
        M_PRM_MIN: (N_SHARED * n, judge_calls, sc5_tok, judge_tok, t_sc5 + t_judge + sel_time("prm_min")),
        M_PRM_PROD: (N_SHARED * n, judge_calls, sc5_tok, judge_tok, t_sc5 + t_judge + sel_time("prm_product")),
        M_PRM_PEN: (N_SHARED * n, judge_calls, sc5_tok, judge_tok, t_sc5 + t_judge + sel_time("prm_step_penalty")),
    }
    correct_by_method = {  # method -> list of 0/1 per problem
        m: [int(is_correct(r["predictions"][PRED_KEY[m]], g)) for r, g in zip(records, gold)] for m in METHODS
    }
    summary = []  # rows of results_summary.json
    for m in METHODS:  # fixed row order
        gc, jc, gt, jt, secs = cost[m]  # unpack costs
        summary.append({
            "Method": m,
            "Accuracy (%)": pct(sum(correct_by_method[m]) / n),
            "Generation Calls": gc,
            "Judge Calls": jc,
            "Total LLM Calls": gc + jc,
            "Avg Gen Tokens / Problem": round(gt / n, 2),
            "Avg Judge Tokens / Problem": round(jt / n, 2),
            "Total Time (s)": round(secs, 2),
        })
    save_results_summary(summary, os.path.join(OUT_DIR, "results_summary.json"))  # schema-exact writer

    # ---------- aggregation_ablation.json [PA3 §5.4, report §2] ----------
    ablation = []  # rows of aggregation_ablation.json
    for strategy, lam in [("min", None), ("product", None)] + [("step_penalty", l) for l in ABLATION_LAMBDAS]:
        hits, ties = 0, 0  # correct selections / problems with more than one tied winner
        for r, g in zip(records, gold):  # re-select from the stored step scores (no new judge calls)
            sel = select_best_path(r["candidates"][:N_SHARED], strategy, MAIN_LAMBDA if lam is None else lam)
            hits += int(is_correct(sel.get("answer"), g))  # accuracy
            ties += int(sel["num_tied"] > 1)  # tie rate: how often voting had to break a tie
        ablation.append({"Strategy": strategy, "Lambda": lam, "Accuracy (%)": pct(hits / n), "Tie Rate (%)": pct(ties / n)})
    save_aggregation_ablation(ablation, os.path.join(OUT_DIR, "aggregation_ablation.json"))

    # ---------- sc_scaling_curve.json [PA3 §5.4] ----------
    curve = []  # rows of sc_scaling_curve.json
    for k in range(1, N_MATCHED + 1):  # N = 1..15
        acc = sum(is_correct(r["sc_curve_answers"][k - 1], g) for r, g in zip(records, gold)) / n  # vote over first k
        toks = sum(sum(c["num_tokens"] for c in r["candidates"][:k]) for r in records) / n  # avg gen tokens
        curve.append({"N": k, "Accuracy (%)": pct(acc), "Avg Gen Tokens / Problem": round(toks, 2)})
    save_sc_scaling_curve(curve, os.path.join(OUT_DIR, "sc_scaling_curve.json"))

    # ---------- report_stats.json: numbers needed for report §1, §4 ----------
    def win_loss(a: str, b: str) -> dict:  # paired comparison of two methods on the same problems [PA3 §7.1]
        wa, wb = correct_by_method[a], correct_by_method[b]
        return {"wins": sum(x and not y for x, y in zip(wa, wb)),  # a right, b wrong
                "losses": sum(y and not x for x, y in zip(wa, wb)),  # a wrong, b right
                "both_correct": sum(x and y for x, y in zip(wa, wb)),
                "both_wrong": sum((not x) and (not y) for x, y in zip(wa, wb))}

    all_shared = [c for r in records for c in r["candidates"][:N_SHARED]]  # every judged path
    all_scores = [s for c in all_shared for s in c.get("step_scores", [])]  # every step verdict
    stats = {
        "num_problems": n,
        "bootstrap_95ci_accuracy": {m: bootstrap_ci(correct_by_method[m]) for m in METHODS},  # [PA3 §7.1]
        "paired_win_loss": {  # PRM vs SC baselines [PA3 §7.1]
            f"{p} vs {b}": win_loss(p, b) for p in (M_PRM_MIN, M_PRM_PROD, M_PRM_PEN) for b in (M_SC5, M_SC15)
        },
        "greedy_format_failures": sum(r["greedy"]["answer"] is None for r in records),  # [PA3 §4.2]
        "sampled_format_failure_rate_%": pct(
            sum(c.get("answer") is None for r in records for c in r["candidates"]) / max(1, n * N_MATCHED)),
        "mean_steps_per_path_Kbar": round(sum(len(c.get("steps", [])) for c in all_shared) / max(1, len(all_shared)), 3),
        "judge_invalid_rate_%": pct(sum(s < 0 for s in all_scores) / max(1, len(all_scores))),
        "token_budget_10x_greedy": 10 * greedy_tok,  # report §4 definition [PA3 §7.4]
        "tokens_spent": {  # generated + judge tokens per method (whole benchmark)
            M_GREEDY: greedy_tok, M_SC5: sc5_tok, M_SC15: sc15_tok, "PRM Best-of-5 (any)": sc5_tok + judge_tok,
        },
        "avg_tokens_per_call": {  # generator vs judge call asymmetry [PA3 §7.4]
            "generation": round(sc15_tok / max(1, N_MATCHED * n), 2),
            "judge_output": round(judge_tok / max(1, judge_calls), 2),
        },
    }
    with open(os.path.join(OUT_DIR, "report_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)  # helper file for writing REPORT.pdf (not graded)

    write_judge_audit(records)  # spreadsheet for the required 30-step hand audit [PA3 §7.3]
    write_failure_candidates(records)  # shortlist of failure cases [PA3 §7.3]
    make_plots(summary, curve, n)  # two scaling plots [PA3 §7.1]
    return {"summary": summary, "ablation": ablation, "curve": curve, "stats": stats}


def write_judge_audit(records: list[dict]) -> None:
    """One row per judged step; fill the 'human_label' column by hand for >= 30 rows (>= 10 INVALID)."""
    path = os.path.join(OUT_DIR, "judge_audit.csv")  # opens in Excel / Google Sheets
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)  # CSV writer
        w.writerow(["problem_id", "candidate", "step_index", "judge_verdict", "judge_rationale", "step_text",
                    "candidate_answer", "ground_truth", "candidate_correct", "human_label(VALID/INVALID)"])
        for r in records:  # every problem
            for ci, c in enumerate(r["candidates"][:N_SHARED]):  # only the 5 judged candidates
                for si, (step, score) in enumerate(zip(c.get("steps", []), c.get("step_scores", []))):
                    w.writerow([r["id"], ci, si + 1, "VALID" if score > 0 else "INVALID",
                                c.get("rationales", [""] * (si + 1))[si], step, c.get("answer"),
                                r["ground_truth_answer"], is_correct(c.get("answer"), r["ground_truth_answer"]), ""])


def write_failure_candidates(records: list[dict]) -> None:
    """Shortlists likely judge false positives/negatives and shared flawed premises for report §3."""
    possible_fp, possible_fn, shared_flaw = [], [], []  # three failure categories [PA3 §7.3]
    for r in records:  # every problem
        g = r["ground_truth_answer"]  # gold answer
        shared = r["candidates"][:N_SHARED]  # judged paths
        for ci, c in enumerate(shared):  # every judged path
            scores = c.get("step_scores", [])  # its verdicts
            ok = is_correct(c.get("answer"), g)  # final answer right?
            if scores and all(s > 0 for s in scores) and not ok and c.get("answer") is not None:
                possible_fp.append({"id": r["id"], "candidate": ci, "answer": c.get("answer"), "gold": g})  # all VALID yet wrong
            if ok and any(s < 0 for s in scores):
                possible_fn.append({"id": r["id"], "candidate": ci,  # right answer but a step rejected
                                    "invalid_steps": [i + 1 for i, s in enumerate(scores) if s < 0]})
        if not any(is_correct(c.get("answer"), g) for c in shared):  # all 5 samples wrong
            shared_flaw.append({"id": r["id"], "answers": [c.get("answer") for c in shared], "gold": g})
    with open(os.path.join(OUT_DIR, "failure_candidates.json"), "w", encoding="utf-8") as f:
        json.dump({"possible_judge_false_positive_paths": possible_fp,
                   "possible_judge_false_negative_paths": possible_fn,
                   "all_five_wrong_shared_flaw_or_mode_collapse": shared_flaw}, f, indent=2)


def make_plots(summary: list[dict], curve: list[dict], n: int) -> None:
    """Accuracy vs total LLM calls and vs total output tokens, SC curve with PRM points overlaid [PA3 §7.1]."""
    try:
        import matplotlib  # plotting is optional; the JSON artifacts are what is graded
        matplotlib.use("Agg")  # headless backend (no display on Kaggle)
        import matplotlib.pyplot as plt
    except Exception:  # matplotlib missing -> skip plots silently
        print("[plots] matplotlib not available, skipping plots")
        return

    ink, muted, grid = "#2b2b2b", "#6b6b6b", "#e6e6e6"  # text and gridline colors
    c_sc, c_prm, c_greedy = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (blue, orange, aqua)
    rows = {r["Method"]: r for r in summary}  # method -> summary row
    prm_markers = {M_PRM_MIN: ("o", "PRM min", 110), M_PRM_PROD: ("s", "PRM product", 60), M_PRM_PEN: ("^", "PRM step penalty", 28)}  # shrinking sizes keep stacked points visible

    def total_tokens(row: dict) -> float:  # generated + judge tokens over the whole benchmark
        return (row["Avg Gen Tokens / Problem"] + row["Avg Judge Tokens / Problem"]) * n

    panels = [  # (file name, x label, SC x-values, function giving a method's x)
        ("scaling_accuracy_vs_calls.png", f"Total LLM calls ({n} problems)",
         [p["N"] * n for p in curve], lambda row: row["Total LLM Calls"]),
        ("scaling_accuracy_vs_tokens.png", f"Total output tokens: generated + judge ({n} problems)",
         [p["Avg Gen Tokens / Problem"] * n for p in curve], total_tokens),
    ]
    for fname, xlabel, sc_x, x_of in panels:  # one figure per x-axis (never a dual axis)
        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)  # report-sized figure
        sc_y = [p["Accuracy (%)"] for p in curve]  # SC accuracy for N = 1..15
        ax.plot(sc_x, sc_y, color=c_sc, linewidth=2, marker="o", markersize=4, label="Self-consistency, N = 1..15")
        for k in (1, 5, 15):  # selective direct labels, not every point
            ax.annotate(f"N={k}", (sc_x[k - 1], sc_y[k - 1]), textcoords="offset points", xytext=(4, -12),
                        fontsize=7, color=muted)
        g = rows[M_GREEDY]  # greedy point
        ax.scatter([x_of(g)], [g["Accuracy (%)"]], color=c_greedy, s=64, zorder=3,
                   edgecolors="white", linewidths=1.5, label="Greedy")
        for m, (marker, label, size) in prm_markers.items():  # the three PRM variants share calls/tokens
            row = rows[m]
            ax.scatter([x_of(row)], [row["Accuracy (%)"]], color=c_prm, marker=marker, s=size, zorder=4,
                       edgecolors="white", linewidths=1.5, label=label)
        ax.set_xlabel(xlabel, color=ink, fontsize=9)  # axis titles in text ink
        ax.set_ylabel("Accuracy (%)", color=ink, fontsize=9)
        ax.grid(True, color=grid, linewidth=0.8)  # recessive grid
        ax.set_axisbelow(True)  # grid behind the marks
        for side in ("top", "right"):  # remove chart junk
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)  # muted tick labels
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, labelcolor=ink)  # legend below the axes, never over data
        fig.tight_layout()  # no clipped labels
        fig.savefig(os.path.join(OUT_DIR, fname))  # PNG for REPORT.pdf
        plt.close(fig)  # free memory
    print("[plots] wrote outputs/scaling_accuracy_vs_calls.png and outputs/scaling_accuracy_vs_tokens.png")


def write_run_info(total_seconds: float) -> None:
    """Hardware, library versions and runtime for report §5 [PA3 §7.5]."""
    info = {"python": platform.python_version(), "seed_protocol": f"{GLOBAL_SEED} + problem_index",
            "total_runtime_seconds": round(total_seconds, 1)}
    try:
        import torch, transformers  # versions of the core stack
        info.update({"torch": torch.__version__, "transformers": transformers.__version__,
                     "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
        import bitsandbytes  # 4-bit NF4 quantization library
        info["bitsandbytes"] = bitsandbytes.__version__
    except Exception as e:  # never fail the run over metadata
        info["version_error"] = str(e)
    with open(os.path.join(OUT_DIR, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="CS5326 PA3 pipeline")
    parser.add_argument("--limit", type=int, default=None, help="only the first K problems")
    parser.add_argument("--check-format", action="store_true", help="greedy only: report Final Answer compliance")
    parser.add_argument("--artifacts-only", action="store_true", help="rebuild artifacts from per_problem.jsonl")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)  # outputs/ must exist
    problems = load_problems(DATA_PATH)[: args.limit]  # all 50 (or the first K)

    if args.artifacts_only:  # no GPU needed: just recompute metrics
        records = list(load_done_records(PER_PROBLEM_PATH).values())
        build_artifacts(records)
        print(json.dumps(json.load(open(os.path.join(OUT_DIR, "results_summary.json"))), indent=2))
        return

    import torch  # only needed for real runs
    if not torch.cuda.is_available():  # this assignment must run on GPU [PA3 §3.4]
        raise SystemExit("No CUDA GPU found. Set Kaggle Accelerator to 'GPU T4 x2'.")
    print(f"GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    t_start = time.perf_counter()  # total runtime for the report
    # Generator (1.5B fp16) on cuda:0, judge (7B NF4) on cuda:1 - never device_map="auto" [PA3 §3.4].
    gen_model, gen_tok, judge_model, judge_tok = load_models(generator_device="cuda:0", judge_device="cuda:1")
    gen_fn = make_hf_generate_fn(gen_model, gen_tok)  # generate_fn(prompt, n, kwargs)
    judge_fn = make_hf_judge_fn(judge_model, judge_tok)  # judge_fn(prompt)
    print(f"Models loaded in {time.perf_counter() - t_start:.0f}s")

    if args.check_format:  # check delimiter compliance before the long run [PA3 §4.2 warning]
        failures = 0  # completions without a parseable Final Answer
        for i, p in enumerate(problems):
            set_seed_for_problem(i)
            g = run_greedy(question=p["question"], generate_fn=gen_fn)
            ok = g["answer"] is not None
            failures += int(not ok)
            print(f"[{i + 1}/{len(problems)}] steps={len(g['steps'])} answer={g['answer']} "
                  f"gold={p['ground_truth_answer']} {'OK' if ok else 'FORMAT FAIL'}")
            if not ok:
                print("   tail of completion:", repr(g["completion"][-200:]))  # see what went wrong
        print(f"Format compliance: {len(problems) - failures}/{len(problems)}")
        return

    done = load_done_records(PER_PROBLEM_PATH)  # resume support
    print(f"{len(done)} problems already finished; resuming")
    for i, problem in enumerate(problems):  # main benchmark loop
        if problem["id"] in done:  # finished in an earlier session
            continue
        t0 = time.perf_counter()
        record = solve_problem(i, problem, gen_fn, judge_fn)  # all six configurations
        append_record(PER_PROBLEM_PATH, record)  # checkpoint immediately
        done[problem["id"]] = record
        p = record["predictions"]
        print(f"[{i + 1}/{len(problems)}] {problem['id']} gold={problem['ground_truth_answer']} "
              f"greedy={p['greedy']} sc5={p['sc5']} sc15={p['sc15']} prm_min={p['prm_min']} "
              f"judge_calls={record['judge_calls']} ({time.perf_counter() - t0:.0f}s)", flush=True)

    records = [done[p["id"]] for p in problems if p["id"] in done]  # only the requested problems
    build_artifacts(records)  # write every JSON artifact + plots
    write_run_info(time.perf_counter() - t_start)  # reproducibility info
    with open(os.path.join(OUT_DIR, "results_summary.json"), encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=2))  # show the final table in the log
    print(f"Done in {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
