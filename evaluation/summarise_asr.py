"""
summarise_asr.py
----------------
Report attack success rate across scoring passes.

Each re-scoring pass writes alongside the raw generations with its own suffix:
  _harmbench.csv     v1  original scorer
  _harmbench_v2.csv  v2  model-aware response extraction
  _harmbench_v3.csv  v3  + canonical judge behaviors, + GLM reasoning handling

SafeBench evaluation set
------------------------
The paper evaluates on 315 queries: 500 total, minus the three excluded topics
(3 x 50 = 150), minus the 35 SafeBench-Tiny training queries that survive the
topic filter.  Some of this fork's earlier attack configs were built without
`--exclude-train` and so contain 350 rows -- i.e. they score the patch partly on
its own training queries.  By default this script filters every SafeBench run
down to the canonical 315 and reports how many rows it dropped; pass --raw to
see the unfiltered numbers.

Usage
  python evaluation/summarise_asr.py                 # every run, all passes
  python evaluation/summarise_asr.py --pattern advbench
  python evaluation/summarise_asr.py --paper         # compare against reported ASR
  python evaluation/summarise_asr.py --raw           # no eval-set filtering
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from create_attack_configs import load_safebench  # noqa: E402

PASSES = [("v1", "_harmbench.csv"), ("v2", "_harmbench_v2.csv"), ("v3", "_harmbench_v3.csv")]

SAFEBENCH_PATH = "datasets/safebench.csv"
SAFEBENCH_TINY_PATH = "datasets/SafeBench-Tiny.csv"

# ASR reported in the paper (Main Results table), keyed by (benchmark, model).
PAPER_ASR = {
    ("safebench", "Qwen2-VL-7B-Instruct"): 81.59,
    ("safebench", "Qwen-VL-Chat"): 72.70,
    ("safebench", "Qwen2.5-VL-7B-Instruct"): 60.32,
    ("safebench", "llava-v1.6-mistral-7b-hf"): 88.25,
    ("safebench", "Kimi-VL-A3B-Instruct"): 67.94,
    ("safebench", "GLM-4.1V-9B-Thinking"): 66.03,
    ("advbench", "Qwen2-VL-7B-Instruct"): 72.69,
    ("advbench", "Qwen-VL-Chat"): 71.92,
    ("advbench", "Qwen2.5-VL-7B-Instruct"): 35.77,
    ("advbench", "llava-v1.6-mistral-7b-hf"): 92.88,
    ("advbench", "Kimi-VL-A3B-Instruct"): 30.38,
    ("advbench", "GLM-4.1V-9B-Thinking"): 30.00,
    ("mm-safetybench", "Qwen2-VL-7B-Instruct"): 57.26,
    ("mm-safetybench", "Qwen-VL-Chat"): 53.10,
    ("mm-safetybench", "Qwen2.5-VL-7B-Instruct"): 45.83,
    ("mm-safetybench", "llava-v1.6-mistral-7b-hf"): 71.90,
    ("mm-safetybench", "Kimi-VL-A3B-Instruct"): 54.58,
    ("mm-safetybench", "GLM-4.1V-9B-Thinking"): 67.08,
}

# The paper's "No Attack" column -- the reference every ASR should be read against.
NO_ATTACK_ASR = {
    ("safebench", "Qwen2-VL-7B-Instruct"): 18.41,
    ("safebench", "Qwen-VL-Chat"): 22.86,
    ("safebench", "Qwen2.5-VL-7B-Instruct"): 14.29,
    ("safebench", "llava-v1.6-mistral-7b-hf"): 80.32,
    ("safebench", "Kimi-VL-A3B-Instruct"): 39.37,
    ("safebench", "GLM-4.1V-9B-Thinking"): 46.03,
    ("advbench", "Qwen2-VL-7B-Instruct"): 0.38,
    ("advbench", "Qwen-VL-Chat"): 1.92,
    ("advbench", "Qwen2.5-VL-7B-Instruct"): 0.00,
    ("advbench", "llava-v1.6-mistral-7b-hf"): 21.35,
    ("advbench", "Kimi-VL-A3B-Instruct"): 4.42,
    ("advbench", "GLM-4.1V-9B-Thinking"): 2.12,
    ("mm-safetybench", "Qwen2-VL-7B-Instruct"): 26.19,
    ("mm-safetybench", "Qwen-VL-Chat"): 21.49,
    ("mm-safetybench", "Qwen2.5-VL-7B-Instruct"): 33.45,
    ("mm-safetybench", "llava-v1.6-mistral-7b-hf"): 35.06,
    ("mm-safetybench", "Kimi-VL-A3B-Instruct"): 41.79,
    ("mm-safetybench", "GLM-4.1V-9B-Thinking"): 43.69,
}

_CLEAN_TARGETS_CACHE = {}


def benchmark_of(label):
    """Infer which benchmark a run path belongs to."""
    lowered = label.lower()
    if "advbench" in lowered:
        return "advbench"
    if "mm_safety" in lowered or "mm-safety" in lowered or "mmsafety" in lowered:
        return "mm-safetybench"
    return "safebench"


def clean_safebench_targets():
    """
    The canonical 315-query SafeBench evaluation set, as `target` strings.

    Derived from the source datasets rather than a generated attack config, so it
    cannot drift with whatever flags a config happened to be built with.  Mirrors
    build_attack_config()'s `target` column: the clean instruction, minus a
    trailing period.
    """
    if "safebench" not in _CLEAN_TARGETS_CACHE:
        df = load_safebench(
            SAFEBENCH_PATH,
            categories=None,
            exclude_train_path=SAFEBENCH_TINY_PATH,
            exclude_categories=True,
        )
        _CLEAN_TARGETS_CACHE["safebench"] = {
            t.strip().rstrip(".").lower() for t in df["clean_target"]
        }
    return _CLEAN_TARGETS_CACHE["safebench"]


def asr(path, benchmark, clean=True):
    """Return (asr_pct, n_success, n_rows, n_dropped, status_counts) for a scored file."""
    df = pd.read_csv(path)
    if "attack_success" not in df.columns:
        return None

    n_dropped = 0
    if clean and benchmark == "safebench" and "target" in df.columns:
        keep = df["target"].astype(str).str.strip().str.rstrip(".").str.lower().isin(
            clean_safebench_targets()
        )
        n_dropped = int((~keep).sum())
        df = df[keep]

    if len(df) == 0:
        return None
    success = df["attack_success"].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
    statuses = (
        df["extraction_status"].value_counts().to_dict()
        if "extraction_status" in df.columns
        else {}
    )
    return 100 * success.mean(), int(success.sum()), len(df), n_dropped, statuses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="", help="Substring filter on the run path.")
    parser.add_argument("--paper", action="store_true", help="Show reported ASR and the gap.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Do not filter SafeBench runs to the canonical 315-query evaluation set.",
    )
    parser.add_argument(
        "--gain",
        action="store_true",
        help="With --paper, show the paper's No-Attack reference instead of its UltraBreak ASR.",
    )
    args = parser.parse_args()
    clean = not args.raw

    runs = sorted(
        {
            p[: -len(suffix)]
            for _, suffix in PASSES
            for p in glob.glob(f"results/**/*{suffix}", recursive=True)
            if p.endswith(suffix)
        }
    )
    runs = [r for r in runs if args.pattern in r]

    header = f"{'run':<58}" + f"{'n':>6}" + "".join(f"{name:>9}" for name, _ in PASSES)
    if args.paper:
        header += f"{'no-atk':>9}" if args.gain else f"{'paper':>9}"
        header += f"{'gap':>8}"
    print(header)
    print("-" * len(header))

    total_dropped = 0
    for run in runs:
        label = run.replace("results/", "")
        benchmark = benchmark_of(label)
        row, last, statuses, n_rows, dropped = f"{label:<58}", None, {}, None, 0
        cells = ""
        for _, suffix in PASSES:
            path = run + suffix
            result = asr(path, benchmark, clean=clean) if os.path.exists(path) else None
            if result is None:
                cells += f"{'-':>9}"
            else:
                pct, _, n_rows, dropped, statuses = result
                cells += f"{pct:>8.2f}%"
                last = pct
        row += f"{n_rows if n_rows is not None else '-':>6}" + cells
        total_dropped += dropped

        if args.paper:
            model = os.path.basename(run)
            table = NO_ATTACK_ASR if args.gain else PAPER_ASR
            reference = table.get((benchmark, model))
            if reference is None or last is None:
                row += f"{'-':>9}{'-':>8}"
            else:
                row += f"{reference:>8.2f}%{last - reference:>+8.2f}"
        print(row)

        if dropped:
            print(
                f"{'':<58}  dropped {dropped} SafeBench-Tiny training queries "
                f"({dropped + n_rows} -> {n_rows})"
            )
        incomplete = {k: v for k, v in statuses.items() if k != "ok"}
        if incomplete:
            total = sum(statuses.values())
            detail = ", ".join(f"{k}={v}" for k, v in sorted(incomplete.items()))
            print(f"{'':<58}  incomplete generations: {detail} of {total}")

    if clean and total_dropped:
        print(
            f"\nFiltered to the canonical {len(clean_safebench_targets())}-query SafeBench "
            f"evaluation set; {total_dropped} contaminated rows dropped in total. "
            f"Use --raw to see unfiltered numbers."
        )


if __name__ == "__main__":
    main()
