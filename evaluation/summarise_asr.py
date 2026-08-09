"""
summarise_asr.py
----------------
Report attack success rate across scoring passes.

Each re-scoring pass writes alongside the raw generations with its own suffix:
  _harmbench.csv     v1  original scorer
  _harmbench_v2.csv  v2  model-aware response extraction
  _harmbench_v3.csv  v3  + canonical judge behaviors, + GLM reasoning handling

Usage
  python evaluation/summarise_asr.py                 # every run, all passes
  python evaluation/summarise_asr.py --pattern advbench
  python evaluation/summarise_asr.py --paper         # compare against reported ASR
"""

import argparse
import glob
import os

import pandas as pd

PASSES = [("v1", "_harmbench.csv"), ("v2", "_harmbench_v2.csv"), ("v3", "_harmbench_v3.csv")]

# ASR reported in the paper, keyed by (benchmark, model).
PAPER_ASR = {
    ("safebench", "Qwen2-VL-7B-Instruct"): 81.59,
    ("safebench", "Qwen2.5-VL-7B-Instruct"): 60.32,
    ("safebench", "GLM-4.1V-9B-Thinking"): 66.03,
    ("safebench", "llava-v1.6-mistral-7b-hf"): 88.25,
    ("advbench", "Qwen2-VL-7B-Instruct"): 72.69,
    ("advbench", "Qwen2.5-VL-7B-Instruct"): 35.77,
    ("advbench", "GLM-4.1V-9B-Thinking"): 30.00,
    ("advbench", "llava-v1.6-mistral-7b-hf"): 92.88,
}


def asr(path):
    """Return (asr_pct, n_success, n_rows, status_counts) for a scored file."""
    df = pd.read_csv(path)
    if "attack_success" not in df.columns:
        return None
    success = df["attack_success"].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
    statuses = (
        df["extraction_status"].value_counts().to_dict()
        if "extraction_status" in df.columns
        else {}
    )
    return 100 * success.mean(), int(success.sum()), len(df), statuses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="", help="Substring filter on the run path.")
    parser.add_argument("--paper", action="store_true", help="Show reported ASR and the gap.")
    args = parser.parse_args()

    runs = sorted(
        {
            p[: -len(suffix)]
            for _, suffix in PASSES
            for p in glob.glob(f"results/**/*{suffix}", recursive=True)
            if p.endswith(suffix)
        }
    )
    runs = [r for r in runs if args.pattern in r]

    header = f"{'run':<58}" + "".join(f"{name:>9}" for name, _ in PASSES)
    if args.paper:
        header += f"{'paper':>9}{'gap':>8}"
    print(header)
    print("-" * len(header))

    for run in runs:
        label = run.replace("results/", "")
        row, last, statuses = f"{label:<58}", None, {}
        for _, suffix in PASSES:
            path = run + suffix
            result = asr(path) if os.path.exists(path) else None
            if result is None:
                row += f"{'-':>9}"
            else:
                pct, _, _, statuses = result
                row += f"{pct:>8.2f}%"
                last = pct

        if args.paper:
            benchmark = "advbench" if "advbench" in label else "safebench"
            model = os.path.basename(run)
            reported = PAPER_ASR.get((benchmark, model))
            if reported is None or last is None:
                row += f"{'-':>9}{'-':>8}"
            else:
                row += f"{reported:>8.2f}%{last - reported:>+8.2f}"
        print(row)

        incomplete = {k: v for k, v in statuses.items() if k != "ok"}
        if incomplete:
            total = sum(statuses.values())
            detail = ", ".join(f"{k}={v}" for k, v in sorted(incomplete.items()))
            print(f"{'':<58}  incomplete generations: {detail} of {total}")


if __name__ == "__main__":
    main()
