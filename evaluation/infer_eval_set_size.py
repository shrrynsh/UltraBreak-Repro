"""
infer_eval_set_size.py
----------------------
Recover the evaluation-set size the paper actually used, from its reported ASRs.

An ASR printed to two decimals is a rounded fraction n/N.  For a candidate N,
every reported value must land within half a rounding step of some integer n --
so a denominator that fits all of a column's numbers at once is strong evidence
for the set size, and one that fits none rules it out.  Used to settle two
questions this reproduction hit:

  * SafeBench: is the evaluation set 315 (500 - 150 excluded topics - 35
    SafeBench-Tiny training queries) or the 350 some of our configs used?
  * AdvBench: the paper says "500 harmful instructions", but datasets/adv_bench.csv
    ships 520 rows.  Which one produced the reported numbers?

Usage
  python evaluation/infer_eval_set_size.py
"""

from summarise_asr import NO_ATTACK_ASR, PAPER_ASR

# Denominators worth testing per benchmark, with what each would mean.
CANDIDATES = {
    "safebench": [
        (315, "500 - 150 excluded topics - 35 SafeBench-Tiny training queries"),
        (350, "500 - 150 excluded topics, training queries left in"),
        (500, "full SafeBench"),
    ],
    "advbench": [
        (500, "the count stated in the paper"),
        (520, "the row count of datasets/adv_bench.csv"),
    ],
    "mm-safetybench": [
        (1680, "13 scenarios x ~130"),
        (500, "a 500-query subset"),
    ],
}

TOLERANCE = 0.005  # half of the last printed decimal place


def fits(asr_pct, n_total):
    """Is `asr_pct` expressible as a rounded n/n_total, and if so with what n?"""
    exact = asr_pct / 100 * n_total
    n = round(exact)
    if not 0 <= n <= n_total:
        return None
    if abs(100 * n / n_total - asr_pct) <= TOLERANCE:
        return n
    return None


def main():
    for benchmark, candidates in CANDIDATES.items():
        reported = {
            model: value
            for (bench, model), value in {**PAPER_ASR, **NO_ATTACK_ASR}.items()
            if bench == benchmark
        }
        # Keep UltraBreak and No-Attack separate so a model appearing in both counts twice.
        values = [v for (b, _), v in PAPER_ASR.items() if b == benchmark]
        values += [v for (b, _), v in NO_ATTACK_ASR.items() if b == benchmark]
        if not values:
            continue

        print(f"\n{benchmark}  ({len(values)} reported values)")
        print("-" * 78)
        for n_total, meaning in candidates:
            hits = [fits(v, n_total) for v in values]
            n_fit = sum(h is not None for h in hits)
            verdict = "ALL FIT" if n_fit == len(values) else f"{n_fit}/{len(values)} fit"
            # A larger N fits more easily: attainable ASRs are spaced 100/N apart, so an
            # arbitrary value falls inside the tolerance band with probability
            # 2*TOLERANCE / (100/N).  Report the odds of this many fits by chance so a
            # big denominator cannot look more convincing than it is.
            p_single = min(1.0, 2 * TOLERANCE / (100 / n_total))
            print(
                f"  N = {n_total:<5} {verdict:<16} p(by chance) = {p_single ** len(values):>8.1e}"
                f"   {meaning}"
            )

        best = max(candidates, key=lambda c: sum(fits(v, c[0]) is not None for v in values))
        n_total = best[0]
        print(f"\n  Best denominator: {n_total}")
        for (bench, model), value in sorted(PAPER_ASR.items()):
            if bench != benchmark:
                continue
            n = fits(value, n_total)
            shown = f"{n}/{n_total}" if n is not None else "no integer fit"
            print(f"    {model:<28} {value:>6.2f}%  =  {shown}")


if __name__ == "__main__":
    main()
