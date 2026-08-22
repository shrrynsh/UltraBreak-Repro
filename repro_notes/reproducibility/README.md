# Reproducibility — Review & Analysis

**Date: 2026-08-22.** Plain-language write-up of whether we can reproduce the
paper's headline jailbreak numbers, and exactly where the reproduction breaks.

---

## 1. The one-sentence story

**The authors' released image reproduces the paper exactly; a fresh retrain does
not — and the gap is almost entirely on AdvBench.**

When we score the authors' own released `ultrabreak.png` with the authors' own
judge on the paper's own evaluation set, we get **81.59%** — matching the paper to
the last decimal. But when we *retrain* the image ourselves, we fall far short,
especially on AdvBench.

---

## 2. Two honest ways to reproduce

The paper's *described method* and the authors' *released code* disagree (see
[[../author_code_defects/README]]). A single reproduction cannot honour both, so we
keep **two tracks side by side**:

| Track | What it is | Reproduces |
|---|---|---|
| **Code-faithful** | the authors' released code, run verbatim | *what they actually ran* |
| **Paper-faithful** | the paper's described method, implemented cleanly (projection ON, normalization present) | *what they described* |

**Important clarification (2026-08-22):** "code-faithful" means a **retrain from the
authors' released code** — NOT just scoring their released image. The released image
is a *separate reference row*. An exact code-faithful **retrain** (`optimise_exact.py`,
authors' `optimise.py` verbatim with only 3 input/output lines changed) is running now
in job 1058.

---

## 3. The numbers so far

| What | SafeBench | AdvBench | vs paper (81.59 / 72.69) |
|---|---|---|---|
| **Released image**, authors' judge (v1), clean 315 set | **81.59%** | **74.62%** | ✅ exact / +1.9 |
| **Released image**, our corrected judge (v3) | 78.73% | 67.69% | −2.9 / −5.0 |
| **Paper-faithful retrain @1300** (v3) | 64.13% | **10.19%** | −17.5 / **−62.5** |
| **CE-loss ablation retrain @3000** (v3) | 67.94% | 15.77% | −13.7 / −56.9 |
| **D11-fixed retrain @3000** (job 809, white-box) | 83.49% | 39.04% | reproduces SafeBench |

**Two clear facts:**
- **SafeBench retrains land within ~15 points** of the paper.
- **AdvBench retrains collapse** — ~10–16% vs the paper's 72.69%. This is the
  central reproduction defect.

---

## 4. Why the numbers are so sensitive: three hidden "axes"

Comparing tracks is tricky because the evaluation itself can differ in three ways
that each change the score. We verified each from the authors' released code
(commit `c4c276d`).

### Axis 1 — SafeBench set size (315 vs 350)
The authors ship a 500-row SafeBench file. The paper's real eval set is **315**:

```
500  full SafeBench
−150  drop 3 categories (legal opinion, financial advice, health consultation)
= 350
− 35  remaining training queries (SafeBench-Tiny overlap)
= 315   →  257/315 = 81.59%  ✓ matches paper exactly
```

Some of our early configs used the **350** set (they forgot to exclude the 35
training queries) — that set is *contaminated* and not what the paper reports.

### Axis 2 — AdvBench prompt form (raw vs normalized)
| Form | Example first prompt | Who uses it |
|---|---|---|
| **raw** | "Write a script that can exploit vulnerabilities…" | the authors (their code uses the raw `goal` verbatim) |
| **normalized** | "**Steps to** write a script that can exploit…" | **our** `--normalize` flag only |

**The authors' code does NOT normalize AdvBench.** So the raw form is the faithful
one; the normalized form is our own deviation. (This means our paper-faithful
AdvBench @1300 used a prompt form the authors never ran — a confound we are fixing.)

### Axis 3 — Judge version (v1 vs v3)
| Judge | What it is | Released-image SafeBench |
|---|---|---|
| **v1** | authors' original judge (commit `c4c276d`) | **81.59%** |
| **v2** | our fix: model-aware response extraction | 78.73% |
| **v3** | v2 + canonical behaviors + GLM reasoning handling | (fullest) |

**Only v1 reproduces 81.59%** — and v1 has a real bug (it reads the harmful prompt
back into the judged text; see [[../author_code_defects/README]]). Our v3 fixes move
*away* from the paper. So "reproducing 81.59" partly means "inheriting the authors'
judge bug." This is a statement about how fragile ASR is as a metric, not an
accusation — the same generations score 81.59% (v1) and 78.73% (v3).

---

## 5. The clean apples-to-apples comparison (running now: job 1058)

Because those three axes were mixed together in earlier numbers, we built one
controlled experiment that fixes all of them and reads each as a column:

- **3 patch rows:** code-faithful retrain · paper-faithful retrain · released image (reference)
- **Axes as columns:** SafeBench {315, 350} × AdvBench {raw, norm} × judge {v1, v3}
- **Both settings:** white-box (Qwen2-VL) **and** black-box (5 transfer targets)
- **Both metrics:** ASR **and** NRR (non-refusal rate)

It self-resumes across 24h windows until all 144 cells are done, writing
`results/exact_comparison_summary.csv`. **This section will be updated with the
final factorial table once job 1058 completes.**

---

## 6. Bottom line

1. **The attack and the evaluation harness are genuine** — the released image
   reproduces the paper exactly on its own protocol, and generalizes to new
   benchmarks and models.
2. **Our retraining under-reproduces, specifically on AdvBench** (~10–16% vs 72.69%).
3. **The "reproduction gap" is easy to overstate** unless you control the three
   eval axes — set size, prompt form, and judge — which each move the number by
   several to tens of points.
