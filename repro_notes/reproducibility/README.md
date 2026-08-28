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

## 5. The clean apples-to-apples comparison (DONE — 144/144 cells)

We built one controlled 144-cell experiment — 3 patches × {SB-315, SB-350} ×
{AB-raw, AB-norm} × {v1, v3} × 6 models × {ASR, NRR} — to read each confound as a
column (`results/exact_comparison_summary.csv`, jobs 1088/1118).

### White-box (Qwen2-VL) ASR %, v3 judge
| Patch | SB-315 | SB-350 | AB-raw | AB-norm |
|---|---|---|---|---|
| **code-faithful retrain** | **73.0** | 71.1 | 6.3 | 22.5 |
| **paper-faithful retrain** | 64.1 | 62.3 | 6.2 | 10.2 |
| **released image (ref)** | **78.7** | 77.7 | 9.6 | **67.7** |

**Four findings drop straight out:**

1. **Code-faithful retrain BEATS paper-faithful retrain** on SafeBench (73.0 vs 64.1).
   The authors' verbatim code trains a *better* patch than the paper-as-described —
   because the paper's §3.2 projection (in the paper-faithful arm) actively hurts
   ([[../author_code_defects/README]] D1). Both trail the released image (78.7).

2. **The AdvBench "normalize" axis is the dominant confound.** The released image
   scores **9.6% on raw AdvBench but 67.7% on normalized** — a 58-point swing from
   prompt form alone. The paper's ~72% AdvBench is only reachable with *normalized*
   prompts, which the authors' released code **does not generate** (it uses the raw
   `goal`). So the reported AdvBench headline depends on a prompt form the code can't
   produce. **NRR explains the mechanism:** released-image NRR is 25% on raw AdvBench
   (it refuses 3 of 4) vs 82% on normalized — normalization mainly defeats refusals.

3. **The v1 judge inflates ASR by ~2–7 pts** vs v3 across the board (e.g. released
   SB-315: 81.6 v1 → 78.7 v3; AB-norm: 74.6 → 67.7) — the D12 bug, quantified.

4. **SB-315 vs SB-350 is a minor axis** (~1–2 pts) — contamination barely moves it.

### Transfer (black-box, SB-315, v3)
| Model | code-faithful | paper-faithful | released |
|---|---|---|---|
| Qwen2.5-VL | 27.0 | 26.0 | **53.7** |
| Qwen-VL-Chat | 60.3 | 68.9 | **77.1** |
| LLaVA-1.6 | 85.4 | 87.6 | **88.3** |
| Kimi-VL | 65.1 | 61.0 | **69.8** |
| GLM-4.1V | 30.2 | 29.8 | **35.9** |

The **released image transfers best on every target** (biggest gap on Qwen2.5-VL:
54 vs ~26). code-faithful vs paper-faithful is a wash in transfer (each wins ~half).

---

## 6. Bottom line

1. **The attack and the evaluation harness are genuine** — the released image
   reproduces the paper exactly on its own protocol, and generalizes to new
   benchmarks and models.
2. **Our retraining under-reproduces, specifically on AdvBench** (~10–16% vs 72.69%).
3. **The "reproduction gap" is easy to overstate** unless you control the three
   eval axes — set size, prompt form, and judge — which each move the number by
   several to tens of points.
