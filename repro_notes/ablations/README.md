# Ablations (Table 3 + Table 6) — Review & Analysis

**Date: 2026-08-23.** Plain-language write-up of the component ablations and the
TV-weight sweep (pipeline steps 9–13, jobs 1005/1020/1041/1066/1074). Each run is
the **809 recipe** (D11 fix, no projection, 3000 steps, seed 0, semantic-attention
loss τ=0.5, transforms on, TV=0.5) with **one knob flipped**, so every number is a
single-variable change against the reproduced baseline. All ASR% is white-box
Qwen2-VL-7B, v3 judge.

---

> ## UPDATE 2026-10-21 — everything below is superseded for *reproduction* purposes
>
> Job 9187 retrained all six variants at the paper's **1300 iterations** and re-evaluated on
> all five models. **The recipe is otherwise identical to the 809 recipe used below** — same D11
> normalisation fix, no projection, lr 0.01, seed 0, tau 0.5, transforms on, same full-50 corpus.
> Only the budget differs, so the 1300-vs-3000 comparison is clean.
>
> The paper-table verdicts now live in [`../TABLES.md`](../TABLES.md#table-3) and
> [`#table-6`](../TABLES.md#table-6). **This page is kept as the record of our own 3001-epoch
> experiment**, which is still the configuration that reproduces the headline (job 809, 83.49%).
>
> ### Baseline audit (2026-10-22) — and why THIS page's §1 was right all along
>
> Checked from each run's own `outputs/<exp>/run_config.json`. The five 3001-epoch ablations below
> differ from **job 809** by exactly one field each, so §1's deltas are properly single-variable.
> The 1300-iteration retrains differ from **job 773** by exactly one field each — *not* from
> `repro_paper_faithful_1300`, which differs in three structural settings (projection on, no latent
> clamp, image init). The first write-up of the 1300 table used that wrong baseline; corrected in
> [`../TABLES.md`](../TABLES.md#table-3).
>
> ### The one result that only the pair of budgets reveals
>
> Component contribution = baseline − ablated, white-box Qwen2-VL SafeBench, judge v3.
> Positive means the component helps. Both columns use their own matched control.
>
> | component removed | **@1300** (vs job 773's 23.49) | **@3000** (vs job 809's 83.49) |
> |---|---|---|
> | attention weighting | **+4.44** | **+68.89** |
> | input constraints | **−15.24** | **+51.74** |
> | **semantic loss (CE)** | **−66.03 — it HURTS** | **+15.55** |
>
> **At 3000 iterations every component contributes and the paper's ablation claim reproduces. At the
> paper's stated 1300 iterations none of them does.** The semantic-attention loss is the extreme
> case: replacing it with plain cross-entropy moves SafeBench from 23.49% to **89.52%** and AdvBench
> from 1.35% to **80.00%**, and `a_ce_1300` is the best white-box SafeBench patch in the study.
>
> | budget | full method (semantic) | CE variant | winner |
> |---|---|---|---|
> | **1300 — the paper's** | 23.49 | **89.52** | **CE, by 66 points** |
> | 3000 | **83.49** | 67.94 | semantic, by 15.6 |
>
> Nothing but the budget differs across those four runs. It is the same shape as the D11
> budget-gating finding, so two separate parts of the method need a budget the paper does not state
> before they help at all — one story, not two anomalies.
>
> Attention weighting and the constraints keep their sign and ranking **at 3000**; at 1300 they are
> within noise. So the component ranking attention > constraints > semantic holds only at the larger
> budget.

---

## 1. Table 3 — remove one component at a time

| Configuration | SafeBench | AdvBench | Δ SafeBench vs 809 |
|---|---|---|---|
| **809 — full method** | **83.49%** | **39.04%** | — |
| w/o semantic loss → CE (B13, job 1005) | 67.94% | 15.77% | −15.5 |
| w/o attention weighting → token, τ→0 (B14, job 1020) | **14.60%** | 1.15% | **−68.9** |
| w/o constraints — no transforms, TV=0 (B12, job 1041) | 31.75% | 2.12% | −51.7 |

**Reading it simply — every piece matters, and here is the ranking:**
1. **Attention weighting is the single most important component.** Turning it off
   (token mode) collapses the attack from 83% to **14.6%** — the biggest drop in the
   whole study. The relaxed, attention-weighted target is what makes the attack work.
2. **Input constraints (random transforms + TV) are next** — removing them more than
   halves the score (83 → 32).
3. **Semantic loss vs plain cross-entropy** is a real but smaller effect (83 → 68).

So the component importance order is: **attention-weighting ≫ constraints > semantic-vs-CE.**
This cleanly reproduces the paper's "each component contributes" claim and, beyond
that, *quantifies* which contributes most.

---

## 2. Table 6 — TV-weight sweep (transforms ON, vary only TV)

| TV weight | SafeBench | AdvBench |
|---|---|---|
| 0.2 (B16a, job 1066) | 40.00% | 1.54% |
| **0.5 (= 809, peak)** | **83.49%** | **39.04%** |
| 1.0 (B16b, job 1074) | 56.51% | 30.38% |

> **Scope correction (2026-10-21).** This sweep measures the **white-box surrogate**; the paper's
> Table 6 measures **transfer to Qwen-VL-Chat and GLM**. As a *Table 6 verdict* the comparison below
> was withdrawn. Job 9187 then ran the sweep properly — 1300 iterations, on the paper's own two
> victims — and found the optimum at **TV=0.5 on both**, so the conclusion survived the correction;
> see [`../TABLES.md#table-6`](../TABLES.md#table-6). What follows remains valid as a statement about
> the surrogate.

**The sweep is complete, and on the surrogate it puts the optimum away from the paper's 0.2.** The curve
**rises 0.2 → 0.5 then falls 0.5 → 1.0**, so there is a clear **interior optimum at
TV=0.5**, not at 0.2 as the paper claims:

```
SafeBench ASR vs TV:   0.2 → 40.0   |   0.5 → 83.5 (peak)   |   1.0 → 56.5
```

This is the direct test of [[DISCREPANCIES#D10]]. On the surrogate the optimum sits at
**0.5** (our default), not 0.2 — and the properly-matched sweep on the paper's own victims
agrees (68.89% and 60.95% at TV=0.5, both peaks). AdvBench tells the same story (peak 0.5: 39.0%, vs 1.5%
at 0.2 and 30.4% at 1.0). Both benchmarks agree: **0.5 > 1.0 > 0.2.**

(Note: B12's "no constraints" row is TV=0 **and** transforms off, so it is *not* a
clean TV=0 point for this sweep — it drops both knobs at once.)

---

## 2b. Table 5 — affirming-phrase sensitivity (B15, job 1158)

Authors' released image on GLM-4.1V / SafeBench-315, v3, with three affirming
phrases:

| Phrase | Ours (GLM) | Paper (GLM) |
|---|---|---|
| `[Jailbroken Mode]` | 60.63% | 66.03% |
| `[ACCESS AUTHORISED]` | 54.60% | 72.06% |
| `[CLASSIFIED INFO]` | 59.68% | 66.98% |

**Largely reproduces, after the D14 fix.** These numbers were re-measured with GLM
generating at `max_new_tokens=4096` (see [[../author_code_defects/README]] D14); at the
old 512 budget GLM was truncated mid-`<think>` on a large fraction of rows and the three
phrases read 39.05 / 30.48 / 32.70 — roughly 30 points low. Corrected, they land
**within 5.4–17.5 points of the paper** and all three sit in a tight 54.6–60.6 band,
which is the qualitative claim Table 5 makes: the attack is only mildly sensitive to the
exact affirming phrase.

The *ordering* still differs (paper: ACCESS > CLASSIFIED > JBM; ours: JBM > CLASSIFIED >
ACCESS), and the residual gap is largest on `[ACCESS AUTHORISED]` (54.6 vs 72.1). Judge
choice does not explain it — under v1 the same rows read 64.4 / 55.6 / 61.3, preserving
our ordering. (On the white-box Qwen2-VL the phrases swing far harder —
78.7 / 58.7 / 7.9 — so phrase choice matters much more there than on GLM.)

---

## 3. Cross-cutting: the AdvBench collapse is everywhere

Every ablation lands at **1–16% AdvBench**; only the full 809 reaches **39%**.

| Run | AdvBench |
|---|---|
| 809 full | 39.04% |
| CE | 15.77% |
| token | 1.15% |
| no-constraints | 2.12% |
| TV=0.2 | 1.54% |
| TV=1.0 | 30.38% |

So the AdvBench under-reproduction flagged throughout this study is **not specific to
any one ablation** — it is a general property of retraining on SafeBench-Tiny. The
components lift SafeBench a lot, but AdvBench stays weak unless *everything* is on and
trained to the full 3000 steps. This reinforces the reproducibility finding
([[../reproducibility/README]]): the gap is concentrated on AdvBench.

---

## 4. Bottom line

- **Table 3 reproduces and ranks the components:** attention-weighting is decisive,
  constraints are major, semantic-vs-CE is secondary.
- **Table 6 contradicts the paper (sweep complete):** the TV optimum is an interior
  **0.5** (83.5%), not 0.2 (40.0%); it falls again to 56.5% at 1.0. Ordering 0.5 > 1.0 > 0.2.
- **AdvBench stays weak across every ablation** — the reproduction gap lives on
  AdvBench, not in any single component.

**How it was produced:** `jobs/pipeline.sh` steps 9–13 (`train_abl`), the 809 recipe
with one flag changed per run (`--loss ce`, `--loss_mode token`, `--no_transforms
--tv_weight 0`, `--tv_weight {0.2,1.0}`). Update the TV=1.0 cell when job 1074 lands.
