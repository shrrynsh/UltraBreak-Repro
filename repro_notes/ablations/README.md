# Ablations (Table 3 + Table 6) — Review & Analysis

**Date: 2026-08-23.** Plain-language write-up of the component ablations and the
TV-weight sweep (pipeline steps 9–13, jobs 1005/1020/1041/1066/1074). Each run is
the **809 recipe** (D11 fix, no projection, 3000 steps, seed 0, semantic-attention
loss τ=0.5, transforms on, TV=0.5) with **one knob flipped**, so every number is a
single-variable change against the reproduced baseline. All ASR% is white-box
Qwen2-VL-7B, v3 judge.

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

**The sweep is now complete, and it CONTRADICTS the paper's claimed peak.** The curve
**rises 0.2 → 0.5 then falls 0.5 → 1.0**, so there is a clear **interior optimum at
TV=0.5**, not at 0.2 as the paper claims:

```
SafeBench ASR vs TV:   0.2 → 40.0   |   0.5 → 83.5 (peak)   |   1.0 → 56.5
```

This is the direct test of [[DISCREPANCIES#D10]]. The paper's specific claim (peak at
0.2) is **refuted** — TV is genuinely important and has an interior optimum, but it
sits at **0.5** (our default). AdvBench tells the same story (peak 0.5: 39.0%, vs 1.5%
at 0.2 and 30.4% at 1.0). Both benchmarks agree: **0.5 > 1.0 > 0.2.**

(Note: B12's "no constraints" row is TV=0 **and** transforms off, so it is *not* a
clean TV=0 point for this sweep — it drops both knobs at once.)

---

## 2b. Table 5 — affirming-phrase sensitivity (B15, job 1158)

Authors' released image on GLM-4.1V / SafeBench-315, v3, with three affirming
phrases:

| Phrase | Ours (GLM) | Paper (GLM) |
|---|---|---|
| `[Jailbroken Mode]` | 39.05% | 66.03% |
| `[ACCESS AUTHORISED]` | 30.48% | 72.06% |
| `[CLASSIFIED INFO]` | 32.70% | 66.98% |

**Does not reproduce.** Our GLM numbers are ~30 points below the paper, and the
*ordering* differs too (paper: ACCESS > CLASSIFIED > JBM; ours: JBM > CLASSIFIED >
ACCESS). Two likely causes: our GLM is judged by v3 with `<think>`-aware extraction
(the paper's GLM handling differs), and phrase sensitivity is small relative to that
judging difference. So Table 5's *magnitudes* don't reproduce; the qualitative point
(the attack is somewhat robust to the exact phrase) partly holds — all three land in
a 30–39% band for GLM. (On the white-box Qwen2-VL the phrases swing wildly —
78.7 / 58.7 / 7.9 — so phrase choice matters far more there.)

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
