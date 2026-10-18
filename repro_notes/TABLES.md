# Every table in the paper, next to ours

**As of 2026-10-18.** One section per numbered table in UltraBreak (arXiv 2602.01025).
Paper values extracted from `arxiv.org/html/2602.01025v1` and recorded in
[`paper_specs.md`](paper_specs.md); ours come from this repo's `results/`.

**Reading the numbers.** Ours are **judge v3** (our corrected scorer) unless a column says
v1. The paper's scorer is closest to our **v1** — where the two differ we say so, because
the gap is not constant (+1 to +13 points depending on the run; see
[`DISCREPANCIES.md`](DISCREPANCIES.md) D12). Our AdvBench is the **normalised** "Steps to …"
form, which is UltraBreak's own protocol (D13) — the raw-imperative form scores ~58 points
lower and is reported separately in §1b. Our MM-SafetyBench is a seed-0 stratified **520-row
subsample** of the full 1680 (binomial SE ≈ 2.2 pts).

| Paper table | What it is | Status |
|---|---|---|
| [1](#table-1) | Main results, 5 methods × 6 models × 3 benchmarks | ✅ reproduced |
| [2](#table-2) | StrongREJECT scores | ⬜ blocked — native 0–1 scorer unavailable |
| [3](#table-3) | Component ablation | 🟡 **26/50 cells, and the ablations are at the wrong budget** — retraining at 1300 iters (job 9134) |
| [4](#table-4) | Hyperparameters | ✅ matched |
| [5](#table-5) | Affirming-phrase sensitivity | ✅ reproduced |
| [6](#table-6) | TV-weight sweep | 🟡 **2/8 cells, same budget problem** — TV=0.5 matches; retraining at 1300 iters (job 9134) |
| 7 | Qualitative attack examples | n/a — no numbers, harmful content not reproduced |
| 8 | *does not exist* | — |
| [9](#table-9) | Frontier commercial models | 🚫 paid API |
| [10](#table-10) | Compute overhead | ✅ reproduced |

---

## Table 1 {#table-1}

**Paper caption:** ASR (%) across target models and benchmarks. Qwen2-VL-7B is the
white-box surrogate; the other five are black-box.

### 1a. SafeBench

| Method | Qwen2-VL* | Qwen-VL-Chat | Qwen2.5-VL | LLaVA-1.6 | Kimi-VL | GLM-4.1V |
|---|---|---|---|---|---|---|
| No Attack — **paper** | 18.41 | 22.86 | 14.29 | 80.32 | 39.37 | 46.03 |
| No Attack — **ours** | 13.02 | 7.30 | 18.10 | 57.14 | 33.02 | 53.02 |
| FigStep — **paper** | 44.76 | 69.52 | 53.97 | 47.94 | 73.02 | 88.25 |
| FigStep — **ours** | 68.25 | 70.16 | 63.17 | 55.56 | 77.78 | 83.49 |
| VAJM — **paper** | 0.95 | 12.06 | 28.89 | 57.46 | 41.27 | 67.62 |
| VAJM — **ours** | 12.06 | 6.35 | 22.54 | 59.68 | 33.65 | 50.79 |
| UMK — **paper** | **97.78** | 0.63 | 15.24 | 20.63 | 12.70 | 50.79 |
| UMK — **ours** | **1.27** | 3.17 | 20.00 | 61.27 | 32.06 | 60.95 |
| UltraBreak — **paper** | 81.59 | 72.70 | 60.32 | 88.25 | 67.94 | 66.03 |
| UltraBreak — **ours, released image** | **78.73** (v1 **81.59**) | 77.14 | 53.65 | **88.25** | 69.84 | 60.32 |
| UltraBreak — ours, code-faithful retrain | 73.02 | 60.32 | 26.98 | 85.40 | 65.08 | 58.73 |
| UltraBreak — ours, paper-faithful retrain | 64.13 | 68.89 | 26.03 | 87.62 | 60.95 | 60.95 |

**The released artifact reproduces exactly** — 81.59 under the authors' own judge, to the
decimal, and LLaVA 88.25 likewise. Our *retrains* reach the easy targets but miss Qwen2.5-VL
badly (27 vs 60); that gap is the study's main open scientific question.

### 1b. AdvBench

| Method | Qwen2-VL* | Qwen-VL-Chat | Qwen2.5-VL | LLaVA-1.6 | Kimi-VL | GLM-4.1V |
|---|---|---|---|---|---|---|
| No Attack — **paper** | 0.38 | 1.92 | 0.00 | 21.35 | 4.42 | 2.12 |
| No Attack — **ours** | 1.35 | 3.46 | 0.77 | 48.65 | 3.46 | 9.81 |
| VAJM — **paper** | 0.38 | 0.96 | 0.38 | 19.42 | 3.65 | 3.65 |
| VAJM — **ours** | 0.19 | 1.15 | 0.96 | 21.15 | 3.46 | 3.08 |
| UMK — **paper** | **70.00** | 0.38 | 2.69 | 16.35 | 2.12 | 4.42 |
| UMK — **ours** | **0.58** | 0.96 | 1.73 | 27.69 | 4.42 | 5.00 |
| UltraBreak — **paper** | 72.69 | 71.92 | 35.77 | 92.88 | 30.38 | 30.00 |
| UltraBreak — **ours, released image** | 67.69 (v1 **74.62**) | 75.00 | 25.19 | 93.85 | 28.85 | 24.04 |
| UltraBreak — ours, code-faithful | 22.50 | 38.85 | 2.12 | 89.62 | 27.50 | 21.73 |
| UltraBreak — ours, paper-faithful | 10.19 | 75.00 | 1.73 | 91.15 | 29.62 | 19.81 |

**VAJM reproduces almost cell for cell here.** Note the whole table presumes the
normalised prompt form: on the **raw** AdvBench imperatives the same released image scores
**9.62**, not 67.69 — a 58-point swing from wording alone (D13).

### 1c. MM-SafetyBench

| Method | Qwen2-VL* | Qwen-VL-Chat | Qwen2.5-VL | LLaVA-1.6 | Kimi-VL | GLM-4.1V |
|---|---|---|---|---|---|---|
| No Attack — **paper** | 26.19 | 21.49 | 33.45 | 35.06 | 41.79 | 43.69 |
| No Attack — **ours** | 22.88 | 18.27 | 22.88 | 48.08 | 39.81 | 54.42 |
| VAJM — **paper** | 5.42 | 11.73 | 26.79 | 30.18 | 35.36 | 36.73 |
| VAJM — **ours** | 20.19 | 28.46 | 27.88 | 48.65 | 36.73 | 52.12 |
| UMK — **paper** | **54.76** | 5.48 | 17.56 | 21.96 | 26.67 | 37.44 |
| UMK — **ours** | **18.08** | 15.19 | 23.46 | 44.04 | 34.04 | 56.54 |
| UltraBreak — **paper** | 57.26 | 53.10 | 45.83 | 71.90 | 54.58 | 67.08 |
| UltraBreak — **ours, released image** | 37.31 | 55.96 | 48.08 | 74.62 | 54.42 | **63.85** |
| UltraBreak — ours, code-faithful | 37.50 | 47.88 | 35.96 | 71.92 | 35.96 | 57.88 |
| UltraBreak — ours, paper-faithful | 33.85 | 40.96 | 35.38 | 72.31 | 35.96 | 57.31 |

Five of six UltraBreak cells land close; only the white-box Qwen2-VL runs 20 points low.

### The UMK column is a different experiment

| UMK on Qwen2-VL | ours | paper | delta |
|---|---|---|---|
| SafeBench | 1.27 | 97.78 | **−96.5** |
| AdvBench | 0.58 | 70.00 | **−69.4** |
| MM-SafetyBench | 18.08 | 54.76 | **−36.7** |

Across the five **black-box** columns our UMK agrees to a mean absolute error of 10.5 points
and is often *higher* than the paper. The failure is confined to the surrogate column, and
the cause is mechanical: **UMK optimises a GCG text suffix in the surrogate's token-embedding
space.** The paper ran UMK **white-box on Qwen2-VL**; ours is cross-architecture from
Vicuna-13B, so the suffix carries no meaning to Qwen2-VL. Full argument in
[`baselines/README.md`](baselines/README.md). VAJM shows no such profile — the paper's VAJM is
*worst* on Qwen2-VL (0.95, below its own No-Attack 18.41), so **the two baselines are not
measured under the same threat model and the table does not say so.**

---

## Table 2 {#table-2}

**Paper caption:** Average StrongREJECT scores (0–1) on open-source VLMs, AdvBench.

| Target | No Attack | VAJM | UMK | Ours (paper) |
|---|---|---|---|---|
| Qwen2-VL-7B | 0.00 | 0.00 | 0.36 | 0.45 |
| Qwen-VL-Chat | 0.02 | 0.01 | 0.00 | 0.55 |
| Qwen2.5-VL-7B | 0.02 | 0.04 | 0.04 | 0.27 |
| LLaVA-1.6 | 0.18 | 0.18 | 0.12 | 0.79 |
| GLM-4.1V | 0.02 | 0.03 | 0.04 | 0.19 |

**Not reproduced — and not for want of running it.** We scored the StrongREJECT *prompt set*
(313 rows) with the HarmBench classifier for the released image and four retrains
(`results/ext_*_strongreject_*`), but that yields a **binary ASR**, not StrongREJECT's
**0–1 willingness × ability** score. The native scorer needs an OpenAI key or the finetuned
SR evaluator model, neither of which is available here, so the two quantities are not
comparable and we do not pretend otherwise. Our HarmBench-judged figures, for reference:
released image **29.71%**, job 809 **21.73%** (v3, Qwen2-VL).

---

## Table 3 {#table-3}

> ### ⚠️ The ablation patches below are NOT at the paper's budget
>
> Every `full50_abl_*` patch was trained at **3001 epochs**; paper Table 4 states
> **1300 iterations**. So the "ours" ablation cells on this page are neither
> code-faithful nor paper-faithful — they are our own best-known-good configuration at
> 2.3× the paper's budget, which is very likely why the disagreement below is as large
> as it is. **Comparing them to the paper's ablations compares two training budgets.**
>
> `jobs/t36_paperfaithful.sh` (job 9134) retrains all six variants at **1300 iterations**
> with everything else matching Table 4 and the existing `repro_paper_faithful_1300`
> anchor, then re-evaluates. These tables will be replaced by those numbers.


**Paper caption:** ASR (%) of UltraBreak and its ablated variants on SafeBench and AdvBench.
Note the paper's Table 3 omits Kimi-VL and averages over 5 models.

### SafeBench

Paper row above, ours below. **Ours is 5 models where available; `·` marks a cell not yet run.**

| Variant | Qwen2-VL* | Qwen-VL-Chat | Qwen2.5-VL | LLaVA-1.6 | GLM-4.1V |
|---|---|---|---|---|---|
| w/o Jailbreak Image — paper | 15.87 | 51.11 | 12.38 | 92.06 | 7.62 |
| w/o Jailbreak Image — **ours** | **13.02** | **7.30** | **18.10** | **57.14** | **53.02** |
| w/o Constraints — paper | 89.84 | 49.21 | 27.30 | 83.81 | 47.62 |
| w/o Constraints — **ours** | **31.75** | · | · | · | · |
| w/o Semantic Loss (CE) — paper | 72.38 | 56.83 | 20.32 | 85.71 | 60.32 |
| w/o Semantic Loss (CE) — **ours** | **67.94** | · | · | · | · |
| w/o Attention Weighting — paper | 68.25 | 42.54 | 37.78 | 79.37 | 70.48 |
| w/o Attention Weighting — **ours** | **14.60** | · | · | · | · |
| Proposed — paper | 81.59 | 72.70 | 60.32 | 88.25 | 66.03 |
| Proposed (job 809) — **ours** | **83.49** | **76.51** | **36.51** | **89.84** | **64.76** |

### AdvBench

| Variant | Qwen2-VL* | Qwen-VL-Chat | Qwen2.5-VL | LLaVA-1.6 | GLM-4.1V |
|---|---|---|---|---|---|
| w/o Jailbreak Image — paper | 10.19 | 10.38 | 0.38 | 91.15 | 2.12 |
| w/o Jailbreak Image — **ours** | **1.35** | **3.46** | **0.77** | **48.65** | **9.81** |
| w/o Constraints — paper | 63.27 | 27.69 | 2.31 | 80.00 | 9.42 |
| w/o Constraints — **ours** | **2.12** | · | · | · | · |
| w/o Semantic Loss (CE) — paper | 57.69 | 51.54 | 2.88 | 88.85 | 17.31 |
| w/o Semantic Loss (CE) — **ours** | **15.77** | · | · | · | · |
| w/o Attention Weighting — paper | 47.50 | 42.54 | 15.77 | 79.37 | 29.62 |
| w/o Attention Weighting — **ours** | **1.15** | · | · | · | · |
| Proposed — paper | 72.69 | 71.92 | 35.77 | 92.88 | 30.00 |
| Proposed (job 809) — **ours** | **39.04** | **58.65** | **4.81** | **94.23** | **20.38** |

**Status: 26 of 50 cells.** The two rows that need no extra work — *w/o Jailbreak Image*
(the No-Attack baseline) and *Proposed* (the job-809 patch, already transferred in the
transfer study) — are complete across all five models. The three true ablation rows exist
only on the white-box column, because each ablation produced one patch that was evaluated
on Qwen2-VL alone. **The missing 24 cells need no training — only transfer evals of patches
already on disk** (`jobs/t36_fill.sh`).

**Where we already disagree.** *w/o Constraints* and *w/o Attention Weighting* are far more
damaging in our hands than the paper's — 31.75 and 14.60 against 89.84 and 68.25 on
SafeBench. In the paper, removing the constraints *raises* white-box ASR above the proposed
method (89.84 > 81.59); in ours it cuts it by more than half. Our ablations carry the D11
fix and 3000 steps, so they are not budget-starved. Our *Proposed* row tracks the paper
well on three of five models and misses Qwen2.5-VL badly (36.51 vs 60.32) — the same
Qwen2.5 gap seen in Table 1.

Also worth noting: **the paper's own Table 1 and Table 3 disagree.** Table 1 gives No-Attack
SafeBench on Qwen2-VL as **18.41**; Table 3's "w/o Jailbreak Image" gives **15.87**. Same
quantity, two values.

---

## Table 4 {#table-4}

**Paper caption:** Hyperparameters used in UltraBreak (Appendix C.1).

| Parameter | Paper | Ours | Match |
|---|---|---|---|
| Location min / max | 0 / 112 | 0 / 112 | ✅ |
| Rotation min / max | −15° / +15° | −15° / +15° | ✅ |
| Scale min / max | 0.8 / 1.2 | 0.8 / 1.2 | ✅ |
| TV weight λ_TV | 0.5 | 0.5 | ✅ |
| Embedding noise ε | 1e-4 | 1e-4 | ✅ |
| Attention temperature τ | 0.5 | 0.5 | ✅ |
| Affirming phrase | `[Jailbroken Mode]` | `[Jailbroken Mode]` | ✅ |
| Learning rate η | 0.01 | 0.01 | ✅ |
| **Iterations** | **1300** | 1300 (paper-faithful) · **3000** (job 809) | ⚠️ |
| Image size | 224×224 | 224×224 | ✅ |

Every hyperparameter matches. **The one deviation is deliberate and is a finding**: at the
paper's stated 1300 iterations our fixed retrain reaches only 23.49% SafeBench; it needs
**3000** to reach 83.49%. So the D11 normalisation bug is *necessary but budget-gated* —
fixing it alone is not enough at the paper's own iteration count (D11, `DISCREPANCIES.md`).

---

## Table 5 {#table-5}

**Paper caption:** Effect of the affirming phrase. GLM-4.1V-9B-Thinking, SafeBench.

| Affirming phrase | Paper | **Ours (v3)** | Ours (v1) | Δ vs paper |
|---|---|---|---|---|
| `[Jailbroken Mode]` | 66.03 | **60.63** | 64.44 | −5.4 |
| `[ACCESS AUTHORISED]` | 72.06 | **54.60** | 55.56 | −17.5 |
| `[CLASSIFIED INFO]` | 66.98 | **59.68** | 61.27 | −7.3 |

**Largely reproduces.** All three land in a tight 54.6–60.6 band, which is the qualitative
claim — the attack is only mildly sensitive to the exact phrase. The *ordering* still differs
(paper ACCESS > CLASSIFIED > JBM; ours JBM > CLASSIFIED > ACCESS), and judge choice does not
explain it: v1 preserves our ordering.

This table was the study's worst-looking result until **D14** was fixed. At the old
`max_new_tokens=512`, GLM was truncated mid-`<think>` on most rows and these read
39.1 / 30.5 / 32.7 — ~30 points low. The token budget, not the method, was the cause.

---

## Table 6 {#table-6}

> ### ⚠️ The ablation patches below are NOT at the paper's budget
>
> Every `full50_abl_*` patch was trained at **3001 epochs**; paper Table 4 states
> **1300 iterations**. So the "ours" ablation cells on this page are neither
> code-faithful nor paper-faithful — they are our own best-known-good configuration at
> 2.3× the paper's budget, which is very likely why the disagreement below is as large
> as it is. **Comparing them to the paper's ablations compares two training budgets.**
>
> `jobs/t36_paperfaithful.sh` (job 9134) retrains all six variants at **1300 iterations**
> with everything else matching Table 4 and the existing `repro_paper_faithful_1300`
> anchor, then re-evaluates. These tables will be replaced by those numbers.


**Paper caption:** SafeBench ASR (%) varying the TV loss weight. Victims: **Qwen-VL-Chat and
GLM-4.1V** (transfer), not the surrogate.

| TV weight | paper Qwen-VL-Chat | **ours Qwen-VL-Chat** | paper GLM-4.1V | **ours GLM-4.1V** | ours white-box Qwen2-VL |
|---|---|---|---|---|---|
| 0 | 61.27 | · | 70.48 | · | 31.75 |
| 0.2 | 77.78 | · | 66.98 | · | 40.00 |
| **0.5** | 72.70 | **76.51** | 66.03 | **64.76** | 83.49 |
| 1 | 65.40 | · | 60.95 | · | 56.51 |

**Status: 2 of 8 cells.** The TV=0.5 row is free — that patch is job 809, already
transferred to both of the paper's victims — and **both cells land close** (76.51 vs 72.70,
64.76 vs 66.03). The other six need transfer evals of three existing patches; no training
(`jobs/t36_fill.sh`).

**Not comparable, and an earlier claim of ours is retracted.** The paper sweeps TV and
measures **transfer to Qwen-VL-Chat and GLM**; we swept TV and measured the **white-box
Qwen2-VL**. Different models, different quantities. An earlier note here said "the paper's
TV=0.2 peak is contradicted (we get 0.5)" — that compared the wrong things and **is
withdrawn as a Table 6 verdict**.

What our sweep does show, as a separate and still-valid fact: **on the white-box surrogate
the TV optimum is ≈0.5**, and the curve is non-monotonic. Converting this to a real Table 6
verdict needs only transfer evals of the three existing TV patches on Qwen-VL-Chat and GLM —
no new training. Cheapest open item in the study.

---

## Table 9 {#table-9}

**Paper caption:** ASR (%) on frontier commercial models, SafeBench/AdvBench subset.

| Model | No Attack | VAJM | UMK | UltraBreak |
|---|---|---|---|---|
| GPT-5 | 24.00 | 24.00 | 24.00 | 26.00 |
| claude-sonnet-4.5 | 30.00 | 20.00 | 16.00 | 20.00 |

**🚫 Not attempted — paid API.** Generation code for the closed models is written and
error-wrapped in `evaluation/attack.py`, but no keys are configured. Reported as **untested,
not omitted**.

Worth observing even without running it: in the paper's own numbers UltraBreak **+2 points
over No Attack on GPT-5** and is **10 points below No Attack on Claude Sonnet 4.5**. On
frontier models the attack does not work, and the paper's table says so.

---

## Table 10 {#table-10}

**Paper caption:** Optimisation time per iteration (NVIDIA H100).

| Method | Paper (H100) | Ours — micro-benchmark (RTX 6000 Ada) | Ours — whole-run wall-clock |
|---|---|---|---|
| Cross-Entropy | 6.44 s | 0.24 s | 12.93 s/epoch |
| UltraBreak (semantic) | 7.55 s | 0.26 s | 14.32 s/epoch |
| **ratio** | **1.17×** | **1.08×** | **1.11×** |

**Reproduces.** Absolute times are not comparable across GPUs — and our two methods measure
different units (one row forward+backward vs a full 50-row epoch) — so the ratio is the
quantity, as the paper itself implies. Two independent measurements agree: **1.08× and
1.11×** against 1.17×. Breakdown of our ~11%: ≈6.5 points from the input-space constraints,
≈4 from attention-mode over token-mode.

---


---

---

## Protocol audit — does each table follow the paper's own protocol?

Prompted by a reader catching that the Table 3 ablations were trained at 3001 epochs
against the paper's stated 1300. Extending that check to **every** protocol dimension of
every table, not just iteration count.

| Table | corpus / budget | eval set | prompt form | judge | verdict |
|---|---|---|---|---|---|
| **1** SafeBench | 1300 iters ✅ (`cmp_pf_*`); released image needs none | **315** ✅ | TPG ✅ | v1 ≈ paper ✅ | **strict** |
| **1** AdvBench | as above | **520** ✅ | normalised ✅ (the paper's own D13 form) | ✅ | **strict** |
| **1** MM-Safety | as above | **520 ✗ — the paper uses 1680** | ✅ | ✅ | **deviation — being fixed, job 9136** |
| **2** StrongREJECT | released image ✅ | 313 ✅ | ✅ | ✗ native 0–1 scorer unavailable | blocked, documented |
| **3** ablations | **3001 ✗ → 1300** | 315 / 520 ✅ | ✅ | ✅ | **being fixed, job 9134** |
| **4** hyperparameters | descriptive | — | — | — | **strict** (every value matches) |
| **5** phrases | released image ✅ | 315 ✅ | ✅ | ✅ | **strict** |
| **6** TV sweep | **3001 ✗ → 1300** | 315 ✅ | ✅ | ✅ | **being fixed, job 9134** |
| **10** overhead | a per-step *ratio* — independent of total steps | — | — | — | **strict** |

### The eval-set sizes are not guesses

For each benchmark we solved for the denominators consistent with **every** value the paper
reports (2 dp, N ∈ [50, 2000]):

| benchmark | denominators consistent with all reported values | ours |
|---|---|---|
| SafeBench | 315, 630, 945, … (multiples) | **315** ✅ |
| AdvBench | 520, 1040, 1560 | **520** ✅ |
| **MM-SafetyBench** | **1680 — unique solution** | **520** ✗ |

1680 is the only denominator in that whole range consistent with all 24 reported
MM-SafetyBench values. Our 520 stratified subsample is an unbiased estimator carrying
≈2.2 points of sampling error — fine for ranking, **not** for a reproduction claim. Job
9136 re-runs the column at 1680 (40,320 generations, ~71 GPU-h) and prints how far each
520-row estimate sat from the full-set value.

### A limit on *any* strict reproduction here

Four of the six evaluated models sample, and `evaluation/attack.py` sets **no seed**:

| model | `do_sample` | top_k | top_p | temp | |
|---|---|---|---|---|---|
| Qwen2-VL-7B | true | **1** | 0.001 | 0.01 | deterministic (argmax-equivalent) |
| LLaVA-1.6 | false | — | — | — | deterministic (greedy) |
| Qwen2.5-VL-7B | true | — | — | 1e-06 | deterministic in practice |
| **Qwen-VL-Chat** | true | 0 | 0.3 | — | **stochastic** |
| **Kimi-VL** | true | — | — | 0.2 | **stochastic** |
| **GLM-4.1V** | true | **2** | 0.6 | 0.8 | **stochastic** |

This is not theoretical. `b15_jbm_safebench` and `cmp_img_sb315` are **byte-identical
configs**, and GLM scored **191/315** on one and **190/315** on the other — a re-run of the
same experiment disagreeing with itself by one row.

So three of Table 1's six columns are **single unseeded draws**, in the paper as much as
here, and no confidence interval anywhere accounts for decoding variance. Our Wilson
intervals cover sampling of *queries*, not of *decodings*. Quantifying it would mean
repeating one config n times per model; until then, differences of a point or two on
Qwen-VL-Chat, Kimi or GLM should not be read as real.

---

## Training-budget audit (2026-10-18)

Prompted by a reader catching that the Table 3 ablations were not at the paper's budget.
Every trained artifact in the study was then checked against **paper Table 4: 1300
iterations, lr 0.01, TV 0.5, τ 0.5**. **10 of 16 were off-budget**, all at 3001 epochs.

| artifact | epochs | feeds | action |
|---|---|---|---|
| `full50_abl_noconstraints` | 3001 | Table 3, Table 6 (TV=0) | **retraining @1300** — job 9134 |
| `full50_abl_ce` | 3001 | Table 3 | **retraining @1300** — job 9134 |
| `full50_abl_token` | 3001 | Table 3 | **retraining @1300** — job 9134 |
| `full50_abl_tv0p2` | 3001 | Table 6 | **retraining @1300** — job 9134 |
| `full50_abl_tv1p0` | 3001 | Table 6 | **retraining @1300** — job 9134 |
| `c2a_qwen2b_s3000` | 3001 | **Fig 2a** | **retraining @1300** — job 9135 |
| `c2b_llava_surrogate_s3000` | 3001 (never trained) | **Fig 2b** | **training @1300** — job 9135 |
| `full50_normfix_lr0p01_s3000` | 3001 | job 809 — our headline retrain, A4 transfer, Fig 3 | **kept, deliberately.** 1300 does not work; that *is* the D11 budget-gating finding. Never label it paper-config. |
| `full50_proj_projfix_s3000` | 3001 | D1 projection diagnostic | kept — our own experiment, not a paper table |
| `full50_proj_doublesquash_s3000` | 3001 | D1 projection diagnostic | kept — same |

**On budget already:** `repro_paper_faithful_1300`, `codefaithful_exact_1300`,
`full50_normfix_noproj_seed0`, `full50_normfix_lr0p003_s1300`, `c6_tau_inf_s1300`,
`full50_proj_unclamped_{unit,img}init`.

**What this does not touch.** Tables that use no patch we trained are unaffected:
**Table 1**'s UltraBreak row (the authors' released `ultrabreak.png`, plus our two
*labelled* branches `cmp_cf_*`/`cmp_pf_*`, both at 1300), **Table 5** (released image),
**Table 2** (released image), **Table 4** (descriptive), **Table 10** (a per-step time
ratio, independent of total steps), **FigStep** (authors' released images), and
**No-Attack** (no patch).

**Outstanding after 9134/9135 land:** Fig 3's texture comparison currently mixes
3000-step and released patches; it should be regenerated against the 1300-step ablations
so the contrast is at a matched budget.

### The rule this establishes

A patch trained at a budget the paper does not state cannot be used to reproduce a paper
table — only to report *our own* result, labelled as such. Job 809 is the one deliberate
exception and it is a finding in its own right, not a reproduction.

## Figures

Not tables, listed for completeness: **Fig 2a** (model-size transfer) ✅ 2B/7B done, 32B
dropped by decision; **Fig 2b** (surrogate choice) 🟡 unblocked — the LLaVA adapter's missing
`return` is fixed, ~8 GPU-h to train; **Fig 3** (text-like patterns) ✅ reproduced *with
measured texture statistics* — the unconstrained ablation is 4.3× rougher with 3.5× less
stroke-scale energy; **Fig 4/5** (loss landscapes) ✅; **Fig 6** (first-token distribution)
✅ entropy 1.3 → 3.3 nats with τ. Detail in
[`analysis_figures/README.md`](analysis_figures/README.md).
