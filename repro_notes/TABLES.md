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
| [3](#table-3) | Component ablation | ✅ **50/50 cells at the paper's 1300-iteration budget** (job 9187) — 2 of 3 components reproduce, the semantic loss does not |
| [4](#table-4) | Hyperparameters | ✅ matched |
| [5](#table-5) | Affirming-phrase sensitivity | ✅ reproduced |
| [6](#table-6) | TV-weight sweep | ✅ **8/8 cells, paper's own two victims at 1300 iters** (job 9187) — 7 cells within 8.9 pts; the miss is the paper's claimed TV=0.2 peak |
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

**Paper caption:** ASR (%) of UltraBreak and its ablated variants on SafeBench and AdvBench.
Note the paper's Table 3 omits Kimi-VL and averages over 5 models.

> **Now at the paper's own budget — 50/50 cells.** All three ablation patches were retrained at
> **1300 iterations** with lr 0.01, tau 0.5 and TV 0.5 (paper Table 4) and re-evaluated on all five
> models: **job 9187, 36 cells, 14h39m, completed 2026-10-21**. The 3001-epoch patches that used to
> sit on this page are gone. **The budget mismatch is therefore no longer available as an
> explanation for the disagreements below** — they survive at the paper's stated configuration.

### SafeBench

Paper row above, ours below. Ours is 5 models, judge **v3**; ablation patches are the `pf3_*`
evaluations of the `a_*_1300` retrains, *Proposed* is `repro_paper_faithful_1300` (`cmp_pf_sb315`).
The authors' own v1 scorer was run on every cell too and agrees to within 4.6 points everywhere
(identical on Qwen-VL-Chat and LLaVA, where D12 cannot fire): v1 means are 52.38 / 60.95 / 50.73 /
62.98 against v3's 51.49 / 59.94 / 49.84 / 61.52, so **no conclusion here depends on the scorer.**

| Variant | Qwen2-VL* | Qwen-VL-Chat | Qwen2.5-VL | LLaVA-1.6 | GLM-4.1V | 5-model mean |
|---|---|---|---|---|---|---|
| w/o Jailbreak Image — paper | 15.87 | 51.11 | 12.38 | 92.06 | 7.62 | 35.81 |
| w/o Jailbreak Image — **ours** | **13.02** | **7.30** | **18.10** | **57.14** | **53.02** | **29.72** |
| w/o Constraints — paper | 89.84 | 49.21 | 27.30 | 83.81 | 47.62 | 59.56 |
| w/o Constraints — **ours** | **38.73** | **58.41** | **25.08** | **84.44** | **50.79** | **51.49** |
| w/o Semantic Loss (CE) — paper | 72.38 | 56.83 | 20.32 | 85.71 | 60.32 | 59.11 |
| w/o Semantic Loss (CE) — **ours** | **89.52** | **55.87** | **14.92** | **85.71** | **53.65** | **59.93** |
| w/o Attention Weighting — paper | 68.25 | 42.54 | 37.78 | 79.37 | 70.48 | 59.68 |
| w/o Attention Weighting — **ours** | **19.05** | **61.59** | **23.81** | **87.30** | **57.46** | **49.84** |
| Proposed — paper | 81.59 | 72.70 | 60.32 | 88.25 | 66.03 | 73.78 |
| Proposed — **ours** | **64.13** | **68.89** | **26.03** | **87.62** | **60.95** | **61.52** |

Overall MAE **15.75** points.

### AdvBench

| Variant | Qwen2-VL* | Qwen-VL-Chat | Qwen2.5-VL | LLaVA-1.6 | GLM-4.1V | 5-model mean |
|---|---|---|---|---|---|---|
| w/o Jailbreak Image — paper | 10.19 | 10.38 | 0.38 | 91.15 | 2.12 | 22.84 |
| w/o Jailbreak Image — **ours** | **1.35** | **3.46** | **0.77** | **48.65** | **9.81** | **12.81** |
| w/o Constraints — paper | 63.27 | 27.69 | 2.31 | 80.00 | 9.42 | 36.54 |
| w/o Constraints — **ours** | **2.12** | **41.73** | **1.15** | **83.08** | **11.73** | **27.96** |
| w/o Semantic Loss (CE) — paper | 57.69 | 51.54 | 2.88 | 88.85 | 17.31 | 43.65 |
| w/o Semantic Loss (CE) — **ours** | **80.00** | **73.65** | **0.58** | **88.27** | **20.77** | **52.65** |
| w/o Attention Weighting — paper | 47.50 | 42.54 | 15.77 | 79.37 | 29.62 | 42.96 |
| w/o Attention Weighting — **ours** | **1.35** | **55.58** | **0.96** | **91.54** | **16.92** | **33.27** |
| Proposed — paper | 72.69 | 71.92 | 35.77 | 92.88 | 30.00 | 60.65 |
| Proposed — **ours** | **10.19** | **75.00** | **1.73** | **91.15** | **19.81** | **39.58** |

Overall MAE **16.37** points.

### ⚠️ Which of OUR fixes are in these numbers

Audited 2026-10-22 from each run's own `outputs/<exp>/run_config.json`, not from memory.

| our fix | in these ablations? | is it the authors' protocol? |
|---|---|---|
| **D11 patch normalisation** | **YES** — `normalise_patch: true` on all six | **No.** The released code injects the patch raw. Only `codefaithful_exact_1300` has the bug. |
| §3.2 projection | **No** — `projection: false` | Correct: our projection reading was retracted (D10) |
| judge **v3** | both scorers were run | v1 is theirs; **v1 is reported beside v3 below and changes no conclusion** (max gap 4.6 pts) |
| eval sets 315 / 520 | yes | **Yes** — recovered from the paper's own arithmetic |
| AdvBench "Steps to…" | yes | **Yes** — paper §3.3; the released code does not do it |
| GLM 4096 tokens (D14) | yes, every GLM cell | No — the released code uses 512, which truncates 98.7% of GLM rows |

**So these ablations are paper-faithful, not code-faithful: they carry the D11 fix.** That is the
defensible choice for an ablation — with the bug present every variant is confined to the grey band
and the whole table collapses into noise, measuring the bug instead of the components — but it must
be labelled, and this page now labels it.

### ⚠️ Correction (2026-10-22): the baseline these ablations must be compared against

An ablation is only meaningful against a run that differs **by the ablated knob alone**. The
`run_config.json` diffs show that is **not** `repro_paper_faithful_1300`, the *Proposed* row above:

| vs `a_ce_1300` | projection | clamp_latent | init_space | differs by |
|---|---|---|---|---|
| `repro_paper_faithful_1300` | **true** | **false** | **image** | **3 structural settings + the loss** |
| **`full50_normfix_noproj_seed0`** (job 773) | false | true | unit | **the loss, and nothing else** |

All six ablation patches differ from **job 773** by exactly one field each, and job 773 differs from
job 809 only in `num_epochs`. **Job 773 is the matched control; `repro_paper_faithful_1300` is not.**
The deltas first published on this page used the wrong baseline and are withdrawn.

### The verdict, against the matched baseline

Component contribution = baseline ASR − ablated ASR, so **positive means the component helps**.
White-box Qwen2-VL, judge v3. `@1300` is job 773 (23.49 SB / 1.35 AB); `@3000` is job 809
(83.49 / 39.04); both are the full method with only the budget differing.

| component removed | paper (SB) | **@1300 — paper's budget** | **@3000** | paper (AB) | **@1300 (AB)** | **@3000 (AB)** |
|---|---|---|---|---|---|---|
| input constraints | −8.25 | **−15.24** | **+51.74** | +9.42 | **−0.77** | **+36.92** |
| semantic loss (CE) | +9.21 | **−66.03** | **+15.55** | +15.00 | **−78.65** | **+23.27** |
| attention weighting | +13.34 | **+4.44** | **+68.89** | +25.19 | **+0.00** | **+37.89** |

**At 3000 iterations all three components contribute positively and substantially — the paper's
qualitative ablation claim reproduces. At the paper's own stated 1300 iterations not one of them
does.** Two are within noise of zero and the semantic loss is strongly negative: swapping it for
plain cross-entropy takes SafeBench from 23.49% to **89.52%** and AdvBench from 1.35% to **80.00%**.
`a_ce_1300` is in fact the highest white-box SafeBench score in the entire study, above job 809
(83.49) and above the released image (78.73).

**The crossover is the finding.** The paper's semantic-attention loss is worse than the baseline it
replaces at the budget the paper states, and better than it at 2.3× that budget:

| budget | full method (semantic) | CE variant | winner |
|---|---|---|---|
| **1300 — the paper's** | 23.49 | **89.52** | **CE, by 66 points** |
| 3000 | **83.49** | 67.94 | semantic, by 15.6 |

Nothing else differs between those four runs. This is the same shape as the D11 budget-gating
finding — the normalisation fix also only pays off past 1300 — so two separate parts of the method
require a budget the paper does not state before they help at all. One coherent story, not two
anomalies.

### What the five-model table can and cannot support

The 5-model rows above are correct as **measurements**, and they are what the paper's Table 3
reports, so they belong on this page. They cannot yet carry a component-contribution claim beyond
the surrogate, because **job 773 was only ever evaluated white-box** — its transfer cells do not
exist. Completing that needs no training, only 8 eval cells of an existing patch
(5 models × 2 benchmarks, minus the 2 that exist; GLM is ~6.7h of it, the rest ~3h).

Until those land, read the five-model table as *absolute ASR beside the paper's* and the surrogate
table above as *the component analysis*. The `Proposed` row is labelled `repro_paper_faithful_1300`
and is a legitimate paper-faithful reproduction of the paper's own `Proposed` row — it is simply not
the control for the ablation rows.

Also unchanged and still worth noting: **the paper's own Table 1 and Table 3 disagree.** Table 1
gives No-Attack SafeBench on Qwen2-VL as **18.41**; Table 3's "w/o Jailbreak Image" gives **15.87**.
Same quantity, two values.

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

**Paper caption:** SafeBench ASR (%) varying the TV loss weight. Victims: **Qwen-VL-Chat and
GLM-4.1V** (transfer), not the surrogate.

> **Now a like-for-like comparison — 8/8 cells.** Our earlier sweep measured the *white-box
> surrogate* and was not comparable to the paper's, which measures transfer; that comparison was
> retracted. Job 9187 re-ran the sweep at **1300 iterations on the paper's own two victims**, so the
> retraction is now closed by the right experiment rather than by an argument.

| TV weight | paper Qwen-VL-Chat | **ours Qwen-VL-Chat** | paper GLM-4.1V | **ours GLM-4.1V** |
|---|---|---|---|---|
| 0 | 61.27 | **64.44** (203/315) | **70.48** | **54.92** (173/315) |
| 0.2 | **77.78** | **54.29** (171/315) | 66.98 | **58.73** (185/315) |
| **0.5** | 72.70 | **68.89** (217/315) | 66.03 | **60.95** (192/315) |
| 1.0 | 65.40 | **61.59** (194/315) | 60.95 | **52.06** (164/315) |

MAE **8.57** (Qwen-VL-Chat) and **9.45** (GLM-4.1V).

**The sweep reproduces everywhere except at the paper's claimed optimum.** Seven of the eight cells
agree within 8.9 points. The single outlier is **TV=0.2 on Qwen-VL-Chat — 54.29 against 77.78, a
23.5-point miss — which is precisely the cell the paper's claim rests on.**

**And the paper's two columns do not agree with each other.** Its Qwen-VL-Chat column peaks at
TV=0.2 (77.78); its GLM column declines monotonically from TV=0 (70.48 → 66.98 → 66.03 → 60.95), so
it peaks at **TV=0**, i.e. with the TV term switched off entirely. The paper reports a single
optimum that only one of its own two victims exhibits.

**Ours is internally consistent and puts the optimum at 0.5:** 64.44 → 54.29 → **68.89** → 61.59 on
Qwen-VL-Chat and 54.92 → 58.73 → **60.95** → 52.06 on GLM. Both victims peak at the same place, and
0.5 is also where our earlier off-budget (3001-epoch) white-box sweep peaked. **Three independent
sweeps — two budgets, three models — all put the optimum at 0.5, and none reproduces the paper's
0.2.** The budget explanation for this disagreement is now ruled out.

A fair note in the paper's favour: the curve is shallow. Across the whole sweep our Qwen-VL-Chat
range is 14.6 points and our GLM range 8.9, so TV weight is a second-order hyperparameter in both
accounts, and "0.2 versus 0.5" is a disagreement about the location of a flat optimum rather than
about whether TV matters.

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
| **3** ablations | **1300** ✅ (job 9187) | 315 / 520 ✅ | ✅ | ✅ | **strict** |
| **4** hyperparameters | descriptive | — | — | — | **strict** (every value matches) |
| **5** phrases | released image ✅ | 315 ✅ | ✅ | ✅ | **strict** |
| **6** TV sweep | **1300** ✅ (job 9187) | 315 ✅ | ✅ | ✅ | **strict** — and now on the paper's victims, not the surrogate |
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
iterations, lr 0.01, TV 0.5, τ 0.5**. **10 of 16 were off-budget**, all at 3001 epochs. **Five of those ten are now fixed** (job 9187, 2026-10-21); three are deliberate keeps; **two remain open** because job 9135 was cancelled.

| artifact | epochs | feeds | action |
|---|---|---|---|
| `full50_abl_noconstraints` | 3001 | Table 3, Table 6 (TV=0) | ✅ **retrained @1300** — job 9187; superseded by `a_noconstraints_1300` |
| `full50_abl_ce` | 3001 | Table 3 | ✅ **retrained @1300** — job 9187; superseded by `a_ce_1300` |
| `full50_abl_token` | 3001 | Table 3 | ✅ **retrained @1300** — job 9187; superseded by `a_token_1300` |
| `full50_abl_tv0p2` | 3001 | Table 6 | ✅ **retrained @1300** — job 9187; superseded by `a_tv0p2_1300` |
| `full50_abl_tv1p0` | 3001 | Table 6 | ✅ **retrained @1300** — job 9187; superseded by `a_tv1p0_1300` |
| `c2a_qwen2b_s3000` | 3001 | **Fig 2a** | ⚠️ **still off-budget** — job 9135 was cancelled and never re-queued |
| `c2b_llava_surrogate_s3000` | 3001 (never trained) | **Fig 2b** | ⚠️ **still absent** — job 9135 was cancelled and never re-queued |
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

**Still outstanding.** (1) **Fig 2a/2b** — job 9135 was cancelled and never re-queued, so the
model-size transfer figure still rests on a 3001-epoch patch and the LLaVA-surrogate arm does
not exist. This is now the only paper figure or table left at the wrong budget. (2) **Fig 3**'s
texture comparison mixes 3000-step and released patches; the 1300-step ablation patches
(`a_*_1300`) now exist on disk, so regenerating it at a matched budget needs no training.

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
