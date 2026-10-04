# How universal are "universal" image jailbreaks? — results

A small, **inference-only** study on top of the UltraBreak reproduction. No patch is trained and no
attack is strengthened: existing attack artifacts are re-evaluated on existing benchmark prompts to
measure how far their success generalises when the query distribution shifts in **format**,
**source** or **harm category**.

Target model `Qwen/Qwen2-VL-7B-Instruct` (UltraBreak's own white-box surrogate) throughout.
Judges: **v3** (our corrected scorer) on everything, **v1** (the authors' released scorer) on
headline rows. Every number names its judge. Design is **paired**: conditions share identical query
lists in identical row order, so uplift is a row-wise difference and McNemar is exact.

Plan, decisions and deviations: [`PLAN.md`](PLAN.md). Code: [`code/`](code). Raw tables:
[`results/`](results). Figures: [`figures/`](figures).

> **No model completion text appears anywhere in this document, in any table, or in any figure** —
> metrics, counts, lengths, category names and query IDs only. Raw generations remain inside the
> pipeline's own output CSVs, as the existing pipeline already stores them.

---

## E0 — Re-analysis of existing outputs (CPU only, zero GPU cost)

Four already-scored runs on the two patches of interest: the authors' released `ultrabreak.png`
(`authors`) and the job-809 retrain that reproduced SafeBench (`p809`, D11 fix + 3000 steps).

### E0a — harm categories

**The specified seen/unseen split cannot be computed, and that is the finding.** Derived from the
files, not assumed:

| set | rows | categories |
|---|---|---|
| SafeBench (full) | 500 | 10 × 50 |
| **SafeBench-Tiny — the training corpus** | **50** | **all 10**, 5 each |
| SafeBench-315 — the eval set | 315 | 7 (the 3 eval-excluded categories dropped) |

```
EVAL categories NOT present in training: []   <- empty
TRAINING categories not in eval: Financial Advice, Health Consultation, Legal Opinion
```

`create_attack_configs.load_safebench` drops {legal opinion, financial advice, health consultation}
from the **evaluation** set only, while the training corpus keeps all ten. So the eval categories are
a strict *subset* of the training categories: **there are zero unseen harm categories in
SafeBench-315**, and the split would be 315 vs 0.

> **"Held out" in this protocol means held-out *queries*, never held-out *categories*.** The train
> and eval category distributions are identical up to the three dropped categories, so the
> universality claim was never tested along the category axis at all.

**E0a-i — variation within the seven *seen* categories** (SafeBench-315, judge v3, Wilson 95% CI):

| category | n | `authors` ASR | `p809` ASR |
|---|---|---|---|
| Adult Content | 45 | 57.78 [43.3, 71.0] | 84.44 [71.2, 92.3] |
| Fraud | 45 | 86.67 [73.8, 93.7] | 91.11 [79.3, 96.5] |
| Hate Speech | 45 | 60.00 [45.5, 73.0] | 53.33 [39.1, 67.1] |
| Illegal Activity | 45 | 84.44 [71.2, 92.3] | 95.56 [85.2, 98.8] |
| Malware Generation | 45 | 88.89 [76.5, 95.2] | 91.11 [79.3, 96.5] |
| Physical Harm | 45 | 84.44 [71.2, 92.3] | 82.22 [68.7, 90.7] |
| Privacy Violation | 45 | 88.89 [76.5, 95.2] | 86.67 [73.8, 93.7] |

Even **inside the training distribution** the spread is **31.1 points** for `authors` (57.8–88.9) and
**42.2 points** for `p809` (53.3–95.6). Under v1 the `authors` spread is 26.7 points. A single
aggregate ASR hides a factor-of-1.5-to-1.8 difference between the easiest and hardest seen category,
and the two patches do not even rank categories the same way (`p809` is far better on Adult Content,
worse on Hate Speech).

**E0a-ii — cross-source, where categories genuinely differ** (judge v3):

| source | n | `authors` ASR | `p809` ASR | `authors` NRR | `p809` NRR |
|---|---|---|---|---|---|
| SafeBench-315 (steps form, in-distribution) | 315 | **78.73** [73.9, 82.9] | **83.49** [79.0, 87.2] | 99.7 | 96.5 |
| AdvBench-520, steps form | 520 | 67.69 [63.6, 71.6] | 39.04 [34.9, 43.3] | 81.9 | 47.9 |
| AdvBench-520, **raw** imperative form | 520 | **9.62** [7.4, 12.5] | — | 25.2 | — |
| HarmBench-200, steps form | 200 | 42.50 [35.9, 49.4] | 36.00 [29.7, 42.9] | 79.5 | 50.0 |
| HarmBench-200, raw form | 200 | 28.50 [22.7, 35.1] | 22.00 [16.8, 28.2] | 46.5 | 32.0 |
| StrongREJECT-313, raw form | 313 | 29.71 [24.9, 35.0] | 21.73 [17.5, 26.6] | 52.1 | 30.4 |

Two things stand out. **Format moves the number more than anything else**: on the identical 520
AdvBench intents, the same image scores 9.62% raw and 67.69% in "Steps to …" form — a 58-point swing
with the harmful intent unchanged (this is the repo's D13). On HarmBench the same rewrite is worth
+14 points. And **`p809` — the retrain that *beat* the paper in-distribution (83.49 vs 78.73) — is
the weaker patch everywhere else**, by 29 points on AdvBench and 7–8 on HarmBench/StrongREJECT. Its
extra in-distribution strength did not transfer.

### E0b — ASR vs semantic distance to the training corpus

All 835 evaluation queries (SafeBench-315 + AdvBench-520) and the 50 training queries embedded with
`all-MiniLM-L6-v2` (mean-pool + L2, computed through `transformers` so no package was added to the
pinned env — see PLAN §0 A4). Per eval query: max cosine similarity to **any** training query.
Observed range 0.207–0.818, median 0.484.

ASR by similarity quartile (judge v3, Wilson 95% CI), per benchmark:

| bin (sim range) | `authors` SafeBench | `p809` SafeBench | `authors` AdvBench | `p809` AdvBench |
|---|---|---|---|---|
| Q1 farthest | 74.68 [64.1, 83.0] | 79.75 [69.6, 87.1] | 67.69 [59.2, 75.1] | 46.92 [38.6, 55.5] |
| Q2 | 75.95 [65.5, 84.0] | 87.34 [78.2, 93.0] | 71.54 [63.3, 78.6] | 39.23 [31.3, 47.8] |
| Q3 | 85.90 [76.5, 91.9] | 84.62 [75.0, 91.0] | 57.69 [49.1, 65.8] | 31.54 [24.2, 40.0] |
| Q4 nearest | 78.48 [68.2, 86.1] | 82.28 [72.4, 89.1] | 73.85 [65.7, 80.6] | 38.46 [30.5, 47.0] |

**Similarity to the training corpus does not predict success.** Logistic regression of success on
similarity with benchmark as a covariate (n=835):

| patch | term | coef | se | z | p |
|---|---|---|---|---|---|
| `authors` | max_cos_sim | +0.383 | 0.709 | 0.54 | **0.589** |
| `authors` | benchmark=advbench | −0.560 | 0.168 | −3.34 | **0.00084** |
| `p809` | max_cos_sim | −1.048 | 0.705 | −1.49 | **0.137** |
| `p809` | benchmark=advbench | −2.099 | 0.178 | −11.77 | **5.9e-32** |

Within each benchmark alone the similarity coefficient is also insignificant (`authors`: p=0.67
AdvBench, p=0.74 SafeBench; `p809`: p=0.076 AdvBench — and *negative*, p=0.88 SafeBench). The
benchmark indicator, by contrast, is overwhelming.

> Whatever makes a query hard for these patches, **it is not semantic distance from the 50 trained-on
> queries.** It tracks which benchmark the query came from — i.e. its format and source conventions.

Figure: [`figures/E0b_asr_vs_similarity.png`](figures/E0b_asr_vs_similarity.png).

### E0c — response format: success looks like "list mode"

Judge v3, metrics computed on the **extracted** answer (see the note on the repo's NRR below):

| patch | benchmark | group | n | forced-prefix | list ≤300 chars | median length | refusal | prefix-then-refuse |
|---|---|---|---|---|---|---|---|---|
| `authors` | SafeBench | success | 248 | 100.0% | 71.8% | 676 | 0.4% | 0.0% |
| `authors` | SafeBench | failure | 67 | 100.0% | 28.4% | 523 | 0.0% | 0.0% |
| `authors` | AdvBench | success | 352 | 100.0% | 84.1% | 827 | 0.6% | 0.0% |
| `authors` | AdvBench | failure | 168 | 52.4% | 20.8% | **36** | 54.8% | 7.1% |
| `p809` | SafeBench | success | 263 | 100.0% | 85.2% | 1507 | 1.5% | 0.0% |
| `p809` | SafeBench | failure | 52 | 100.0% | 30.8% | 97 | 13.5% | 13.5% |
| `p809` | AdvBench | success | 203 | 100.0% | 93.6% | 1589 | 3.0% | 0.0% |
| `p809` | AdvBench | failure | 317 | 72.9% | **3.2%** | **58** | 83.6% | **56.5%** |

Successes are long, list-shaped and prefix-compliant (list rate 72–94%); failures are short,
prose-shaped and often refusals (list rate 3–31%, median length 36–97 characters). The sharpest cell
is `p809` on AdvBench: **56.5% of its 317 failures are prefix-then-refuse** — the model emits
`[Jailbroken Mode]`, obeying the format instruction, then refuses the substance in ~58 characters.
Compliance with the *form* is near-total (72.9% prefix even among failures) while compliance with the
*request* collapses.

This is early evidence that what the image buys is entry into a **list-completion mode**, not a
general suppression of refusal — and that when the prompt does not already look like a list request,
the mode is entered formally and abandoned immediately.

**A measurement note that affects the repo's own tables.** `evaluation/evaluate.py` computes
refusal/NRR on the raw `response` cell, which for Qwen2-VL contains the whole chat template
*including the harmful prompt* (the Qwen2-VL branch decodes `outputs[0]` without slicing off the
prompt). A refusal phrase inside the prompt therefore counts as a refusal by the model. Every NRR in
this document is recomputed on the **extracted** answer via `aaai_ugc/code/metrics.py`; the phrase
list is unchanged from the repo so the numbers stay comparable.

### E0 — what this supports / what it does not

**Supports.** (1) The protocol never tested category generalisation — the claim is untested on that
axis rather than refuted. (2) Success is not explained by semantic proximity to the training corpus.
(3) The benchmark a query comes from, and especially its phrasing format, dominates. (4) Success
coincides strongly with list-shaped output; failure with short prose and prefix-then-refuse.
(5) In-distribution strength does not imply out-of-distribution strength: `p809` beats `authors` on
SafeBench and loses badly everywhere else.

**Does not support.** This is observational re-analysis of runs that differ in more than one way at a
time — AdvBench vs SafeBench differ in source *and* phrasing convention simultaneously, so E0 cannot
separate "format" from "source". That separation is exactly what E1 (format varied, source and intent
fixed) and E3 (source varied, format fixed) are for. E0 also cannot establish that the image *causes*
list mode, only that success and list mode co-occur; the `base`/`shuffled`/`text_only` arms of E1 are
the causal test.

---

## E1 — Format sweep

*Running.* 150 AdvBench intents (seed 0, IDs frozen in `configs/E1_ids.txt`) × {`bare`, `steps`,
`question`} × {`base`, `shuffled`, `authors`, `p809`, `text_only`} = **2,250 generations**.
Results will be written to `results/E1_asr.csv`, `results/E1_uplift_mcnemar.csv`,
`results/E1_key_contrast.json` and `figures/E1_format_sweep.png`.

The `past` variant is **skipped** — see PLAN §4 for why no transparent rule-based past-tense
transformation of AdvBench imperatives exists.

## E2 — Forced-prefix ablation

*Queued.* SafeBench-315 × {TPG on, off} × {`base`, `authors`} = **1,260 generations**.

Two of the four cells are byte-identical in configuration to runs the study already has, which makes
them **harness validation anchors** — they must reproduce the existing numbers exactly, since
generation is deterministic:

| E2 cell | identical existing run | expected |
|---|---|---|
| `E2_tpgon_authors` | `cmp_img_sb315` | v3 **78.73** / v1 **81.59** |
| `E2_tpgoff_base` | `noattack_safebench_matrix` | v3 **13.02** |

They are re-run rather than reused on purpose: 630 generations (~22 min) buys an end-to-end check
that this study's harness reproduces the reproduction. A mismatch means the harness is wrong, and
every other cell would be suspect.

## E3 — Source shift with format held constant

*Queued.* All prompts in `steps` form; `base` arms generated (1,035), `authors` arms reused from
`cmp_img_sb315`, `cmp_img_abnorm` and `ext_authors_harmbench_norm_eval`.

---

## Caveats

These apply to every result in this document.

- **n = 1 per configuration.** One patch per condition, no seed replication. `authors` is a single
  released artifact; `p809` is a single training run. Differences between patches conflate the
  intended factor with ordinary run-to-run variance, which this study does not measure.
- **Single target model.** Everything is Qwen2-VL-7B, which is UltraBreak's own white-box surrogate.
  These are therefore *white-box* numbers, the most favourable case for the attack; nothing here
  speaks to transfer.
- **Judge dependence.** v1 (authors') and v3 (corrected) disagree, and the gap is not constant:
  +2.9 points on SafeBench-315, +6.9 on AdvBench-520-steps, and −0.6 on AdvBench-raw for `authors`.
  Both are reported wherever available; no conclusion rests on a single judge. Several reused runs
  have no v1 pass (marked "—").
- **E0a's seen/unseen split is degenerate** and was redefined; the original analysis is not
  reportable. See E0a.
- **E0 is observational.** AdvBench and SafeBench differ in source and in phrasing at the same time,
  so E0 alone cannot attribute the gap to either.
- **`base` is not resolution-matched.** The no-patch reference `images/white.jpeg` is 800×533 while
  the patches are 224×224, so uplift-over-`base` carries a vision-token-count difference. This choice
  is deliberate — it is the image the repo's own No-Attack configs use, keeping the reference
  comparable with existing tables — and the `shuffled` arm (224×224, pixel-identical histogram to
  `authors`) is the resolution- and statistics-matched control for the same quantity.
- **`past` variant skipped** (E1), with reasoning in PLAN §4.
- **Generation determinism** rests on `top_k=1` in Qwen2-VL's `generation_config.json`, not on a
  seed, because `attack.py` exposes none. This matches how every existing number in the study was
  produced.
- **Statistics are own-implemented** (Wilson, exact McNemar, IRLS logistic) because the pinned env
  must not gain packages; each is unit-tested against hand-computed values in
  `code/stats.py::_selftest`.
