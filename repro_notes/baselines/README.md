# A5 — Table 1 baselines (FigStep / VAJM / UMK) — Review & Analysis

**Date: 2026-10-18.** Table 1's comparative claim rests on three baselines. All three are
now measured. The headline is that **VAJM reproduces and UMK does not — and the way it
fails identifies an undocumented asymmetry in the paper's own table.**

Job `jobs/a5_vajm_umk.sh` (8956 → 8971 → 9000, self-chained across three 24h windows,
finished 2026-10-12). Data: [`a5_vajm_umk_results.csv`](a5_vajm_umk_results.csv), 36 cells.

---

## 1. What was run

Per the UltraBreak paper's stated protocol, **not** the vendored repos' defaults, which
contradict it in three places (all overridden — see [`../../baselines/patches/DEVIATIONS.md`](../../baselines/patches/DEVIATIONS.md)):

| | paper says | shipped code does | we ran |
|---|---|---|---|
| VAJM | derogatory corpus, **5,000** iters | same | 5,000 ✓ |
| UMK | "an additional **2,000** iterations" | hardcodes `num_iter=250` | **2,000** |
| UMK seed | "that optimised image" (VAJM's) | its own `prompt_unconstrained_init.bmp` | **VAJM's output** |
| few-shot corpus | **SafeBench-Tiny**, "consistent with UltraBreak" | 66 *unseeded* random AdvBench rows | **all 50 SafeBench-Tiny pairs** |

Both attacks trained on **MiniGPT-4 / Vicuna-13B**, the original repos' native stack.
VAJM's loss descended 2.59 → 1.52 over 5,001 iterations; UMK produced an image plus a
95-character GCG suffix. Both artifacts span the full 0–255 range (std 79.2 / 80.1).

---

## 2. VAJM reproduces

Judge v3, ours vs paper. **Black-box columns: mean absolute error 6.3 points, median 2.2.**

| benchmark | Qwen-VL-Chat | Qwen2.5-VL | LLaVA-1.6 | Kimi-VL | GLM-4.1V |
|---|---|---|---|---|---|
| SafeBench ours | 6.35 | 22.54 | 59.68 | 33.65 | 50.79 |
| SafeBench paper | 12.06 | 28.89 | 57.46 | 41.27 | 67.62 |
| AdvBench ours | 1.15 | 0.96 | 21.15 | 3.46 | 3.08 |
| AdvBench paper | 0.96 | 0.38 | 19.42 | 3.65 | 3.65 |

AdvBench is near-identical across the board. SafeBench is close on LLaVA and Qwen2.5-VL
and runs 6–17 points low on Qwen-VL-Chat, Kimi and GLM. **Verdict: reproduces.**

MM-SafetyBench is the weak spot — ours runs 8–18 points *high* on four of six models. Our
MM-SafetyBench is a seed-0 stratified 520-row subsample of the full 1680, so a few points
of that is sampling error (binomial SE ≈ 2.2), but not 18.

---

## 3. UMK does not reproduce — and the failure is diagnostic

| UMK, **Qwen2-VL** (the paper's surrogate column) | ours | paper | delta |
|---|---|---|---|
| SafeBench | **1.27** | **97.78** | **−96.5** |
| AdvBench | **0.58** | **70.00** | **−69.4** |
| MM-SafetyBench | **18.08** | **54.76** | **−36.7** |

On the five *black-box* columns ours is in range and frequently **higher** than the paper
(SafeBench: LLaVA 61.27 vs 20.63, Kimi 32.06 vs 12.70, GLM 60.95 vs 50.79; mean absolute
error 10.5 points). The failure is confined entirely to one column.

**Why.** UMK jointly optimises an image *and a GCG text suffix*, and the suffix is selected
token-by-token in the **surrogate's embedding space**. A suffix tuned on Vicuna-13B's
tokenizer is not a sentence — it is a string chosen for its gradient behaviour in one
specific embedding matrix — and it carries no meaning to Qwen2-VL. Our UMK is therefore a
**cross-architecture transfer** result; the paper's 97.78 is a **white-box** result.

> **The paper ran UMK with Qwen2-VL-7B as the surrogate.** It never says so, but the
> numbers leave no room: 97.78 / 70.00 / 54.76 on Qwen2-VL against ≤50.79 / ≤16.35 /
> ≤37.44 on every other model is a textbook white-box profile.

---

## 4. The asymmetry this exposes in Table 1

**VAJM and UMK are not measured the same way in the paper, and the table does not say so.**

- UMK is white-box on Qwen2-VL (97.78 there).
- VAJM is *not*: its **lowest** SafeBench cell is Qwen2-VL at **0.95 — below No-Attack's
  18.41**, i.e. the image actively suppresses compliance. Our MiniGPT-4-trained VAJM scores
  12.06 on that same cell, *higher* than the paper's. A genuinely white-box VAJM should not
  be the worst cell in its own row.

(An adversarial image lowering ASR is real, not an artifact — we measured it independently
in the shift study: the authors' own patch gave a significant **−15.33** point uplift on
LLaVA once the prompt scaffold was removed.)

**This flatters UltraBreak's comparison in a specific, checkable way.** UMK's reported
black-box average is **20.00** precisely because its one strong cell — the 97.78 — is
excluded from that average. UltraBreak's black-box average is 71.05. Had UMK been run
black-box like VAJM, its black-box columns would look like ours do: LLaVA 61.27, GLM 60.95,
Kimi 32.06 — a black-box average near **35**, not 20.

Neither reading rescues UMK as a *better* attack than UltraBreak. The point is narrower and
still worth stating: **the baselines are measured under different threat models, the table
presents them as one, and the choice moves the comparison in UltraBreak's favour.**

---

## 5. FigStep

Done earlier (authors' released typography images, SafeBench only per the paper, 6 models).
Numbers and verdict in [[../REPRO_CHECKLIST]].

---

## 6. What this supports / what it does not

**Supports.** VAJM reproduces at the paper's own protocol. UMK's black-box behaviour is
reproducible and stronger than reported. The paper's UMK row is a white-box measurement
presented alongside a non-white-box VAJM row.

**Does not support.** We did not run UMK white-box on Qwen2-VL, so we have not *directly*
reproduced the 97.78 — the surrogate inference rests on the numbers' shape plus the
mechanics of GCG, not on a replication. Doing that properly means porting UMK's GCG loop
onto `Qwen2Adapter` (its attacker is written against MiniGPT-4's prompt wrapper and
embedding accessors), which is implementation work, not a config change. Until then our
UMK row must be labelled **cross-architecture transfer** wherever it is quoted.

One patch per method, no seed replication, and MM-SafetyBench is a 520-row subsample.
