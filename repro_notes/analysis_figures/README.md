# Analysis Figures & Tables (A10, C2a, C4, C6) — Review & Analysis

**Date: 2026-08-28.** The paper's analysis experiments, now implemented and run
(pipeline3 jobs 1172/1174/1178/1184; scripts in `analysis/`). Exact paper settings
in [[paper_specs]]. All reproduce the paper's *qualitative* claims.

---

## A10 / Table 10 — compute overhead (job 1172)

| Method | Time (s), our RTX 6000 Ada | Paper (H100) |
|---|---|---|
| Cross-Entropy | 0.24 | 6.44 |
| UltraBreak (semantic) | 0.26 | 7.55 |
| **ratio semantic/CE** | **1.08×** | **1.17×** |

Absolute times differ by GPU (the reproducible quantity is the ratio). **Reproduces
the claim:** the semantic loss costs only ~8% more than CE — a negligible overhead
for the large ASR gain. Same ballpark as the paper's 1.17×.

### Independent corroboration from the real training runs (2026-10-10)

The table above is a micro-benchmark (job 1172: one row, forward+backward, 5 timed
iterations). A second, independent estimate comes from the **end-to-end training runs
themselves** — every run checkpoints every 20 epochs, so consecutive checkpoint mtimes
give a per-step wall-clock over the full 3000-epoch run, including optimiser, TV/L2 and
checkpointing. Median of 150 deltas each, same GPU, same 50-row corpus:

| run | s / epoch | ratio vs CE |
|---|---|---|
| CE (B13, job 1005) | 12.93 | 1.000 |
| no constraints (B12) | 13.77 | 1.065 |
| semantic, token mode (B14) | 13.84 | 1.071 |
| **semantic, attention, τ=0.5 (job 809 — the default)** | **14.32** | **1.107** |

**Two independent methods agree:** 1.08× (micro-benchmark) and **1.11×** (whole-run
wall-clock), against the paper's 1.17×. The overhead claim reproduces. The breakdown is
also informative — of the ~11% total, roughly 6.5 points come from the input-space
constraints (TV + random transforms) and the remaining ~4 from attention-mode semantic
loss over token-mode. Note "epoch" here is one pass over the 50-row corpus, so absolute
seconds are not comparable to the micro-benchmark's per-row figure; only the ratios are.

Data: `analysis/out/a10_overhead.csv` / `.json`.

---

## C2a / Fig 2a — model-size transfer (job 1184), SafeBench-315 v3

| Surrogate ↓ / Victim → | Qwen2.5-VL-3B | Qwen2.5-VL-7B |
|---|---|---|
| **Qwen2-VL-7B** | **62.5%** | 36.8% |
| **Qwen2-VL-2B** | 35.9% | 29.2% |

**Reproduces the paper's claim on both axes:**
- **Bigger surrogate transfers better:** 7B surrogate (62.5 / 36.8) > 2B surrogate (35.9 / 29.2).
- **Smaller victim is easier:** victim-3B (62.5 / 35.9) > victim-7B (36.8 / 29.2).

So "transferability improves as the surrogate grows or the victim shrinks" holds
cleanly in our runs. (32B victim dropped — won't fit a 49 GB GPU.)

---

## C4 / Fig 4-5 — loss landscape (job 1178)

Loss sampled on a grid along two random image-space directions (Fig 4: 100-epoch
image, 30×30; Fig 5: fully-trained, 200×200). PNGs in `analysis/out/fig{4,5}_*.png`.
Landscape **spread** (max−min, a flatness proxy) on the Fig 4 grid:

| Loss | spread |
|---|---|
| Cross-Entropy | 0.106 (roughest) |
| semantic τ=0 | 0.026 |
| semantic τ=0.5 | 0.016 |
| semantic τ→∞ | 0.008 (smoothest) |

**Reproduces the core claim:** the semantic loss gives a **much smoother** landscape
than CE (≈4–13× smaller spread), and **higher τ → smoother** still. This is the
mechanism the paper credits for better generalisation.

---

## C6 / Fig 6 — first-token probability distribution (job 1174)

First-token entropy at the first target position, per training temperature:

| τ | entropy (nats) |
|---|---|
| τ=0 | 1.288 |
| τ=0.5 | 1.629 |
| τ→∞ | 3.338 |

Bar charts in `analysis/out/ftok_*.png`. **Reproduces the expected behaviour:** τ
controls how peaked the target distribution is — **higher τ spreads probability
across more tokens** (entropy rises 1.3 → 3.3 nats), consistent with the attention
temperature relaxing the exact-match target.

---

## C3 / Figure 3 — do the input-space constraints really produce text-like patterns?

**Date: 2026-10-10.** The paper shows patches side by side and asserts qualitatively
that random transforms + TV yield *structured, text-like* patterns while unconstrained
optimisation yields noise. A contact sheet cannot settle that, so the figure pairs the
images with three measured texture statistics (numpy only — the pinned env gains no
packages): **TV/px** (local smoothness), **mid-band FFT power** in 8–32 cycles/image
(the scale glyph strokes live at), and **Sobel edge density**.

| patch | TV/px | mid-band | high-band | edge density |
|---|---|---|---|---|
| authors' `ultrabreak.png` | 0.0793 | 0.2247 | 0.7621 | 0.641 |
| **ours, job 809 (semantic + constraints)** | **0.0572** | **0.2926** | 0.6235 | 0.598 |
| TV 1.0 (B16) | 0.0596 | 0.2217 | 0.7027 | 0.611 |
| token mode (B14) | 0.0666 | 0.2474 | 0.6948 | 0.595 |
| TV 0.2 (B16) | 0.0805 | 0.2929 | 0.6446 | 0.693 |
| CE loss (B13) | 0.1335 | 0.2121 | 0.7373 | 0.862 |
| **no constraints (B12)** | **0.2485** | **0.0838** | 0.9109 | 0.930 |

**Reproduces, and now quantitatively.** Dropping TV *and* the random transforms (B12)
makes the patch **4.3× rougher** (TV 0.2485 vs 0.0572) and collapses mid-band energy to
**0.084 — a 3.5× drop** — with 91% of its power above 32 cycles/image and 93% of pixels
sitting on a strong edge. That is the signature of pixel noise, not structure. Every
constrained run is smoother *and* more mid-band-concentrated.

The images make the same point by eye: the constrained patches carry legible glyph-like
fragments (`Sure`, `Steps to`, `Mode`, `YES`, `TIPS`), while B12 is uniform RGB static.
This is the clearest visual evidence in the study that the constraints, not the semantic
loss alone, are what produce the paper's "text-like" patterns — note that **CE loss with
constraints still intermediate** (TV 0.134, edges 0.862): the loss choice matters less
for *texture* than the constraints do.

Figure: [`../../figures/fig3_patch_patterns.png`](../../figures/fig3_patch_patterns.png) ·
data `analysis/out/fig3_texture_stats.csv` · script `analysis/fig3_patterns.py`.

---

## Bottom line

All five analysis experiments **reproduce the paper's qualitative claims**: semantic
loss is nearly free (A10 — 1.08× and 1.11× by two independent methods vs the paper's
1.17×), it smooths the landscape and higher τ smooths it more (C4), τ spreads the
first-token mass (C6), transfer improves with a larger surrogate / smaller victim
(C2a), and the input-space constraints — not the loss — are what produce the
structured, text-like patterns (C3, measured: 4.3× smoother, 3.5× more mid-band
energy than the unconstrained ablation). These are mechanism confirmations, complementing
the ASR-level reproduction gap documented in [[../reproducibility/README]].
