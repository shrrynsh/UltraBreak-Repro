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

## Bottom line

All four analysis experiments **reproduce the paper's qualitative claims**: semantic
loss is nearly free (A10), it smooths the landscape and higher τ smooths it more
(C4), τ spreads the first-token mass (C6), and transfer improves with a larger
surrogate / smaller victim (C2a). These are mechanism confirmations, complementing
the ASR-level reproduction gap documented in [[../reproducibility/README]].
