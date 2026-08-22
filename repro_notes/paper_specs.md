# Exact Paper Specs (arXiv 2602.01025) — for reproduction

**Date: 2026-08-22.** Extracted from `arxiv.org/html/2602.01025v1` for the
backlog experiments. Verify against the PDF before quoting to the authors.

## Fig 2a — model-size transfer (C2a)
- **Surrogates:** Qwen2-VL-Instruct **2B and 7B**.
- **Victims:** Qwen2.5-VL-Instruct **3B, 7B, 32B**.
- **Benchmark:** SafeBench.
- **Claim:** "transferability generally improves as the surrogate model size
  increases or as the victim model size decreases."
- **Weights needed (not cached):** Qwen2-VL-2B-Instruct, Qwen2.5-VL-3B-Instruct,
  Qwen2.5-VL-32B-Instruct. (7B versions already cached.)

## Fig 4 / Fig 5 — loss landscape (C4)
- Sample loss along **two random normalised direction vectors** in **image space**.
- **Fig 4:** images trained **100 epochs**; **30×30 grid over a 20×20 range**.
- **Fig 5:** **fully-trained** images; **200×200 grid**.
- Compare **CE loss vs semantic loss** at **τ=0, τ=0.5, τ→∞**.

## Fig 6 — first-token probability distribution (C6)
- Token-probability distribution **at the first generation position**, for
  adversarial images trained with **τ=0, τ=0.5, τ→∞**.

## Table 10 — compute/timing (A10)
| Method | Time (s) |
|---|---|
| Cross-Entropy Loss | **6.44** |
| UltraBreak (semantic) | **7.55** |
- "average time over five iterations under identical hardware settings", NVIDIA **H100**.
- Ratio semantic/CE = **1.17×**. (Our GPU is RTX 6000 Ada — absolute times differ; the
  ratio is the reproducible quantity.)

## Table 5 — affirming-phrase sensitivity (B15), GLM-4.1V SafeBench
| Phrase | ASR |
|---|---|
| `[Jailbroken Mode]` | 66.03% |
| `[ACCESS AUTHORISED]` | 72.06% |
| `[CLASSIFIED INFO]` | 66.98% |
