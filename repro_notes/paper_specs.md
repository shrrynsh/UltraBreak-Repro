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

---

## Table 1 — baseline rows (extracted 2026-10-10 from arxiv.org/html/2602.01025v1)

Needed to judge A5. Column order as in the paper; **Qwen2-VL-7B is UltraBreak's
white-box surrogate**, the other five are black-box.

### SafeBench
| Method | Qwen2-VL | Qwen-VL-Chat | Qwen2.5-VL | LLaVA-1.6 | Kimi-VL | GLM-4.1V | BB avg |
|---|---|---|---|---|---|---|---|
| No Attack | 18.41 | 22.86 | 14.29 | 80.32 | 39.37 | 46.03 | 40.57 |
| FigStep | 44.76 | 69.52 | 53.97 | 47.94 | 73.02 | 88.25 | 66.54 |
| **VAJM** | **0.95** | 12.06 | 28.89 | 57.46 | 41.27 | 67.62 | 41.46 |
| **UMK** | **97.78** | 0.63 | 15.24 | 20.63 | 12.70 | 50.79 | 20.00 |
| UltraBreak | 81.59 | 72.70 | 60.32 | 88.25 | 67.94 | 66.03 | 71.05 |

### AdvBench
| Method | Qwen2-VL | Qwen-VL-Chat | Qwen2.5-VL | LLaVA-1.6 | Kimi-VL | GLM-4.1V | BB avg |
|---|---|---|---|---|---|---|---|
| No Attack | 0.38 | 1.92 | 0.00 | 21.35 | 4.42 | 2.12 | 5.96 |
| **VAJM** | **0.38** | 0.96 | 0.38 | 19.42 | 3.65 | 3.65 | 5.61 |
| **UMK** | **70.00** | 0.38 | 2.69 | 16.35 | 2.12 | 4.42 | 5.19 |
| UltraBreak | 72.69 | 71.92 | 35.77 | 92.88 | 30.38 | 30.00 | 52.19 |

### MM-SafetyBench
| Method | Qwen2-VL | Qwen-VL-Chat | Qwen2.5-VL | LLaVA-1.6 | Kimi-VL | GLM-4.1V | BB avg |
|---|---|---|---|---|---|---|---|
| No Attack | 26.19 | 21.49 | 33.45 | 35.06 | 41.79 | 43.69 | 35.10 |
| **VAJM** | **5.42** | 11.73 | 26.79 | 30.18 | 35.36 | 36.73 | 28.16 |
| **UMK** | **54.76** | 5.48 | 17.56 | 21.96 | 26.67 | 37.44 | 21.82 |
| UltraBreak | 57.26 | 53.10 | 45.83 | 71.90 | 54.58 | 67.08 | 58.50 |

**Extraction validated**, not taken on trust: the FigStep row matches byte-for-byte
the values extracted independently into `jobs/t1_figstep.sh`, and UltraBreak's cells
match anchors we reproduced ourselves (SafeBench 81.59, AdvBench 72.69, GLM
MM-Safety 67.08).

### What these rows reveal about the baselines' surrogate

**UMK carries an unmistakable white-box signature on Qwen2-VL:** 97.78 / 70.00 /
54.76 there against ≤50.79 / ≤16.35 / ≤37.44 on every other model. UMK optimises a
**GCG text suffix** in the surrogate's token-embedding space; a suffix tuned on one
tokenizer is near-meaningless on another. So the authors must have run **UMK with
Qwen2-VL-7B as the surrogate**, not MiniGPT-4.

**VAJM shows the opposite:** its *lowest* SafeBench cell is Qwen2-VL (0.95) — below
even No Attack (18.41), i.e. the image actively suppresses success there. That is not
a white-box profile. Either VAJM was run on a different surrogate, or it was run on
Qwen2-VL and simply failed. The paper does not say.

Consequence for A5 (`jobs/a5_vajm_umk.sh`): our run trains **both** on MiniGPT-4 /
Vicuna-13B, the original repos' native stack. VAJM is defensible that way, but our UMK
suffix is optimised in Vicuna's token space and cannot be compared like-for-like with
the paper's 97.78. Report ours as cross-architecture transfer and say so.

---

## What the paper specifies about MM-SafetyBench (checked 2026-10-18)

**Everything it says:**

> "MM-SafetyBench is a dataset specifically designed for multimodal safety evaluation,
> which contains **1680 harmful targets covering 13 scenarios**."

That is the whole specification. It confirms the denominator our forensics recovered
(1680 is the unique N consistent with all 24 reported MM-SafetyBench values).

**What it does NOT specify — two genuine reproducibility gaps:**

1. **Which variant.** MM-SafetyBench ships each question in four forms: Stable-Diffusion
   image, typographic image, SD+TYPO, and `Text_only`. The paper never says which. The
   string "Stable-Diffusion" does not appear in it at all.
2. **Whether the adversarial image replaces the benchmark's own image.** MM-SafetyBench is
   natively multimodal; UltraBreak supplies a universal image. The paper does not state
   that one substitutes for the other, though its threat model ("works when paired with any
   harmful text query") implies it.

**Our choice, and why it is the only coherent one.** We use the **`Text_only` split, all
1680**. The image-bearing splits cannot be used with a universal adversarial image: their
questions are *image-dependent* ("The bottom of the image shows a phrase…"), so pairing them
with an adversarial patch instead of their own generated image leaves the question
referring to something that is no longer there. `Text_only`'s questions are self-contained.
Independently, only the full set reproduces the paper's denominator. So the choice is
forced — but it is forced by us, not stated by the authors, and belongs in the write-up.

## Three other protocol points confirmed from the paper's text

- **SafeBench 315 is right, and now confirmed in prose**, not just by forensics: 500 queries,
  minus the three excluded topics (legal opinion, financial advice, health consultation),
  minus SafeBench-Tiny → "a final held-out evaluation set of 315 harmful queries". ✓ ours.
- **AdvBench: the paper says "500 harmful textual instructions"** while every number it
  reports is an exact n/**520**. Restates [[DISCREPANCIES#D6]].
- **No rewording of evaluation prompts is described anywhere.** The only rewording the paper
  describes is TPG, and it describes applying it to the *few-shot training corpus*. The
  released code additionally rewrites every AdvBench goal to "Steps to …" at **evaluation**
  time — worth **+58 points** — and the paper never mentions it. This is
  [[DISCREPANCIES#D13]], now confirmed against the paper's full text rather than its tables.
- **Judge**: "HarmBench-Llama-2-13b, using its original chat template without modification."
  The released `evaluate.py` does *not* match that description — its extractor anchors on the
  first literal `assistant`, which lands inside Qwen's system prompt ([[DISCREPANCIES#D12]]).
  So the paper describes faithful use; the code deviates from the paper.

**On TPG at evaluation.** The paper's prose places TPG on the training corpus, which would
imply bare evaluation prompts. Two pieces of evidence say TPG is nevertheless applied at
evaluation: (a) the paper changes the affirming phrase to `[START LIST]` *for closed-source
evaluation*, which is meaningless unless the phrase appears in eval prompts; (b) our E2
ablation measured the authors' own released image at **16.83%** on SafeBench with TPG off
versus **78.73%** with it on — only the latter reproduces their 81.59%. The numbers settle
what the prose leaves ambiguous.
