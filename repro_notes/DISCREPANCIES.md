# UltraBreak — code vs. paper discrepancies

**Scope.** Every point where the released implementation (authors' commit `c4c276d`) diverges from *Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models* (ICLR 2026, arXiv 2602.01025). Companion to [`FINDINGS.md`](FINDINGS.md), which covers the reproduction gap itself; this file is the audit trail behind it.

**Status as of 2026-08-10.**

**Sourcing caveat.** Paper quotes below were extracted from the arXiv HTML (`arxiv.org/html/2602.01025v1`). Spot-check them against the PDF before quoting any of them to the authors.

**Severity key.**

| | meaning |
|---|---|
| **Material** | plausibly affects reproduced numbers |
| **Documentation** | paper text and code disagree; code is demonstrably the operative version |
| **Minor** | real but immaterial to results |
| **Ours** | defect in *this* reproduction, not in the authors' release |

---

## Summary

| ID | Discrepancy | Severity | Action |
|---|---|---|---|
| D1 | Section 3.2 projection is never applied — `project_patch()` is dead code | **Material** | Ask authors (email Q1) |
| D2 | Training corpus is 35 queries; paper's arithmetic implies 50 | **Material** | Ask authors (email Q4); fixed in our fork |
| D3 | TV loss: Eq. 9 is isotropic+summed, code is anisotropic+averaged | Documentation | Note to authors; **do not change code** |
| D4 | γ, β typed as scalars in ℝ but assigned per-channel 3-vectors | Minor | Note to authors |
| D5 | Patch location bound is dynamic, not Table 4's fixed `l_max = 112` | Minor | No change |
| D6 | Paper says AdvBench = 500; its own numbers are all `n/520` | Documentation | Erratum note; our 520-row eval is correct |
| D7 | No random seed is set anywhere | Minor (reproducibility) | **Fixed in this fork** (`--seed`); note in write-up |
| D9 | Some of our eval configs kept the 35 SafeBench-Tiny training queries | **Ours** | **Fixed** — summariser filters to the canonical 315 |
| D8 | Undocumented L2 term in code, weight 0.0 | Minor | No change |
| R1 | AdvBench prompt normalisation — **investigated, not a discrepancy** | — | **Do not report** |
| O1 | Our projection double-constrains the latent | **Ours** | Fix before next run |
| O2 | Our latent init is no longer a random image | **Ours** | Fix before next run |

---

## D1 — The Section 3.2 projection is never applied · **Material**

**Paper, Section 3.2 (*Projection onto Constrained Input Space*):**

> The image input is projected onto a constrained subspace to reduce reliance on extreme pixel values during training. We apply an affine projection that shifts and scales pixel values: x_proj = clip(γ·x + β, 0, 1), where γ ∈ ℝ and β ∈ ℝ are fixed scalar scale and shift parameters.

**Paper, Eq. 10** feeds the *projected* patch to the surrogate: `A(x_blank, x_proj, l, r, s)`.

**Paper, appendix:**

> For the affine projection step, we used CLIP's normalisation statistics, γ=CLIP_STD and β=CLIP_MEAN, and clipped pixel values back to [0,1].

**Code.** `project_patch()` has **zero call sites** in the authors' release. The `patch_only=False` path passes the raw patch straight to the model, and the only range constraint is `adv_patch.data = torch.clamp(adv_patch.data, 0, 1)`. That is the projection with γ=1, β=0 — the affine collapses to identity and only the clip survives.

Verify against the authors' commit directly, not the working tree — this fork has since modified these files:

```
$ git grep -n "project_patch" c4c276d -- optimisation/ | grep -v "def project_patch" | wc -l
0
```

Our own changes to `optimisation/` are CLI plumbing and ablation flags whose defaults reproduce Table 4; `optimise.py`'s diff against `c4c276d` touches only argument parsing, never the objective, the transforms or the clamp.

**Impact.** This is the leading candidate cause of Defect B. Wiring the projection in (`optimise_proj.py`, job 625 vs job 622) moved SafeBench +18.4 and AdvBench +20.6, and — the stronger evidence — changed the loss trajectory from stalled to still-descending at step 1400. See `FINDINGS.md` § "Implicated: the omitted Section 3.2 projection".

**Confidence.** The dead-code fact is certain (verified against commit `c4c276d`, as above). Whether `ultrabreak.png` itself was trained with the projection is **unknown** — that is email Q1.

**Note (2026-08-10).** The released artifact now reproduces the paper's SafeBench number *exactly* (257/315 = 81.59%) once evaluated on the canonical 315 set with the authors' own judge. So D1 is a training-side discrepancy only: it does not affect whether the released image scores as reported, just whether the released code can produce such an image.

---

## D2 — Training corpus is 35 queries where the paper implies 50 · **Material**

**Paper:**

> SafeBench provides 500 harmful queries organised into 10 topical categories.

> We exclude three SafeBench topics—legal opinion, financial advice, and health consultation—whose harmfulness may be model-dependent.

> We also remove the SafeBench-Tiny subset (used to train UltraBreak), yielding a final held-out evaluation set of 315 harmful queries.

SafeBench-Tiny is described as *"five harmful queries per topic"* — 5 × 10 topics = **50**, and `datasets/SafeBench-Tiny.csv` indeed has 50 rows.

**The paper's own arithmetic confirms the exclusion is an evaluation step, not a training one:**

```
500  total
-150  drop 3 topics (3 × 50)
= 350
- 35  remove the SafeBench-Tiny queries that survive the topic filter (7 topics × 5)
= 315  ✓ matches the paper
```

That reconciles only if SafeBench-Tiny is 50 queries *and* the topic exclusion applies to the evaluation set. Training is stated to run *"on SafeBench-Tiny"* — all 50.

**Code.** `create_attack_configs.load_safebench()` applied the three-category exclusion when building the **training** config as well, leaving 35 rows.

**Impact.** Real as a faithfulness question, but **ruled out as the cause of Defect B** — job 622 corrected 35→50 (together with the transformers pin) and produced 29.84% / 1.73% against the original 27.4% / 7.69%. See `FINDINGS.md` § "Ruled out".

**Status in this fork.** Fixed. `load_safebench()` now takes `exclude_categories` (`create_attack_configs.py:108`), driven by `--keep-all-categories`, so training configs retain all 50.

---

## D3 — TV loss: Eq. 9 disagrees with the implementation · Documentation

**Paper, Eq. 9** — isotropic, summed:

```
L_TV(x) = Σ_{i=1}^{H-1} Σ_{j=1}^{W-1} √( (x_{i+1,j} − x_{i,j})² + (x_{i,j+1} − x_{i,j})² )
```

**Code** (`optimisation/optimise.py:57`, identical in `optimise_proj.py:64`) — anisotropic, averaged:

```python
h_variation = torch.abs(img[:,:,1:,:] - img[:,:,:-1,:]).mean()
w_variation = torch.abs(img[:,:,:,1:] - img[:,:,:,:-1]).mean()
return h_variation + w_variation
```

Two differences: the directions are combined by L1 (`|dh| + |dw|`) rather than L2 (`√(dh² + dw²)`), and the reduction is a mean rather than a sum. The second dominates numerically:

| image | code (aniso, mean) | Eq. 9 (iso, sum) | ratio |
|---|---|---|---|
| `ultrabreak.png` | 0.186 | 22,538 | ~121,000× |
| random init | 0.667 | 77,285 | ~116,000× |

**The code is demonstrably the operative definition.** Table 4 gives `λ_TV = 0.5`. Against a text loss of ~12:

| | TV contribution |
|---|---|
| code form | 0.093 |
| Eq. 9 as written | 11,269 |

Eq. 9 with λ_TV = 0.5 would swamp the semantic objective ~1000×, driving the patch to a flat grey square with no adversarial pattern at all. Eq. 9 as printed is therefore **arithmetically incompatible with the paper's own hyperparameter table**, and the mean-reduced anisotropic form must be what produced the reported results.

**Action. Do not change the code.** It is what the authors ran and what λ_TV = 0.5 is calibrated to. Worth an erratum-style line to the authors.

**Untested speculation, excluded from the write-up:** anisotropic L1 TV biases toward axis-aligned structure, which is suggestive given the paper's claim that transferability comes from "structured, text-like adversarial patterns". No ablation was run; do not assert this.

---

## D4 — γ and β typed as scalars but assigned per-channel vectors · Minor

Section 3.2 says *"γ ∈ ℝ and β ∈ ℝ are fixed **scalar** scale and shift parameters"*, but the appendix sets them to `CLIP_STD` and `CLIP_MEAN`, which are per-channel 3-vectors:

```
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
```

Immaterial to results — `project_patch()` handles both — but it means Section 3.2 read alone does not tell you the projection is per-channel. Worth one clause to the authors.

---

## D5 — Patch location bound is dynamic, not Table 4's `l_max = 112` · Minor

**Table 4** gives `l_min = 0`, `l_max = 112`, which is exactly `336 − 224` for a 224 patch on a 336 canvas. Geometry confirmed: `base_adapter.py:3` sets `image_size=(336,336)`, `patch_size=(224,224)`.

**Code.** `utils.py:84` samples `top = randint(0, H − tph)` where `tph` is the bounding box *after* rotation and scale (`utils.py:52-53`), not the raw 224. Measured over 200,000 samples of the actual formula:

| | value |
|---|---|
| `l_max` across samples | **6 – 156** (mean 84.8) |
| `l_max` at identity (scale 1.0, 0°) | **111** |
| Table 4 nominal | 112 |
| overflow → `ValueError` | 0 / 200,000 |

The identity case is off by one (`int(...) + 1` in the bounding-box estimate). The coupling to scale/rotation is unavoidable — a 1.2×-scaled 15°-rotated patch is 330px wide and leaves only 6px of slack, so a fixed `l_max = 112` would push it off-canvas.

**Verdict: consistent in intent.** Section 3.2 only specifies `l ∼ U(l_min, l_max)`; Table 4's 112 is the nominal untransformed value. **No change** — a literal 112 would invent a placement strategy the paper doesn't specify. Note: the overflow guard at `utils.py:82` is dead in practice.

---

## D6 — AdvBench row count: the paper says 500, its numbers say 520 · Documentation

Paper: *"AdvBench comprises 500 harmful textual instructions."* `datasets/adv_bench.csv` has **520** rows — the standard AdvBench `harmful_behaviors` set, which contains near-duplicates.

**Resolved 2026-08-10: the reported numbers were computed on 520.** An ASR printed to two decimals is a rounded `n/N`, so N is recoverable. `evaluation/infer_eval_set_size.py` tests each candidate against all 12 reported AdvBench values (UltraBreak + No-Attack columns):

| N | fit | meaning |
|---|---|---|
| **520** | **12/12** (p ≈ 4e-16 by chance) | the row count of `adv_bench.csv` |
| 500 | 2/12 | the count stated in the paper |

72.69% is exactly 378/520; with 500 rows it would require 363.45 successes, which is not an integer. Same method confirms SafeBench = **315** (12/12; 350 fits 2/12) and MM-SafetyBench = **1680** (12/12) — and the MM-SafetyBench text-only set does contain exactly 1680 questions, independently confirming it.

**Consequence: our 520-row AdvBench evaluation is correct, not a deviation.** The discrepancy is in the paper's prose, not its arithmetic. Worth one clause to the authors as an erratum. No code change.

---

## D9 — Our evaluation set kept the training queries · **Ours**

Several of this fork's attack configs were built without `--exclude-train`, giving **350 rows** where the paper's SafeBench evaluation set is **315** (500 − 150 excluded topics − 35 SafeBench-Tiny queries that survive the topic filter). The reference number 77.71% was measured on that contaminated set.

Corrected, the same generations give **78.73%** (v2) and **81.59%** (v1). The patch scored *worse* on its own training queries (25/35 = 71.43%) than on held-out ones, so the contamination was mildly deflating.

**Fixed:** `evaluation/summarise_asr.py` now derives the canonical 315 set from `datasets/` (not from a generated config, which could drift with whatever flags it was built with) and filters every SafeBench run to it, reporting what it dropped. `--raw` disables it.

---

## D7 — No random seed · Minor (reproducibility) · **fixed in this fork**

`grep` for `manual_seed` / `random.seed` / `np.random.seed` across `optimisation/` returns **nothing**. Training samples random transformation parameters every step (`utils.py:37-39`), initialises the patch randomly, and adds Gaussian noise to target embeddings (`utils.py:185`) — all unseeded. The paper does not mention seeding.

**Consequence for our own claims:** run-to-run variance is unquantified, so the +18.4 / +20.6 projection deltas rest on a single run pair (n = 1, no seed control). Already carried as a caveat in `FINDINGS.md`; repeated here because it limits every number in this study, not just that one.

**Fixed in this fork.** `optimise_proj.py --seed` seeds `random`, `numpy` and `torch` (CPU + CUDA), covering all three sources; `optimisation/test_reproducibility.py` asserts the seed reaches each of them. One limit to state honestly in the write-up: this pins the *sampling*, not the arithmetic — `grid_sample`'s backward pass has no deterministic CUDA kernel, and every training step goes through it, so two seeded GPU runs still drift slightly. Seeding makes an n>1 variance study possible; it does not make training bitwise reproducible.

---

## D8 — Undocumented L2 term · Minor

`optimise.py:126` computes `l2_loss = torch.mean((adv_patch − 0.5) ** 2)`, a pull toward mid-grey. Paper Eq. 10 has no such term. It is **inert** — `l2_weight = 0.0` (`optimise.py:109`, `optimise_proj.py:121`) — so the objective as executed matches Eq. 10. No change.

*(Note for the unclamped run: if this term is ever enabled, `0.5` is the neutral point in **pixel** space. In normalised latent space the neutral point is `0.0`, so the constant would need changing.)*

---

## R1 — AdvBench prompt normalisation — **not a discrepancy** · do not report

Recorded here so it is not re-raised. Scoring the authors' released image on AdvBench gave 9.62% against a reported 72.69%; normalising prompts to the paper's Section 3.3 TPG template lifted it to 67.69%.

**This was our omission, not a defect in the paper or code.** The template is defined by equation in Section 3.3, and both `normalize_instruction()` and the README `--normalize` flag were authored by the first author pre-fork. See `FINDINGS.md` § "Defect A".

---

## O1 — Our projection double-constrains the latent · **Ours**

`optimisation/optimise_proj.py:153` retains the original `clamp(adv_patch, 0, 1)` on the raw latent *and then* applies `γ·x + β`. The paper applies exactly one constraint, and it comes **after** the affine:

```
paper:  x_proj = clip(γ·x + β, 0, 1)          x unconstrained
ours:   x      = clamp(x, 0, 1)               ← extra, upstream
        x_proj = clip(γ·x + β, 0, 1)          ← correct, but fed clamped input
```

Because the pre-clamp guarantees `γ·x + β ⊂ [0.41, 0.75]` for every channel, **the paper's clip can never bind** — we implemented the projection faithfully and then starved it of the input range it exists to handle. Since the appendix says pixel values were *"clipped back to [0,1]"*, that clip must be reachable, which requires an unconstrained latent.

Measured consequence:

| | R min | R max | R std |
|---|---|---|---|
| authors' `ultrabreak.png` | 0.000 | 1.000 | 0.165 |
| our projected retrain | 0.478 | 0.741 | 0.025 |

γ·x + β with γ=CLIP_STD, β=CLIP_MEAN is the exact inverse of CLIP normalisation, so the latent lives in normalised space and the projection de-normalises it into pixel space — `x ∈ [−1.79, 1.93] → [0, 1]`, reproducing the authors' full-range image.

**Fix:** delete `optimise_proj.py:151-153`. Leave `project_to_constrained_space()` alone; it is already correct.

**Confidence: strong inference, not a quote.** The paper never says "do not clamp the latent". The argument is the dead-clip one above.

---

## O2 — Our latent init is no longer a random image · **Ours**

**Paper, appendix:** *"The final jailbreak image was optimised with Adam from a random initialisation for 1,300 steps on SafeBench-Tiny."*

In the unprojected code `adv_patch` **is** the image, so `torch.rand(3,224,224)` means "a random image" — correct. Once the projection is active, `adv_patch` is a *latent* and the image is `γ·x + β`, so the same line now produces a washed-out mid-grey square:

| init | latent range | resulting image | image std |
|---|---|---|---|
| `torch.rand(...)` (current) | [0.00, 1.00] | [0.408, 0.750] | 0.083 |
| `(torch.rand(...) − MEAN)/STD` | [−1.79, 2.15] | [0.000, 1.000] | 0.289 |

The run would start inside the very band O1 frees it from, with 3.5× too little contrast. At `lr = 0.01`, walking a latent value from ~0.5 out to 1.93 (white) takes ~143 steps and to −1.79 (black) ~229 — 10–18% of the 1300-step budget spent in transit.

**Fix:** `adv_patch = (torch.rand(3, patch_size, patch_size, device=device) - mean) / std` — the projection run backwards, restoring the released code's original starting distribution.

**Confidence: interpretation.** The paper says "random initialisation" without naming the space. This is the weakest-supported of our two changes and belongs in the email as a sub-question to Q2.

---

## Verified faithful — no discrepancy found

Full Table 4 audit; every row checked against the code.

| Table 4 parameter | Value | Code | |
|---|---|---|---|
| Scale min/max | 0.8 / 1.2 | `utils.py:19` | ✓ |
| Rotation min/max | −15° / +15° | `utils.py:19` | ✓ |
| Location min/max | 0 / 112 | `utils.py:84` | see D5 |
| TV weight λ_TV | 0.5 | `optimise_proj.py:120` | ✓ |
| Embedding noise ε | 1e-4 | `utils.py:184` | ✓ |
| Attention temperature τ | 0.5 | `utils.py:188` | ✓ |
| Affirming phrase p | `[Jailbroken Mode]` | train config | ✓ |
| Learning rate η | 0.01 | `optimise_proj.py:113` | ✓ |
| Optimiser | Adam | `optimise_proj.py:114` | ✓ |
| Iterations | 1300 | `CKPT_STEP=1300` | ✓ |
| Image size | 224×224 | `initialise_patch(..., 224, ...)` | ✓ |

Also faithful: TV is applied to the raw variable `x`, matching Eq. 10's `λ_TV·L_TV(x)`; the projected patch is what gets saved as the deployable image (`optimise_proj.py:161`, `:196`).

**Watch item, not a discrepancy.** Once O1 lands and the latent spans ~3.7× wider, `L_TV(x)` grows correspondingly — the TV term moves from ~0.33 to ~1.24 against a text loss of ~12. That *is* what the paper specifies (Eq. 10 applies TV to raw `x`, Table 4 fixes λ_TV = 0.5), so λ_TV stays at 0.5. But check `losses.csv`: if TV starts dominating, suspect this first.
