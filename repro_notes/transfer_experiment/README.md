# Transfer Experiment — Review & Analysis

**Date: 2026-08-22.** Plain-language write-up of the black-box transfer study
(pipeline steps 1–3, jobs 880 → 935 → 939). All numbers are Attack Success Rate
(ASR %) under our v3 judge unless stated. "White-box" = the Qwen2-VL surrogate the
patch was trained on; "black-box / transfer" = the 5 other models the patch was
never trained on.

---

## 1. What question this experiment answers

The paper's headline method has a **Section 3.2 projection** step. The claim is
that this projection makes the adversarial image transfer better to unseen models.

We test that claim directly by attacking 5 unseen (black-box) models with **three
versions of the trained image**:

| Arm | What it is | White-box strength | Plain meaning |
|---|---|---|---|
| **projoff** | projection OFF (job 809, 3000 steps) | 83.49% | the strong patch, no projection |
| **projon** | projection ON (job 843, 3000 steps) | 42.54% | the paper's method, with projection |
| **control** | projection OFF, but a weaker checkpoint (step 1000) chosen to **match projon's white-box strength** | ~43% (target 42.5) | a fair yardstick |

**Why the control matters.** projon is much weaker at white-box (42.5% vs 83.5%),
so if it also transfers worse, that could just be because it is a *weaker* patch,
not because of the projection. The control is a projection-OFF patch **deliberately
weakened to the same white-box strength as projon**. So:

- **projon vs control** = the projection's *real* effect (same strength, only the
  projection differs).
- **projoff vs control** = what raw strength buys you.

---

## 2. The full results (all 30 cells complete)

### SafeBench ASR %
| Target | projoff | projon | control |
|---|---|---|---|
| Qwen2.5-VL-7B | 36.51 | 23.81 | 28.57 |
| Qwen-VL-Chat | 76.51 | 69.84 | 75.87 |
| LLaVA-1.6-mistral | 89.84 | 88.89 | 87.62 |
| Kimi-VL-A3B | 34.29 | **58.73** | 48.25 |
| GLM-4.1V-9B | 36.51 | 31.11 | 35.87 |

### AdvBench ASR %
| Target | projoff | projon | control |
|---|---|---|---|
| Qwen2.5-VL-7B | 4.81 | 1.54 | 2.12 |
| Qwen-VL-Chat | 58.65 | 57.88 | 69.81 |
| LLaVA-1.6-mistral | 94.23 | 91.73 | 89.23 |
| Kimi-VL-A3B | 20.58 | **25.38** | 24.04 |
| GLM-4.1V-9B | 7.88 | 5.00 | 7.31 |

**Read the projon column against control.** projon (projection ON) is the best of
the three arms on **only one model — Kimi-VL — and it wins there on both
benchmarks**. On the other four models, projoff and/or control beat projon.

---

## 3. The catch: some models barely refuse anything

Raw ASR is misleading, because some of these models are weakly aligned — they
answer harmful prompts even with **no attack at all**. So we ran a No-Attack
baseline (blank white image, no jailbreak phrase):

### No-Attack ASR % (nothing but the plain harmful question)
| Target | SafeBench | AdvBench |
|---|---|---|
| Qwen2.5-VL-7B | 18.10 | 0.77 |
| Qwen-VL-Chat | 7.30 | 3.46 |
| **LLaVA-1.6-mistral** | **57.14** | **48.65** |
| Kimi-VL-A3B | 33.02 | 3.46 |
| GLM-4.1V-9B | 12.70 | 0.96 |

**LLaVA already says yes to ~half of everything.** So LLaVA's headline 90–94%
transfer number is mostly LLaVA being LLaVA, not the attack working.

### The honest number: marginal lift = attack ASR − No-Attack ASR
This is "how much did the attack actually add?" (projoff arm):

| Target | SafeBench lift | AdvBench lift |
|---|---|---|
| Qwen-VL-Chat | **+69.2** | **+55.2** |
| LLaVA-1.6 | +32.7 | +45.6 |
| GLM-4.1V | +23.8 | +6.9 |
| Qwen2.5-VL | +18.4 | +4.0 |
| Kimi-VL | +1.3 | +17.1 |

**Takeaway:** the attack genuinely works (big lifts on Qwen-VL-Chat, LLaVA, and
GLM/SafeBench). But once you subtract the baseline, LLaVA is no longer the star,
and Kimi barely moves on SafeBench.

---

## 4. What we conclude

1. **The projection (projon) does NOT improve black-box transfer.** In 8 of 10
   model×benchmark cases, a projection-OFF patch (projoff or the matched control)
   transfers **as well or better** than projon. The projection's headline benefit
   does not reproduce here.
2. **Kimi-VL is the one exception** — projon beats control on both benchmarks.
   Worth flagging to the authors, not enough to rescue the general claim.
3. **projoff ≈ control on most targets** — meaning at matched strength the
   projection is roughly neutral; the extra raw strength of the full 3000-step
   projoff patch is what actually helps, not the projection.
4. **Always subtract the No-Attack baseline.** LLaVA's raw 90%+ is inflated by its
   own weak alignment; the attack's true contribution is more modest.

---

## 5. How it was produced (for reproducibility)

- **Surrogate (white-box):** Qwen/Qwen2-VL-7B-Instruct.
- **Targets (black-box):** Qwen2.5-VL, Qwen-VL-Chat, LLaVA-1.6-mistral, Kimi-VL,
  GLM-4.1V. GLM runs in a separate env (`repro_glm`, transformers 5.x) because its
  model class needs a newer library; everything else uses the pinned `repro` env.
- **Driver:** `jobs/pipeline.sh` steps 2 (SafeBench) and 3 (AdvBench), each scoring
  every arm on all 5 targets, then the v3 HarmBench judge.
- **Pipeline hardening applied** (see [[repro_transfer_pipeline]] memory): config
  builds are now checked for emptiness, GLM is env-routed, and a resume guard
  reuses already-computed cells. These fixed the earlier all-`nan%` failure.

**Two-line summary:** *The attack transfers, but the paper's projection step is not
what makes it transfer — a projection-OFF patch does just as well or better on 4 of
5 black-box models, with Kimi-VL the lone exception.*
