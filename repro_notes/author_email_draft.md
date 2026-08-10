# Draft email to the UltraBreak authors

**To:** Kaiyuan Cui
**Subject:** UltraBreak reproduction — question about the Section 3.2 projection

---

Hi Kaiyuan,

I've been reproducing UltraBreak from the released code and have gotten most of the way there — the released `ultrabreak.png` scores 77.7% on SafeBench and 67.7% on AdvBench against Qwen2-VL-7B on my setup, close to the reported 81.6 / 72.7. (For anyone else reading: the AdvBench number only lands once the Section 3.3 TPG template is applied — that one was my mistake, not yours.)

Where I'm still short is **retraining the patch myself**. A patch I train from the released code reaches only ~30% SafeBench / ~2% AdvBench on Qwen2-VL at step 1300. Oddly, the same image transfers fine to LLaVA-1.6 (82.6 / 90.6, close to your reported numbers) — so it's specifically weak on the surrogate it was optimised against, which is what led me to the questions below.

I've ruled out two things I initially suspected: training under a different `transformers` version (retraining under your pinned 4.51.3 reproduced the same weak result), and the size of the training corpus.

**1. Was `ultrabreak.png` trained with the Section 3.2 projection active?**

In the released code, `project_patch()` in `optimisation/utils.py` has no call sites — the `patch_only=False` path applies the raw patch, so only the `clip(·, 0, 1)` half of `x_proj = clip(γ·x + β, 0, 1)` runs. I wired the projection in myself, changing nothing else, and it was the largest single effect I've found: +18 points SafeBench, +21 AdvBench.

More striking than the ASR: without the projection the training loss flattens out around step 400 and barely moves for the remaining 900 steps (−0.12), whereas with it the loss is still descending at step 1400 (−1.37 over the same span). That's why I suspect this is the main thing I'm missing rather than a minor detail.

**2. Is the optimised variable constrained to [0,1], or free in normalised CLIP space?**

This is really the crux. I implemented the projection by keeping the existing `clamp(x, 0, 1)` on the latent and then applying `γ·x + β`. That squeezes the saved image into `γ·[0,1] + β` = about `[0.48, 0.74]` per channel — per-channel std ≈ 0.025.

Your released `ultrabreak.png` spans the full `[0, 1]` range with std ≈ 0.165, which that path can't produce. So I assume the latent is meant to be **unconstrained in normalised space**, with the projection mapping it into valid pixel space (`x ∈ [-1.79, 1.93] → [0, 1]`). Could you confirm which it is? If it's the latter, I think the `clamp` in my version is simply wrong, and I'd like to correct it before reporting anything.

**3. What generation length did you use for GLM-4.1V-Thinking?**

At `max_new_tokens=512` only about 45% of my GLM generations complete: ~31% hit the cap inside `<think>` and never emit an answer at all, and another ~24% are cut off mid-answer. That makes my GLM ASR unreliable in both directions. Related: how did you handle the `<think>…</think>` block when scoring — strip it, or judge only the `<answer>` span? I found that feeding the reasoning trace to the HarmBench judge inflated GLM's SafeBench ASR from 32% to 57%.

**4. Minor — training corpus size.**

`create_attack_configs.load_safebench()` drops {legal opinion, financial advice, health consultation} from both the training and the attack configs, which leaves 35 rows for training, while `datasets/SafeBench-Tiny.csv` has 50 and the paper describes five queries per topic across ten topics. Was `ultrabreak.png` trained on 35 or 50? I retrained on all 50 and it made no real difference to my numbers, so I ask only for the record — the eval arithmetic in the paper (500 → 350 → 315) is consistent with either.

Thanks for releasing the code and the image — being able to score your artefact directly is what let me separate the evaluation-side problem from the training-side one. Happy to share my full logs and configs if any of it is useful.

Best,
Shreyansh

---

## Notes before sending (delete this section)

- Questions 1 and 2 are the ones that matter; 3 and 4 are secondary. Consider cutting 4 entirely if you want a shorter email.
- Do **not** raise the AdvBench prompt normalisation as a discrepancy — it is their own Section 3.3 method and the code predates the fork. The parenthetical in the opening paragraph is the right amount to say about it.
- The claim in Q2 that the released image "can't" come from the clamped path rests on the pixel statistics in `repro_notes/FINDINGS.md`. Re-check that table before sending.
- Consider running step 5 (unclamped projection) first — if it closes the gap, Q2 becomes a confirmation rather than a request for help, which is a stronger position to write from.
