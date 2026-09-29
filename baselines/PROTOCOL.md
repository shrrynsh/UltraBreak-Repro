# Table 1 Baselines — Protocol (verbatim from the paper)

The paper's exact baseline setup, to follow strictly:

> Baseline Attacks. We compare UltraBreak to representative methods from both handcrafted and
> optimisation-based attack families, including the typography-based **FigStep** (Gong et al., 2025) and
> the optimisation methods **VAJM** (Qi et al., 2024) and **UMK** (Wang et al., 2024c). For FigStep we use
> the authors' released typography images on SafeBench; because FigStep requires a distinct image
> per jailbreak target and therefore cannot produce a single universal trigger, we evaluate it only on
> SafeBench and exclude it from other benchmarks. For VAJM we follow the original protocol and
> optimise a jailbreak image on the derogatory corpus for 5,000 iterations. That optimised image then
> serves as the toxic, semantically embedded seed for UMK, which jointly optimises an adversarial
> image and a textual suffix; we train this stage for an additional 2,000 iterations as in the original
> work. To keep comparisons fair, all methods use SafeBench-Tiny as the few-shot corpus, consistent
> with UltraBreak.

## Operational reading

| Baseline | Training | Eval scope | Notes |
|---|---|---|---|
| **FigStep** | none | **SafeBench only**, 6 models | Use authors' released typography images; per-target images, no universal trigger, so excluded from AdvBench/MM-SafetyBench. |
| **VAJM** | optimise image on the **derogatory corpus, 5,000 iters** (original protocol) | all benchmarks × 6 models | Few-shot corpus = **SafeBench-Tiny** (fairness, same as UltraBreak). Surrogate = Qwen2-VL-7B (UltraBreak's surrogate). |
| **UMK** | seed = the VAJM image; jointly optimise image + text suffix, **+2,000 iters** | all benchmarks × 6 models | Same SafeBench-Tiny few-shot corpus. |

## Open point to resolve from the VAJM repo
"derogatory corpus" (VAJM's original optimisation target) vs "SafeBench-Tiny as the few-shot corpus"
(the paper's fairness constraint) — confirm from the official VAJM code exactly which corpus drives the
image optimisation and where SafeBench-Tiny plugs in.

## Repos (baselines/, git-ignored external code)
- `figstep/` — github.com/ThuCCSLab/FigStep (released typography images)
- `vajm/` — github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models
- `umk/` — github.com/roywang021/UMK
