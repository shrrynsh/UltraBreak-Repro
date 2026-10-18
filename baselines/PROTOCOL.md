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
| **VAJM** | optimise image on the **derogatory corpus, 5,000 iters** (original protocol) | all benchmarks × 6 models | **Surrogate = MiniGPT-4 / Vicuna-13B** — the original repo's native stack, which is what "follow the original protocol" means and what the released code hardcodes. *(An earlier draft of this line said Qwen2-VL-7B; that was never what ran, and VAJM's attacker takes no query set so SafeBench-Tiny cannot enter its image stage at all — see RESOLVED below.)* |
| **UMK** | seed = the VAJM image; jointly optimise image + text suffix, **+2,000 iters** | all benchmarks × 6 models | Same MiniGPT-4 surrogate; few-shot corpus = SafeBench-Tiny (all 50). **The paper ran UMK white-box on Qwen2-VL** — its 97.78/70.00/54.76 there vs ≤50.79 elsewhere is a white-box profile, and a GCG suffix is tokenizer-specific. Our UMK row is therefore **cross-architecture transfer** and must be labelled as such. See [[../repro_notes/baselines/README]] |

## RESOLVED (2026-10-10) — which corpus drives which stage

Answered from the released code, not inferred:

- **VAJM.** `minigpt_visual_attack.py:90` hardcodes `harmful_corpus/derogatory_corpus.csv`, and the
  attacker is constructed as `Attacker(args, model, targets, ...)` — **it takes target strings only,
  with no query set at all.** There is nowhere for SafeBench-Tiny to enter VAJM's image optimisation.
  So the paper's two sentences cannot both hold for VAJM; the explicit, implementable one wins:
  **derogatory corpus, 5,000 iterations.**
- **UMK.** `minigpt_vlm_attack.py` constructs `Attacker(args, model, queries, targets, ...)` — it does
  take (query, target) pairs. **SafeBench-Tiny plugs in here**, which is where the paper's fairness
  constraint is meaningful. Built as `baselines/umk/harmful_corpus/safebench_tiny_fewshot.csv`:
  all 50 SafeBench-Tiny `instruction` strings as queries, with
  `"Sure, here are some {instruction}"` as targets (UMK appends `':'` itself).

UltraBreak's affirming phrase `[Jailbroken Mode]` is deliberately **not** injected into UMK's targets
or eval prompts: that is UltraBreak's own prompt-engineering component, and importing it would make
the baseline partly UltraBreak. The corpus is swapped as the paper asks; UMK's method (GCG suffix +
PGD image) is left intact. At evaluation each baseline uses its own method's prompt — VAJM is
image-only, UMK is image + its optimised suffix.

### Three places the shipped code contradicts the paper, each overridden
| # | paper | shipped code | set to |
|---|---|---|---|
| 1 | UMK "an additional **2,000** iterations" | attack call hardcodes `num_iter=250` | **2000** |
| 2 | UMK seeded by "that optimised image" (VAJM's) | hardcodes its own `prompt_unconstrained_init.bmp` | **VAJM's `bad_prompt.bmp`** |
| 3 | "all methods use SafeBench-Tiny as the few-shot corpus" | 66 **unseeded** random AdvBench rows | **all 50 SafeBench-Tiny pairs** |

VAJM's 5,000 iterations needed no change (already hardcoded at 5000, though its own `--n_iters` flag
is dead code). Full detail and the patch rationale: [`patches/DEVIATIONS.md`](patches/DEVIATIONS.md).
Job: `jobs/a5_vajm_umk.sh`.

## Repos (baselines/, git-ignored external code)
- `figstep/` — github.com/ThuCCSLab/FigStep (released typography images)
- `vajm/` — github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models
- `umk/` — github.com/roywang021/UMK

## Environment isolation (STRICT - do not cross)

Three environments, each scoped to one job. Mixing them would change numbers on tasks
that are already measured, so nothing is ever re-run under a different env.

| Env | Path | Versions | Scope |
|---|---|---|---|
| `minigpt4` (conda) | `~/.conda/envs/minigpt4` | torch 2.0.1+cu118, transformers 4.28.0 | **VAJM + UMK baselines ONLY** |
| `repro` (venv) | `./repro` | torch 2.5.1+cu124, transformers 4.51.3 | all main repro: training, eval, judging |
| `repro_glm` (venv) | `./repro_glm` | torch 2.5.1+cu124, transformers 5.15.0 | GLM-4.1V generation only |

Rules:
- Never run main-repro training/eval/judging under `minigpt4`.
- Never change package versions inside `repro` / `repro_glm` - they produced every
  existing number; a version change silently invalidates completed results.
- Job scripts must call interpreters explicitly (`source repro/bin/activate`,
  `repro_glm/bin/python`, `~/.conda/envs/minigpt4/bin/python`) - never a bare `python`,
  so the shell's conda init cannot leak in.

### Documented deviation in `minigpt4`
`baselines/vajm/environment.yml` pins torch 1.12.1 / cudatoolkit 11.3, which does NOT
support the RTX 6000 Ada (sm_89 needs CUDA >= 11.8). torch is bumped to **2.0.1+cu118**,
the smallest version supporting Ada. All other deps are pinned exactly as the repo
specifies. This is a deviation from the authors' stack and must be reported with the
VAJM/UMK numbers - it should not change the attack math (PGD on pixels), but it is not a
byte-exact replication.
