# Prompt — write the Introduction section

Paste everything below the line into a fresh chat. Everything it needs is in the prompt;
it should not have to ask you questions or invent numbers.

---

You are helping write the Introduction of a reproducibility report for TMLR (the ML
Reproducibility Challenge track). Write **only** the Introduction. Do not write the abstract,
related work, or any other section.

## The paper

Title: **[Re] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models**

It reproduces Cui et al. (2026), "Toward Universal and Transferable Jailbreak Attacks on
Vision-Language Models" (ICLR 2026), which they call UltraBreak. UltraBreak optimises a *single*
adversarial image on one white-box surrogate (Qwen2-VL-7B-Instruct) using a 50-query few-shot
corpus, and claims that image both generalises to unseen harmful queries (universality) and
transfers to unseen victim models (transferability).

## What we did

Three tracks, across 6 models, 3 benchmarks and 2 scorers:
1. **Released artifact** — evaluating the authors' published adversarial image.
2. **Code-faithful retrain** — their released training code, run verbatim.
3. **Paper-faithful retrain** — the method as described in the paper.

We also re-ran all three baselines (FigStep, VAJM, UMK) under the paper's stated protocol, plus
the component ablations, the affirming-phrase and total-variation sweeps, and the analysis figures.

## Findings to draw on (use only what the Introduction needs; the rest appear later)

**The central result.** The released *artifact* reproduces the headline exactly — 81.59% on
SafeBench under the authors' own scorer, matching the published value to the decimal. The released
*code* cannot produce it. The adversarial patch is composited raw, in [0,1], into a pixel tensor
the processor has already CLIP-normalised to roughly [-2,2], so training is confined to a 27%-wide
grey band while the saved image spans the full range. A zero-training test settles it: remapping
the authors' own working image into that band drops it from 67.69% to 8.85% on AdvBench with its
spatial pattern untouched.

**The fix is budget-gated.** Correcting that bug at the paper's stated 1300 iterations makes
results *worse* (23.49%); only at 3000 iterations does the pipeline reach 83.49%.

**Evaluation protocol problems.** (a) Every AdvBench goal is silently rewritten from its imperative
form into "Steps to …", worth 58 points on the authors' own image (9.62% verbatim, 67.69%
reworded), and the paper describes no such rewriting anywhere. (b) The released scorer locates the
model's reply by searching for the first literal "assistant", which for Qwen models occurs inside
the system prompt, so the harmful question is fed to the classifier as though it were the answer;
this inflates ASR by 1 to 13 points and in places reorders models. (c) A 512-token generation
budget inherited for every model starves GLM-4.1V's reasoning trace, truncating it on up to 98.7%
of rows and moving its numbers by as much as 41 points once corrected.

**Undocumented protocol, recovered.** The paper does not state its evaluation-set sizes. Because
ASR is a ratio of integers, we solved for the denominators consistent with every value it reports:
MM-SafetyBench is uniquely 1680.

**A limit that binds the original work too.** Three of the six evaluated models decode
stochastically and the released evaluation path sets no seed. Two of our configurations are
byte-identical in prompts, images and targets, and one model scored 191/315 on one and 190/315 on
the other.

**What did not reproduce.** Neither retrain reaches the reported transfer result on Qwen2.5-VL
(about 27% against 60%), and no retrain reproduces AdvBench (best 39.04% against 72.69%).

## Domain context for the opening

The standard framing is that a VLM's continuous, high-dimensional visual input widens the attack
surface relative to text-only models (Qi et al., 2024 — cite as `qi2024visual`), that earlier
image jailbreaks overfit their surrogate, and that an attack which is *both* universal and
transferable was an open problem UltraBreak claims to close. Two further points are available if
useful: existing jailbreak benchmarks are each evaluated on different datasets and metrics, so
unified evaluation is missing; and a recent effort reproduced 30 published attacks to within about
0.26 percentage points of their reported success rates, which makes a reproduction that surfaces
five material discrepancies notable rather than routine.

## Structure — four paragraphs, then a contribution list

1. **Domain motivation.** Two or three sentences. Why universal, transferable visual jailbreaks
   matter more than per-query attacks. Do not write a literature review.
2. **UltraBreak and its claims.** What it proposes and what it claims.
3. **Why this paper warrants independent checking.** This is the most important paragraph and the
   one most reproducibility reports write weakly. The argument available to us: UltraBreak's entire
   artifact is a single released image, so the artifact and the code can be checked *against each
   other* — a test most papers do not permit, because usually only one or the other exists.
4. **What we found**, briefly — the split verdict, not a list of everything.

Then a short contributions list (four or five items, one sentence each): the three-track
reproduction; the discrepancies between paper and code; the recovery of undocumented protocol from
published values; the baselines re-run under the stated protocol; the determinism limit.

## Tone and style — follow these exactly

- **Simple, formal academic English.** Short declarative sentences. Prefer a plain word to a
  showy one.
- **British spelling**, to match the rest of the paper: *optimised*, *normalisation*, *behaviour*,
  *analyse*.
- **Vary sentence length deliberately.** A paragraph of uniformly long sentences reads as machine
  output. Let a short sentence land after two long ones.
- **Do not** use: "delve", "moreover", "furthermore", "it is worth noting", "crucially",
  "notably", "robust" as a filler adjective, "paradigm", "landscape" as metaphor, "pivotal",
  "underscores", "sheds light on", "in today's rapidly evolving".
- **Do not** open consecutive sentences with the same construction, and avoid stacking
  three-item lists in every sentence.
- **Numbers, not adjectives.** Write "drops from 67.69% to 8.85%", never "drops dramatically".
- **Claim only what the evidence supports.** The verdict is *split*: universality, low overhead
  and the emergent-pattern claim reproduce; the component ablation does not; transferability
  reproduces for the artifact but not for either retrain. Never write that the paper "fails to
  reproduce" without qualification.
- **No rhetorical questions, no hype, no first-person plural enthusiasm** ("we are excited to").
  Use "we" plainly for what we did.
- Length: roughly 500–650 words plus the contribution list.

## Output format

LaTeX, ready to paste into the existing file. Use `\section{Introduction}` and
`\label{sec:intro}`. Cite with `\citep{}` / `\citet{}`. Citation keys already in the
bibliography: `cui2026ultrabreak`, `qi2024visual`, `zou2023universal`, `gong2025figstep`,
`liu2024mmsafetybench`, `wang2024umk`, `wang2024qwen2vl`, `mazeika2024harmbench`,
`souly2024strongreject`. If you want to cite anything else, mark it `\todo{add bib entry: ...}`
rather than inventing a key. Use `\paragraph{}` only if it genuinely helps; prose is preferred.

Write the section now.
