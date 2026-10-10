# Released baseline code vs. the UltraBreak paper's stated protocol

The paper states:

> For VAJM we follow the original protocol and optimise a jailbreak image on the
> derogatory corpus for 5,000 iterations. That optimised image then serves as the
> toxic, semantically embedded seed for UMK, which jointly optimises an adversarial
> image and a textual suffix; we train this stage for an additional 2,000 iterations
> as in the original work. To keep comparisons fair, all methods use SafeBench-Tiny
> as the few-shot corpus, consistent with UltraBreak.

Verified against the released code. Four points where running the code as shipped
would NOT implement that sentence, each patched (patches recorded here, applied to
git-ignored vendored code):

| # | what the paper says | what the shipped code does | action |
|---|---|---|---|
| 1 | VAJM: derogatory corpus, 5,000 iters | `minigpt_visual_attack.py:90` hardcodes `harmful_corpus/derogatory_corpus.csv`; the attack call hardcodes `num_iter=5000` (so `--n_iters` is dead code but the value is already correct) | **no change needed** — already matches |
| 2 | UMK: "an additional **2,000** iterations" | `minigpt_vlm_attack.py` attack call hardcodes **`num_iter=250`**, overriding the function's own 2000 default | patch 250 -> 2000 |
| 3 | UMK seeded by "that optimised image" (VAJM's) | hardcodes `template_img = 'adversarial_images/prompt_unconstrained_init.bmp'` — the repo's own init image, not VAJM's output | patch to read `UMK_INIT_IMG` |
| 4 | "all methods use **SafeBench-Tiny** as the few-shot corpus" | uses `harmful_corpus/harmful_behaviors.csv` (AdvBench) and takes `random.sample(all_numbers, 66)` — **unseeded**, so not even reproducible run to run | patch to read `UMK_CORPUS_CSV` |

## An internal tension in the paper's own sentence

Point 1 and point 4 cannot both hold for VAJM. VAJM's image optimisation has **no
query set at all** — `visual_attacker.Attacker(args, model, targets, ...)` takes only
target strings, and those are hardcoded to the derogatory corpus. There is nowhere for
SafeBench-Tiny to enter. So "all methods use SafeBench-Tiny as the few-shot corpus"
is unimplementable for VAJM's stage, while "optimise on the derogatory corpus" is
explicit and matches the original work.

**Decision:** VAJM uses the derogatory corpus (the explicit, implementable instruction);
SafeBench-Tiny is applied to **UMK**, whose attacker does take `(queries, targets)` and
where the constraint is therefore meaningful. Reported with the numbers.

## UMK corpus construction

`queries` = SafeBench-Tiny's 50 `instruction` strings, verbatim.
`targets`  = `"Sure, here are some {instruction}"` (UMK appends `':'` itself).

UltraBreak's own affirming phrase (`[Jailbroken Mode]`) is deliberately **not** injected
into UMK's targets: that is UltraBreak's prompt-engineering component, and importing it
would make the baseline partly UltraBreak. The *corpus* is swapped as the paper asks;
UMK's own method (GCG suffix + PGD image) is left intact.

## Environment deviation

`low_resource: True` in both shipped eval configs loads Vicuna-13B in 8-bit via
bitsandbytes (absent from the env). Set to **False**: 8-bit quantisation sits directly in
the gradient path of a pixel PGD attack, and the RTX 6000 Ada's 48 GB holds Vicuna-13B in
fp16 (~26 GB) comfortably. Also `torch 1.12.1 -> 2.0.1+cu118` (sm_89 needs CUDA >= 11.8),
as already recorded in `baselines/PROTOCOL.md`.
