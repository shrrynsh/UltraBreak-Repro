# Branch contract — the definitions of code-faithful and paper-faithful

**Fixed by the project owner, 2026-10-22. This file is the authority.** No job that trains or
evaluates a patch for a paper table may be submitted until it passes
`python reproducibility/check_branch.py`, which encodes every clause below as an assertion.

The release ships **no evaluation pipeline at table scale** — only `attack_configs/test.csv`, a
20-row demo pointing at the authors' own finished image. So evaluation is the **paper's** protocol
on **both** branches, and "code-faithful" constrains training only.

---

## CF — code-faithful

*The released code's training path, at 1300 epochs rather than the code's own 5000, evaluated as
the paper prescribes.*

### Training

| # | clause | how it is checked |
|---|---|---|
| CF1 | Trainer is `c4c276d:optimisation/optimise.py` with **at most two changed lines**, and those two may only be `exp_name` and `num_epochs` | unified diff against the git blob; exactly 4 +/- lines; both anchors named |
| CF2 | `num_epochs = 1301`, so epoch 1300 is saved | string match in the trainer |
| CF3 | Corpus is **`./train_configs/test.csv`** — the authors' own file, byte-identical to `c4c276d` | blob comparison, and the corpus line in the trainer is unaltered |
| CF4 | The whole `optimisation/` tree is byte-identical to `c4c276d` (6 files) | blob comparison |
| CF5 | **Raw patch injection — the D11 bug is PRESENT.** `patch_only=False`, so the else-branch runs | source inspection of `Qwen2Adapter.compute_loss` + `patch_only=False` in the trainer |
| CF6 | **No §3.2 projection** — the release has none (`project_patch()` has zero call sites) | the pristine-tree check covers it |
| CF7 | **No seed inside the authors' file** — adding one would be a third changed line and break CF1. The seed is applied *externally* by `jobs/_seed_and_run.py`, which seeds `random`, `numpy`, `torch` and CUDA and then calls the untouched trainer's `main()`. See S1 for the seed set | no `manual_seed`/`random.seed`/`np.random.seed` in the trainer; the wrapper is used |
| CF8 | Hyperparameters exactly as shipped: lr 0.01, λ_TV 0.5, λ_L2 0.0, `custom_loss=True`, clamp to [0,1] each step | guaranteed by CF1; re-asserted by string match |

**Consequence of CF7 to state in the report:** the authors' own runs were unseeded, so any single
number they report is one draw from an unquantified distribution. Ours are seeded (S1), so our
numbers carry a spread — which is the honest comparison and is stronger than matching a point
estimate.

### Reference implementation
`reproducibility/authors_code/optimisation/optimise_cf_strict_s{0,1,2}.py`, each launched through
`jobs/_seed_and_run.py` → `outputs/cf_strict_test50_1300_s{0,1,2}/`

---

## PF — paper-faithful  ·  **§3.2 = Reading A (projection)**

*Retrained and evaluated exactly as the paper describes, reading §3.2 literally.*

**The §3.2 decision.** The paper defines `x_proj = clip(γ·x + β, 0, 1)` with γ = CLIP_STD,
β = CLIP_MEAN, and Eq. 10 feeds the **projected** patch to the model. Its stated purpose is *"to
reduce reliance on extreme pixel values"*. **PF implements that literally: the projection is an
active compression step.** The competing reading — that γ and β are performing the CLIP
normalisation the code forgot, i.e. a conversion rather than a constraint — is **not** PF. It is
reported as our own extension; see `repro_notes/BEYOND_REPRO.md` §B25.

| # | clause | how it is checked |
|---|---|---|
| PF1 | **§3.2 projection ACTIVE** — `projection: true` | `run_config.json` |
| PF2 | Patch normalised into CLIP space before injection — `normalise_patch: true`. Required for type-consistency: Eq. 10 feeds the model, whose input is the CLIP-normalised tensor, so a raw [0,1] patch is a scale error | `run_config.json` |
| PF3 | `num_epochs = 1301` → 1300 iterations (Table 4) | `run_config.json` |
| PF4 | Table 4 verbatim: lr 0.01, λ_TV 0.5, τ 0.5, ε 1e-4, `loss=semantic`, `loss_mode=attention`, transforms ON, scale 0.8–1.2, rotation ±15 | `run_config.json` |
| PF5 | Corpus is **`./train_configs/test.csv`** — the authors' own SafeBench-Tiny file, byte-identical to `c4c276d`. The paper says "SafeBench-Tiny, 50 queries" and this file is exactly that, so using it satisfies the paper **and** removes the only corpus difference between CF and PF | named corpus is `test.csv`, 50 rows, blob-identical |
| PF6 | Surrogate is `Qwen/Qwen2-VL-7B-Instruct` | `run_config.json` |
| PF7 | `seed` ∈ **{0, 1, 2}** — the same three as CF, per S1 | `run_config.json` |

**Two declared deviations, both unavoidable, both to be stated in the report.**

- ~~**PF5, the keyword field.**~~ **Resolved 2026-10-22.** The paper never defines the `keywords`
  column that drives the attention weighting, so there was no paper standard — but leaving PF on our
  `full50` file while CF used the authors' `test.csv` would make the two branches differ on **two**
  axes, §3.2 *and* a keyword column that disagrees on 47/50 rows. PF5 now requires the authors'
  `test.csv` on both branches. **CF and PF therefore differ in exactly one thing: the §3.2
  treatment.** That is the whole point of having two branches.
- **PF7, the seed.** The paper seeds nothing (D7). PF fixes seeds {0, 1, 2} per S1. This is a
  deliberate deviation in the direction of *more* reproducibility.

**Known property of PF — corrected 2026-10-22.** An earlier version of this file claimed Reading A
confines the patch to `γ·[0,1] + β = [0.478, 0.741]` by arithmetic. **That is wrong, and applies
only when the latent is *also* clamped to [0,1]** (the O1 double-constraint; `clamp_latent: true`,
as in `full50_proj_doublesquash_s3000`, measured `[0.325, 0.855]`, σ = 0.044). PF sets
`clamp_latent: false`, so the latent is free and `γx + β` reaches the full `[0,1]` for
x ∈ [−1.79, 1.93]. **There is no hard range bound in PF.**

What is true is empirical, not arithmetic:

| patch | σ | range |
|---|---|---|
| released `ultrabreak.png` | **0.164** | [0.000, 1.000] |
| Reading B @1300 (job 773) | 0.142 | [0.000, 1.000] |
| CF, raw injection @1300 | 0.131 | [0.000, 1.000] |
| **PF, Reading A @1300** | **0.052** | [0.027, 0.816] |

PF produces **3.1× less contrast** than the released image. The mechanism is D10: the TV and L2
penalties act on the **raw latent** (`optimise_proj.py:248-249`), i.e. in normalised space, where
they are effectively `1/γ ≈ 3.7×` stronger. That is a real property of the literal reading and worth
reporting — but it is a regularisation-scale effect, not an impossibility proof. Do not "fix" it
inside PF.

### Reference implementation
`reproducibility/paper_faithful/optimise_paper_strict_s{0,1,2}.py` (generated from the
single-seed original, now removed) →
`outputs/pf_strict_test50_1300_s{0,1,2}/`

`repro_paper_faithful_1300` is the **superseded** PF artifact: correct on PF1–PF4, PF6, PF7 but
trained on our `full50` corpus, so it fails the tightened PF5.

---

## S1 — seeds  ·  three fixed seeds, the same set on both branches

**Checked first, as the contract requires: the paper says nothing about seeding.** Its full text was
reviewed when `repro_notes/paper_specs.md` was built and records no seed clause, and a grep for
`manual_seed` / `random.seed` / `np.random.seed` across the authors' `optimisation/` tree returns
nothing (D7). There are no authors' seeds to adopt.

| # | clause |
|---|---|
| S1a | Both branches train on **seeds {0, 1, 2}** — the *identical* set, so a CF-vs-PF difference can never be a difference of seeds |
| S1b | Six trainings in all: `cf_strict_test50_1300_s{0,1,2}` and `pf_strict_test50_1300_s{0,1,2}` |
| S1c | **Every table reports all three seeds individually** — one column or row per seed, s0 / s1 / s2 — with mean ± spread as an *additional* summary, never as a replacement. See S1e for which rows this binds |
| S1e | **Scope of S1c** (set by the project owner 2026-10-24). S1c binds every row produced by an optimisation **we run in order to reproduce UltraBreak itself**: the CF and PF branch rows, and the PF ablation and TV variants. Rows that are not that are reported at n=1 and **explicitly labelled as such** — never silently |
| S1d | Three sources of randomness must be covered: patch initialisation (`torch.rand`), per-step transform parameters (`random.uniform`, `utils.py:37-39`), and target-embedding noise (`torch.randn_like`, `utils.py:185`) |

**The honest limit, to be stated wherever the spread is reported.** Seeding pins the *sampling*, not
the arithmetic. `grid_sample`'s backward pass has no deterministic CUDA kernel and every training
step passes through it, so two runs at the same seed still drift. **Three seeds give a variance
estimate; they do not make training bitwise reproducible.** The spread is therefore a lower bound on
the true run-to-run variance, not an exact one.

**Which rows S1c binds (S1e).** A spread can only be reported where there is something random to
vary. Three classes, and all three get a visible label:

| row class | seeds | label in the table |
|---|---|---|
| CF / PF branch rows; PF ablation + TV variants | **3, mandatory** | `s0 / s1 / s2` + mean ± spread |
| VAJM, UMK — our trainings, but baselines | 1 | `n=1, not seed-averaged` |
| released `ultrabreak.png`, No Attack, FigStep | 1 — **nothing to seed** | `single artifact / no training` |

The principle: S1c exists so that a *claim about UltraBreak's reproducibility* carries an interval.
VAJM and UMK are context for that claim, not the claim, so three seeds each — six more trainings
plus evaluation — buys nothing the report needs. The released image is one file the authors
published; No Attack and FigStep involve no optimisation at all. **Saying "this row is one draw" is
more honest than manufacturing a spread for it**, and it keeps the three-seed guarantee meaning
exactly what it claims to mean.

**Reporting rule (S1c), set by the project owner 2026-10-22.** Tables show the *per-seed* numbers,
not a collapsed mean. A reader must be able to see s0, s1 and s2 and judge the spread for
themselves; a mean hides whether three draws sit at 63/64/65 or at 40/64/88, and those support very
different claims. Mean ± spread goes beside the three, not instead of them.

**Why this matters more than usual here.** Every disagreement this study reports has rested on n=1
per configuration — explicitly caveated in `FINDINGS.md`, D7 and the O1/O2 entries. With three
seeds, a gap like PF's 64.13 against the paper's 81.59 becomes a claim with an interval attached
instead of a single draw against a single draw.

---

## Evaluation — identical on both branches (the paper's protocol)

| # | clause | how it is checked |
|---|---|---|
| E1 | SafeBench: **315** rows, built with `--exclude-train datasets/SafeBench-Tiny.csv` | config row count |
| E2 | AdvBench: **520** rows, built with `--normalize` — the §3.3 `"Steps to " + q` form | row count + >90% of rows start with "Steps to" |
| E3 | MM-SafetyBench: **1680** rows — the FULL `Text_only` set, **no subsample anywhere**, decided by the project owner 2026-10-24. The paper states "1680 harmful targets covering 13 scenarios"; sweeping every denominator from 100 to 6000 against all **24** reported MM-SafetyBench values leaves only 1680 and its multiples fitting all 24, while the 520 stratified subsample fits **2/24**. A subsample is therefore arithmetically excluded, and no number computed on one may appear in a CF or PF column. The existing 520-row MM results are retained only as internal instruments (they are a true 519/519 subset, so they remain valid for CF-vs-PF and seed-vs-seed comparisons), never as a Table 1c row | config row count == 1680 |
| E4 | Every config points at the branch's own patch and nothing else | unique image column |
| E7 | **No branch config may reference `outputs/ultrabreak.png`.** Both branches are *retrains*; the released image appears only as a separate, labelled reference row, never inside a branch | image column scanned for `ultrabreak` |
| E5 | **Headline scorer is v1** — `c4c276d:evaluation/evaluate.py`, the paper's "HarmBench-Llama-2-13b, original chat template, without modification". v3 is reported beside it as our correction, never instead of it | both `_harmbench.csv` and `_harmbench_v3.csv` present per cell |
| E6 | GLM generates at **4096** tokens. The release hardcodes 512, which truncates GLM mid-reasoning on up to 98.7% of rows; the paper states no budget, so the paper's protocol does not require 512 | GLM venv in use |

---

## Status of every paper table under this contract

**Audited 2026-10-24 by `python reproducibility/audit_tables.py` — run it, do not trust this table.**
A branch row counts only if, for *every* seed in {0,1,2}, a config exists whose patch is that
branch's contract patch for that seed, evaluated on every model the table reports, under both
judges (E5).

| table | CF | PF | seeds | what it is built on today |
|---|---|---|---|---|
| **1a** SafeBench 315 | ❌ | ❌ | 0/3 | CF `codefaithful_exact_1300` (fails CF1, CF3a), PF `repro_paper_faithful_1300` (fails PF5) |
| **1b** AdvBench 520 | ❌ | ❌ | 0/3 | same two artifacts |
| **1c** MM-SafetyBench | ❌ | ❌ | 0/3 | same two artifacts, and only at n=520 — **E3 wants 1680** |
| **2** StrongREJECT | — | — | — | no branch row: native 0–1 scorer unavailable |
| **3** ablations | **n/a by definition** | ❌ | 0/3 | `a_*_1300` fail PF1 (projection off) and PF5 (corpus); needs **9** PF-style variant retrains |
| **4** hyperparameters | — | — | — | descriptive, no patch |
| **5** affirming phrase | — | — | — | the authors' released image, no retrain |
| **6** TV sweep | **n/a by definition** | ❌ | 0/3 | as Table 3; needs **6** PF-style variant retrains |
| **9** closed models | — | — | — | blocked: paid API |
| **10** overhead | — | — | — | a per-step ratio, branch-invariant |

**Not one table currently satisfies either branch.** Every CF/PF number in `TABLES.md` predates
this contract: the six contract patches finished training on 2026-10-24 and **have never been
evaluated** — no `attack_configs/*.csv` references them and no `results/*strict*` exists.

### Tables 3 and 6 can never have a CF arm

CF1 permits at most two changed lines, `exp_name` and `num_epochs`. An ablation must change a
third (`--loss ce`, `--loss_mode token`, `--tv_weight 0.2`). **A code-faithful ablation is a
contradiction in terms**, so those cells are marked *n/a by definition* rather than missing. The
`a_*_1300` runs stay where they already are: extension work, not a branch.

### A PF ablation row is not built on an anchor patch

It is a PF-style retrain — projection ON, authors' `test.csv`, Table 4 — with exactly **one** knob
moved. That moved knob is the experiment, so it is an *intended* deviation from PF4, not a
violation; `audit_tables.py` reports it as such. Expected artifacts, three seeds each:

| table | variants | artifacts |
|---|---|---|
| 3 | no-constraints, CE, token | `pf_abl_{noconstraints,ce,token}_s{0,1,2}` |
| 6 | TV 0.2, TV 1.0 | `pf_tv{0p2,1p0}_s{0,1,2}` |

The unablated row of Table 3 and TV=0.5 of Table 6 **are the PF anchor itself**, so they need no
extra training — they come free with Table 1.

**Choosing Reading A has one clean side effect worth noting:** a PF ablation arm is trained with
the projection ON, exactly like the PF anchor it is compared against — so the ablations and their
baseline differ by the ablated knob alone. That removes the baseline mismatch recorded in
`repro_notes/TABLES.md` §Table 3 at the same time.
