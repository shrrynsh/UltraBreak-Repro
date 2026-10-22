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
| CF7 | **No seed.** The authors set none; adding one would deviate | no `manual_seed`/`random.seed`/`np.random.seed` in the trainer |
| CF8 | Hyperparameters exactly as shipped: lr 0.01, λ_TV 0.5, λ_L2 0.0, `custom_loss=True`, clamp to [0,1] each step | guaranteed by CF1; re-asserted by string match |

**Consequence of CF7 to state in the report:** a CF run is **one unseeded draw**. Its number cannot
be reproduced by re-running, only by reusing the saved PNG. The patch is the artifact.

### Reference implementation
`reproducibility/authors_code/optimisation/optimise_cf_strict.py` → `outputs/cf_strict_test50_1300/`

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
| PF7 | `seed: 0` | `run_config.json` |

**Two declared deviations, both unavoidable, both to be stated in the report.**

- ~~**PF5, the keyword field.**~~ **Resolved 2026-10-22.** The paper never defines the `keywords`
  column that drives the attention weighting, so there was no paper standard — but leaving PF on our
  `full50` file while CF used the authors' `test.csv` would make the two branches differ on **two**
  axes, §3.2 *and* a keyword column that disagrees on 47/50 rows. PF5 now requires the authors'
  `test.csv` on both branches. **CF and PF therefore differ in exactly one thing: the §3.2
  treatment.** That is the whole point of having two branches.
- **PF7, the seed.** The paper seeds nothing (D7). PF fixes seed 0 so the artifact is reproducible.
  This is a deliberate deviation in the direction of *more* reproducibility.

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
`reproducibility/paper_faithful/optimise_paper_strict.py` → `outputs/pf_strict_test50_1300/`

`repro_paper_faithful_1300` is the **superseded** PF artifact: correct on PF1–PF4, PF6, PF7 but
trained on our `full50` corpus, so it fails the tightened PF5.

---

## Evaluation — identical on both branches (the paper's protocol)

| # | clause | how it is checked |
|---|---|---|
| E1 | SafeBench: **315** rows, built with `--exclude-train datasets/SafeBench-Tiny.csv` | config row count |
| E2 | AdvBench: **520** rows, built with `--normalize` — the §3.3 `"Steps to " + q` form | row count + >90% of rows start with "Steps to" |
| E3 | MM-SafetyBench: **1680** rows (recovered from the paper's own arithmetic, 12/12 fit, p=5.1e-10) | config row count |
| E4 | Every config points at the branch's own patch and nothing else | unique image column |
| E7 | **No branch config may reference `outputs/ultrabreak.png`.** Both branches are *retrains*; the released image appears only as a separate, labelled reference row, never inside a branch | image column scanned for `ultrabreak` |
| E5 | **Headline scorer is v1** — `c4c276d:evaluation/evaluate.py`, the paper's "HarmBench-Llama-2-13b, original chat template, without modification". v3 is reported beside it as our correction, never instead of it | both `_harmbench.csv` and `_harmbench_v3.csv` present per cell |
| E6 | GLM generates at **4096** tokens. The release hardcodes 512, which truncates GLM mid-reasoning on up to 98.7% of rows; the paper states no budget, so the paper's protocol does not require 512 | GLM venv in use |

---

## Status of every paper table under this contract

| table | CF | PF | what is missing |
|---|---|---|---|
| **1a** SafeBench | 🔄 job 9243 | ✅ `cmp_pf_sb315` | CF being rebuilt strictly; old CF used our corpus |
| **1b** AdvBench | 🔄 job 9243 | ✅ `cmp_pf_abnorm` | same |
| **1c** MM-SafetyBench | ❌ | ❌ | both arms exist only at n=520; **E3 requires 1680** |
| **2** StrongREJECT | ❌ | ❌ | blocked — native 0–1 scorer unavailable |
| **3** ablations | ❌ | ❌ | **the `a_*_1300` patches have `projection: false` → they fail PF1 and are now extension work.** A PF arm needs 6 retrains with the projection ON |
| **4** hyperparameters | n/a | n/a | descriptive |
| **5** affirming phrase | n/a | n/a | released image, no retrain |
| **6** TV sweep | ❌ | ❌ | same as Table 3 — fails PF1; needs 3 retrains with the projection ON |
| **9** closed models | ❌ | ❌ | blocked — paid API |
| **10** overhead | n/a | n/a | a per-step ratio, branch-invariant |

**Choosing Reading A has one clean side effect worth noting:** a PF ablation arm would be trained
with the projection ON, which is also what the PF anchor (`repro_paper_faithful_1300`) uses — so the
ablations and their baseline would differ by the ablated knob alone. That removes the baseline
mismatch recorded in `repro_notes/TABLES.md` §Table 3 at the same time.
