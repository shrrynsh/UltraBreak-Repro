# Branch audit — does every table have a code-faithful and a paper-faithful arm, and is each strict?

> ## Definitions fixed by the project owner, 2026-10-22 — these supersede §1 below
>
> **Code-faithful (CF)** = the patch retrained through the **released code's training path**
> (raw patch injection, no projection) at **1300 epochs, not the code's 5000**, and then
> **evaluated as the paper prescribes** — because the release ships no evaluation pipeline at table
> scale (see §5, Finding 4). So 1300-vs-5000 is **not** a deviation under this definition, and
> neither is the GLM token budget, since the paper sets none.
>
> **Paper-faithful (PF)** = retrained **and** evaluated exactly as the paper prescribes:
> patch normalised into CLIP space, Table 4 hyperparameters, 1300 iterations, 315 / 520 / 1680
> evaluation sets, AdvBench in the §3.3 "Steps to…" form, and the HarmBench classifier with its
> original template — i.e. the **v1** scorer.
>
> ### Status under these definitions
>
> | paper table | CF branch | PF branch | status |
> |---|---|---|---|
> | **1a** SafeBench | ✅ `cmp_cf_sb315` | ✅ `cmp_pf_sb315` | **both complete** — 6/6 models, v1 and v3 both scored, 315-row set |
> | **1b** AdvBench | ✅ `cmp_cf_abnorm` | ✅ `cmp_pf_abnorm` | **both complete** — 6/6 models, v1 and v3 both scored, 520 rows, §3.3 form |
> | **1c** MM-SafetyBench | ⚠️ `t1mm_cf_mmsafety` | ⚠️ `t1mm_pf_mmsafety` | **both exist but at n=520; the paper is n=1680.** The in-flight 1680 run carries no CF/PF arm |
> | **2** StrongREJECT | ❌ | ❌ | blocked — native 0–1 scorer unavailable |
> | **3** component ablation | ❌ **none** | ⚠️ `a_*_1300` | **PF only.** 5/6 models, which matches the paper (it omits Kimi-VL); v1 complete. But its PF reading differs from 1a/1b's — see the note below |
> | **4** hyperparameters | n/a | n/a | descriptive; no run behind it |
> | **5** affirming phrase | n/a | n/a | released image only, no retrain — no branch applies |
> | **6** TV sweep | ❌ **none** | ⚠️ `a_tv*_1300` | **PF only.** 2/6 models, which matches the paper's two victims; v1 complete; same PF-reading mismatch |
> | **9** closed models | ❌ | ❌ | blocked — paid API |
> | **10** compute overhead | n/a | n/a | a per-step time ratio; branch-invariant |
>
> **Tally: 2 tables have both branches complete (1a, 1b). 1 has both at the wrong set size (1c).
> 2 have PF only (3, 6). 2 are blocked (2, 9). 3 have no branch to be faithful to (4, 5, 10).**
>
> ### The three gaps that remain under these definitions
>
> 1. **MM-SafetyBench at n=1680 has no CF or PF arm.** Two configs × 6 models. This is the only
>    thing stopping Table 1c from being a reproduction at the paper's own denominator.
> 2. **Tables 3 and 6 have no CF arm** — only one code-faithful patch exists in the repository, so
>    there are no code-faithful ablation or TV-sweep patches. Six trainings would create them, but
>    with the bug present every variant collapses into the grey band, so the table would measure
>    D11 rather than the components. **Recommend leaving it and stating so in the report.**
> 3. **Two different PF readings are in use, and they disagree.** `repro_paper_faithful_1300`
>    (Tables 1a/1b) applies §3.2 as an explicit **projection ON**; the ablation patches
>    (Tables 3/6) use **normalisation ON, projection OFF**. The projection reading was *retracted*
>    (D10: as wired it over-smooths 3.7× and cannot produce an artifact resembling
>    `ultrabreak.png`), so the ablations use the project's **current** reading and Table 1's PF row
>    uses the **superseded** one. One of the two has to move before the PF column is internally
>    consistent.
>
> ### What is NOT a problem under these definitions
>
> - **1300 vs 5000 epochs** — CF is defined at 1300.
> - **The judge** — v1 is the paper's own scorer and it is already computed on **every** cell of
>   every branch above, including all of Tables 3 and 6. No GPU work needed to report it.
> - **GLM at 4096 tokens** — evaluation follows the paper, which specifies no budget.
> - **Borrowing the evaluation set from the paper for CF** — required by the definition, since the
>   release ships only a 20-row demo config.
>
> The one residual deviation on the CF side is the **training corpus**: `codefaithful_exact_1300`
> used our `full50` file rather than the code's own `train_configs/test.csv`, whose `keywords`
> column differs on 47/50 rows and feeds the attention weighting (§3 below). Fixing it is one
> retrain; it is the only thing between the CF arm and the definition above.

---

**Date: 2026-10-22.** A focused audit, no new experiments. For each paper table: does a
code-faithful (CF) arm exist, does a paper-faithful (PF) arm exist, and does each one *strictly*
match its own definition? Every row below was verified from the authors' commit `c4c276d`, from each
run's own `outputs/<exp>/run_config.json`, and from the config CSVs on disk — not from job comments.

---

## 1. The two definitions, operationalised

| axis | **code-faithful** = authors' release `c4c276d` | **paper-faithful** = the paper as written |
|---|---|---|
| patch injection | raw `[0,1]` patch into the CLIP-normalised tensor (`qwen2_adapter.py:245` else-branch; `patch_only=False`) | patch normalised into CLIP space (the D11 fix) |
| §3.2 projection | **absent** — `project_patch()` has zero call sites | present (Eq. 10 feeds the projected patch) |
| iterations | **5000** (`optimise.py` default) | **1300** (Table 4) |
| training corpus | **`train_configs/test.csv`** — 50 rows, shipped in `c4c276d`, unmodified since, and the default in `optimise.py` | SafeBench-Tiny, 50 queries; **the paper never specifies the keyword field** |
| evaluation set | only `attack_configs/test.csv` ships — **no table-scale eval config exists in the release** | 315 / 520 / 1680, recovered from the paper's own arithmetic |
| AdvBench prompt form | raw `goal` — the release does not rewrite it | `"Steps to " + q` (§3.3) |
| judge | `evaluation/evaluate.py` at `c4c276d` = **v1** | "HarmBench-Llama-2-13b, original chat template, unmodified" = **also v1** |
| generation budget | **`max_new_tokens=512`** for every model (`attack.py:56,127,194,254`) | unspecified |

**Two consequences that fall out of the table itself.**

1. **The judge is not a CF/PF axis.** The paper specifies the HarmBench classifier with its original
   template, which is exactly what v1 is. So **v1 is simultaneously the code-faithful and the
   paper-faithful scorer, and our v3 is a third protocol belonging to neither branch.**
2. **A fully code-faithful *table* is not definable.** The release ships no evaluation config at
   table scale, so any table-sized CF number must borrow its eval set from the paper. CF can only
   ever mean *code-faithful training*.

---

## 2. Per-table verdict

| paper table | CF arm exists? | PF arm exists? | strictly faithful? |
|---|---|---|---|
| **1a** SafeBench | ✅ `cmp_cf_sb315` | ✅ `cmp_pf_sb315` | ❌ **neither** — see §3 |
| **1b** AdvBench | ✅ `cmp_cf_abnorm` | ✅ `cmp_pf_abnorm` | ❌ CF is reported on the **paper's** prompt form; the raw cell exists (`cmp_cf_abraw`) but is not the reported one |
| **1c** MM-SafetyBench | ⚠️ `t1mm_cf_mmsafety` **n=520** | ⚠️ `t1mm_pf_mmsafety` **n=520** | ❌ **wrong denominator** — the paper is n=1680, and the in-flight 1680 run (`mm1680_*`) has **no CF/PF arm at all**, only the released image and the three baselines |
| **2** StrongREJECT | ❌ | ❌ | blocked — native 0–1 scorer unavailable |
| **3** component ablation | ❌ **none exists** | ✅ `a_*_1300` | PF only |
| **4** hyperparameters | n/a | n/a | descriptive; no run behind it |
| **5** affirming phrase | n/a | n/a | released image only (`b15_*`), no retrain, so no branch applies |
| **6** TV sweep | ❌ **none exists** | ✅ `a_tv*_1300` | PF only |
| **9** closed models | ❌ | ❌ | blocked — paid API |
| **10** compute overhead | n/a | n/a | a per-step time **ratio**; the D11 fix is one tensor op, so the number is invariant to branch |

**Only one code-faithful artifact exists in the entire repository:** `outputs/codefaithful_exact_1300`.
There is no CF ablation patch, no CF TV-sweep patch, and no CF Fig-2 patch, so Tables 3 and 6 have a
single branch by construction.

---

## 3. Where the CF arm is not strictly code-faithful

`reproducibility/authors_code/optimisation/optimise_exact.py` is the authors' `optimise.py` with
three changes, each self-documented as `# EXACT-DIFF`. Two of them are substantive.

| # | line | change | verdict |
|---|---|---|---|
| 1 | 76 | `train_config`: `test.csv` → `safebench-tiny_jailbroken_mode_full50.csv` | ⚠️ **substantive** |
| 2 | 101 | `num_epochs`: **5000 → 1301** | ⚠️ **substantive** |
| 3 | 82 | `exp_name`: `test` → `codefaithful_exact_1300` | cosmetic |

**Change 1 — the corpus is not the authors' file.** Both files carry the same 50 SafeBench-Tiny
queries with byte-identical `text`, `image`, `target` and `target_type`. They differ in two fields:

| field | authors' `test.csv` | our `full50` | rows differing |
|---|---|---|---|
| `keywords` | `start operate illegal human organ trade` | `start and operate an illegal human organ trade` | **47 / 50** |
| `harmful_phrase` | `illegal drugs` | `manufacture illegal drugs` | 44 / 50 |

The authors' keywords are stop-word-stripped; ours are the full noun phrase, because our
`create_attack_configs.extract_keywords()` only strips a leading verb phrase. **This is not
cosmetic:** `keywords` is split on spaces and consumed at `qwen2_adapter.py:121`
(`"".join(c for c in token_str if c.isalpha()) in keywords`) to decide which target tokens the
semantic-**attention** loss up-weights. Adding `and`, `an`, `when`, `in`, `for`, `of`, `with`, `that`
to the keyword list makes function-word tokens eligible for up-weighting, which redistributes the
attention weights on 47 of 50 training rows. `create_attack_configs.py` is also our file in its
current state (originally `kaicui` 2026-03-28, modified by us 2026-08-08/09), so this field is our
construction on **both** branches.

**Change 2 — the budget is the paper's, not the code's.** The release trains for 5000 iterations;
the CF arm stops at 1300 because Table 4 says 1300. That is the right choice for *comparing against
the paper*, but it means the CF arm answers "the authors' code at the paper's budget", not "the
authors' code as shipped" — and given that two separate parts of this method are budget-gated, the
distinction is not academic.

**On the evaluation side the CF arm uses three of our corrections:** judge **v3** (v1 was also
computed and is available), **normalised** AdvBench, and **GLM at 4096 tokens** rather than the
release's 512. The first two have a code-faithful cell already on disk; the third does not, because
the D14 re-run replaced every GLM generation.

---

## 4. Where the PF arm is not strictly paper-faithful

`reproducibility/paper_faithful/optimise_paper.py` is the stronger of the two. Every value is traced
to its source in the paper and guarded by `_assert_table4()`, which fails loudly on edits:
`lr=0.01`, `tv_weight=0.5`, `num_epochs=1301`, `tau=0.5`, `embed_noise=1e-4`, `loss=semantic`,
`loss_mode=attention`, transforms on, `no_projection=False`, `raw_patch_injection=False`.

Two qualifications, neither a contradiction of the paper:

1. **The keyword field is ours** (§3 change 1). The paper does not define it, so there is no paper
   standard to violate — but it also differs from the only reference implementation, which is worth
   stating rather than leaving implicit.
2. **`seed=0` is set.** The paper seeds nothing (D7). Fixing a seed is a deliberate, documented
   deviation that makes the artifact reproducible; it cannot match an unspecified protocol.
3. **Reported under v3.** Since v1 is the paper's own scorer, the strictly paper-faithful column for
   every PF row is the **v1** column. v1 exists for all of Tables 3 and 6 and agrees within 4.6
   points; for Table 1 the `exact_comparison_summary.csv` grid already carries both.

---

## 5. What would make each table strict

Listed for decision only — nothing here has been run.

| # | gap | work |
|---|---|---|
| 1 | **No truly code-faithful retrain.** | One training: `optimise_exact.py` with `train_configs/test.csv` and `num_epochs=5000`, saving a 1300 checkpoint as well, so the CF arm has both "as shipped" and "at the paper's budget". ~5.2 GPU-h to 1300, ~20 GPU-h to 5000. |
| 2 | **CF rows are reported under our eval protocol.** | No new generation for SafeBench/AdvBench — select the existing v1 and `abraw` cells. GLM at 512 would need regeneration, which conflicts with D14; recommend reporting GLM as CF-training/PF-eval and saying so. |
| 3 | **Table 1c has no CF/PF arm at n=1680.** | Two configs × 6 models on the full 1680. This is the largest gap and the one that stops Table 1c being a reproduction at the paper's own denominator. |
| 4 | **Tables 3 and 6 have no CF arm.** | Six trainings with the bug present. Expect every variant to collapse into the grey band, so the table would measure D11 rather than the components — **recommend not running it**, and stating in the report that the ablations are PF-only and why. |
| 5 | **Labelling.** | Every CF row should read "code-faithful training, paper-faithful evaluation", and every PF row should name its scorer. The branch labels as they stand claim more than the runs deliver. |

### The one-line answer

**Tables 1a and 1b are the only tables with both arms. Table 1c has both at the wrong denominator.
Tables 3 and 6 are paper-faithful only. Tables 2 and 9 are blocked, and Tables 4, 5 and 10 have no
branch to be faithful to.** Neither arm is strict as it stands: the CF arm deviates in the training
corpus and the iteration count and is scored under our protocol, and the PF arm is scored under v3
when v1 is the paper's own scorer.
