# `authors_code/` — the authors' method code, verbatim from `c4c276d`

Extracted unmodified with `git show c4c276d:<path>` for every method file under `optimisation/` and
`evaluation/`. This is the frozen reference for **what the authors actually ran**. Do not edit — if you
want to change behaviour, do it in the working tree or in `../paper_faithful/`.

```
optimisation/   optimise.py, utils.py, base_adapter.py, qwen2_adapter.py, llava16_adapter.py, __init__.py
evaluation/     attack.py, evaluate.py, judge_llms/{harmbench_judge,general_judge,llama_guard,judge_model}.py
```

## What this reproduces

- **Evaluation of the released artifact reproduces the paper.** Scoring the authors' `ultrabreak.png`
  with `evaluation/evaluate.py` here (the v1 judge) gives SafeBench 257/315 = **81.59%** (exact) and
  AdvBench 388/520 = **74.62%** — see `../../repro_notes/FINDINGS.md`.
- **Retraining from scratch does NOT reproduce.** The training path carries the defects the study
  documents, so a fresh optimisation lands far below the paper.

## Known defects in this code (do not "fix" here — that is what `paper_faithful/` is for)

| id | defect | where |
|---|---|---|
| **D11** | the patch is injected **raw** into the already-normalised pixel tensor (never CLIP-normalised) on the `patch_only=False` path every run takes → training optimises inside a 27% grey band | `optimisation/qwen2_adapter.py` `compute_loss` (else branch); `llava16_adapter.py` (commented out) |
| **D1** | the §3.2 projection `project_patch()` has **zero call sites** — dead code; only `clamp(x,0,1)` runs | `optimisation/utils.py` / `optimise.py` |
| **D12** | the judge anchors on the first literal `assistant`, which matches Qwen's system prompt → scores some refusals as successes | `evaluation/evaluate.py` extraction |

Full analysis and the arithmetic proof that "the released training code cannot produce the released
artifact" are in [`../../repro_notes/DISCREPANCIES.md`](../../repro_notes/DISCREPANCIES.md).

## Running it — and why there is no verbatim training runner

The authors' `optimisation/optimise.py` is a **dev placeholder**: `main()` hardcodes
`train_config = "./train_configs/test.csv"`, `exp_name = 'test'`, `num_epochs = 5000`, with no CLI. So it
cannot be run verbatim as a paper reproduction — it would train on a placeholder file. Editing those
constants would break "verbatim," which is the whole point of this frozen copy. We therefore **do not**
ship a training runner here.

What this track is actually used for:
- **Audit / diff** — the frozen reference for what the released method code was (cite it, diff it against
  `../paper_faithful/`).
- **Evaluation reproduction** — `evaluation/attack.py` + `evaluation/evaluate.py` here are the v1 pipeline
  that scores `../../outputs/ultrabreak.png` to the paper's numbers (done in the working tree; see
  `../../repro_notes/FINDINGS.md`).

To *demonstrate* the released training path's failure, the working-tree runs already do it faithfully
(jobs 622/625 take the same `patch_only=False` raw-injection path): washed-out band artifact (std ~10),
low ASR. That is the D11 reproduction; the paper-exact, corrected attempt is `../paper_faithful/`.
