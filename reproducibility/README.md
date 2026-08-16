# `reproducibility/` — canonical, isolated reproduction

This folder is the **frozen reproduction track**, deliberately separated from the working tree
(`../optimisation/`, `../evaluation/`, `../jobs/pipeline.sh`) where our *ablation* experiments modify
code freely. Nothing here imports the working modules, and nothing in the working tree imports these —
so edits on either side cannot silently change the other. Shared, read-only data (`../datasets/`,
`../train_configs/`, `../images/`) is used by relative path; outputs go to dedicated names.

The central finding of this whole study is that **the paper's described method and the authors' released
code disagree** (see [`../repro_notes/DISCREPANCIES.md`](../repro_notes/DISCREPANCIES.md), D1/D10/D11). A
single "reproduction" cannot honour both, so this folder keeps **both, side by side**:

| track | what it is | reproduces |
|---|---|---|
| [`authors_code/`](authors_code) | the authors' method code **verbatim** from commit `c4c276d` | *what they ran*. Their released `ultrabreak.png` scores as reported; a from-scratch retrain hits the D11 units bug (the band), so it does **not** reach the paper's numbers. |
| [`paper_faithful/`](paper_faithful) | the paper's **described** method implemented cleanly — Table 4 + Eq 8-10 + §3.2, with the CLIP normalisation present (D11) | *what they described*. Run at Table 4's **1300 iterations** (paper-exact). |

Diffing the two is the report's core evidence for where "described" and "ran" diverge.

## Key fidelity decisions (chosen with the user)

1. **Both tracks**, so the divergence is explicit rather than hidden behind one implementation.
2. **Paper-exact 1300 iterations**, with the gap documented — *not* silently raised to the 3000 that
   actually reproduces. Table 4 says 1300; we found 1300 under-reproduces (job 773: SafeBench 23.49%)
   and ~3000 reproduces (job 809: 83.49%). The 1300-run here records the paper-as-written result; the
   3000 finding lives in `DISCREPANCIES.md` (D11) and is produced by the working-tree ablation pipeline,
   not here.

## How to run

Always from the **repo root** (relative data paths):

```bash
# paper-faithful, paper-exact 1300 iterations (Qwen2-VL surrogate)
python reproducibility/paper_faithful/optimise_paper.py        # or: sbatch jobs/run_repro_paper_faithful.sh
# -> outputs/repro_paper_faithful_1300/1300.png, then evaluated white-box by the runner
```

`jobs/run_repro_paper_faithful.sh` drives **only** this folder's frozen code and is **not** part of
`jobs/pipeline.sh` — the ablation pipeline stays on the working tree, this reproduction stays here.

`authors_code/` has **no training runner** by design: its `optimise.py` is a dev placeholder (hardcoded
`test.csv`), so it cannot be run verbatim as a reproduction, and editing it would defeat the point of a
verbatim copy. It is a frozen reference + the eval pipeline that scores the released image. See
[`authors_code/README.md`](authors_code/README.md).

## What is NOT duplicated here

Evaluation of the *released artifact* (scoring `../outputs/ultrabreak.png`) and the judge passes already
live in the working tree and `../repro_notes/`; they are protocol, not method, and are shared. This folder
freezes the **training/method** code, which is where paper-vs-code divergence lives.

See [`../repro_notes/REPRO_CHECKLIST.md`](../repro_notes/REPRO_CHECKLIST.md) for what is and isn't reproduced,
and each subfolder's `README.md` for the per-track details and the Table-4 → code map.
