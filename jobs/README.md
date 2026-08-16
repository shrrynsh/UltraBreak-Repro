# jobs/ — SLURM job scripts

Every batch script for the reproduction study. All `cd` to the repo root, assert the pinned `repro/`
venv, and write logs to `logs/job_%j.out`. Submit with `sbatch jobs/<script>.sh` (some take args, noted).
Results land in `results/`, trained patches in `outputs/`, configs in `attack_configs/`.

Grouped by phase, newest-relevant first. "job#" is the SLURM id(s) of the run(s) that produced the
findings; scores are v3 judge on the clean 315 (SafeBench) / 520 (AdvBench) sets unless noted.

## Original authors' drivers (pre-fork)
| script | what it does |
|---|---|
| `run_optimization.sh` | authors' training entry (`optimisation/optimise.py`) |
| `run_attack.sh` | authors' generation entry (`evaluation/attack.py`) |
| `run_evaluation.sh` | authors' judging entry (`evaluation/evaluate.py`) |

## Reproduction & training arc
| script | job# | what / key result |
|---|---|---|
| `run_step1_advbench_ultrabreak.sh` | — | score authors' `ultrabreak.png` on normalised AdvBench (Defect A / [[R1→D13]]) |
| `run_step3_retrain_full50.sh` | 622 | retrain, no projection (buggy) → SafeBench 29.84 / AdvBench 1.73 |
| `run_step4_retrain_full50_proj.sh` | 625 | retrain + projection (double-squash) → 48.25 / 22.31 |
| `run_step5a_proj_unclamped_unitinit.sh` | 639 | projection, latent unclamped, unit init → 53.65 / 5.96 |
| `run_step5b_proj_unclamped_imginit.sh` | 640 | projection, unclamped, image init → 42.54 / 2.31 |
| `run_normfix_confirm.sh` | 773 | **D11 fix** (normalise at injection), no proj, 1300 steps → 23.49 / 1.35 (looked like a regression) |
| `run_normfix_grid.sh <lr> <epochs>` | 808, 809 | grid: 808 lr0.003/1300 → 34.60 ; **809 lr0.01/3000 → 83.49 / 39.04 (SafeBench reproduced)** |
| `run_proj_variants.sh <projfix\|doublesquash> <epochs>` | 843, 844 | projection @3000, fix on vs off — does the projection help post-fix? (in flight) |

## Diagnosis
| script | job# | what / key result |
|---|---|---|
| `run_diag_normalisation.sh` | 683 | zero-training band tests T1/T2/T3 — T1 squeezed authors' patch 67.69→8.85, confirming [[D11]] |

## Scoring / re-judging
| script | job# | what |
|---|---|---|
| `run_v1_rescore.sh` | 647, 682 | re-score generations with the authors' **v1** judge from commit `c4c276d` ([[D12]]); idempotent, skips already-scored inputs |
| `run_harmbench_rejudge.sh` | — | re-judge existing generations with a chosen HarmBench pass |

## Related (lives with its data, not here)
`ext_benchmarks/run_ext_benchmarks.sh` (job 846) — scores patches on StrongREJECT + HarmBench; kept in
`ext_benchmarks/` alongside its datasets and READMEs so that benchmark module is self-contained.

See [`../repro_notes/REPRO_CHECKLIST.md`](../repro_notes/REPRO_CHECKLIST.md) for what each of these does and
does not reproduce, and [`../repro_notes/DISCREPANCIES.md`](../repro_notes/DISCREPANCIES.md) for the D-ids.
