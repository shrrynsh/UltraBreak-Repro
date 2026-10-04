# `aaai_ugc/` — query-distribution-shift study

A small, **inference-only** study layered on the UltraBreak reproduction, gathering preliminary
evidence for a research proposal:

> **How universal are "universal" image jailbreaks?** They are trained on a ~50-query corpus and
> evaluated on held-out queries from the *same* benchmark and style. Does success hold when the query
> distribution shifts in format, source or harm category — or is the image a **format-bound trigger**
> rather than a general refusal-off switch?

**No training. No new patches. No external APIs.** Only existing attack artifacts re-evaluated on
existing benchmark prompts. Defensive measurement research.

## Start here
| file | what |
|---|---|
| [`PLAN.md`](PLAN.md) | what was found in the repo, every design decision, and the five places the task's assumptions had to be reconciled with what the code actually does |
| [`SUMMARY.md`](SUMMARY.md) | **the results** — one section per experiment, with Wilson CIs, exact McNemar p-values, and a conservative "what this supports / what it does not" per experiment |

## Experiments
| id | question | size | compute |
|---|---|---|---|
| **E0** | Does success already fall off for queries unlike the training corpus? Category degeneracy, ASR vs semantic distance, response-shape analysis. | re-analysis of 4 existing runs | **CPU, done** |
| **E1** | Format sweep — does the patch's contribution depend on prompt format with intent held constant? *The central test.* | 150 AdvBench intents × 3 formats × 5 image conditions = 2,250 | GPU ≈ 2.4 h |
| **E2** | Forced-prefix ablation — how much survives without the "[Jailbroken Mode]" instruction? | SafeBench-315 × 2 × 2 = 1,260 | GPU ≈ 1.3 h |
| **E3** | Source shift with format held constant — does a benchmark gap remain once phrasing is normalised? | 1,035 new (authors' arms reused) | GPU ≈ 1.2 h |

## Layout
```
PLAN.md  SUMMARY.md  README.md
code/     stats.py          Wilson CI, exact McNemar, IRLS logistic — numpy only, self-tested
          metrics.py        NRR / prefix-then-refuse / list-mode, on the EXTRACTED answer
          embed.py          all-MiniLM-L6-v2 via transformers (no new packages)
          build_configs.py  every config + the shuffled control image
          run_E0.py         E0 analysis (CPU)
          analyse_shift.py  E1/E2/E3 tables, uplift, McNemar, figures
jobs/     shift_dryrun.sh   10-row validation of every code path
          shift_study.sh    self-chaining E1 -> E2 -> E3, per-config resume markers
configs/  22 configs + E1_ids.txt (frozen) + per-config manifests + INDEX.json
results/  E0/  gen/  dryrun/  E1_*.csv  E2_*.csv  E3_*.csv  .markers/
figures/  E0b_asr_vs_similarity.png  E1_format_sweep.png
images/   shuffled_ultrabreak_seed0.png   (authors' patch, pixels permuted, seed 0)
```

## Ground rules this study follows
- **Nothing existing is modified.** No existing result, config or patch is touched; the repo's
  modules are *imported*, never forked. `build_configs.py` proves its one new builder is
  byte-identical to `create_attack_configs.build_attack_config` wherever both apply.
- **Env isolation.** Everything runs in the pinned `repro` venv (torch 2.5.1+cu124, transformers
  4.51.3), asserted at job start. No package was installed into it — the statistics and the sentence
  embeddings are implemented against what is already there, because that env produced every existing
  number in the study.
- **Protocol held fixed** at jobs 809/683: Qwen2-VL-7B surrogate, `max_new_tokens=512` (asserted —
  the D14 bump to 4096 is GLM-only and must not leak here), TPG template unchanged, SafeBench-315 and
  AdvBench-520 as canonical, judge **v3** everywhere plus **v1** on headline rows. Every table names
  its judge.
- **Paired design.** Conditions share identical query lists in identical order; uplift is a row-wise
  difference, McNemar is exact.
- **No completion text** in any summary, table, figure or markdown file here. Metrics, counts,
  lengths, category names and query IDs only. Raw generations stay in the pipeline's own CSVs.
- **Prompt variants** are mechanical rewordings of existing benchmark rows. No new harmful request is
  written, and no LLM is used to produce them.

## Reproducing
```bash
source repro/bin/activate
repro/bin/python aaai_ugc/code/stats.py          # self-tests
repro/bin/python aaai_ugc/code/metrics.py
repro/bin/python aaai_ugc/code/build_configs.py  # rebuild all 22 configs + control image
repro/bin/python aaai_ugc/code/run_E0.py         # E0, CPU, ~5 s
sbatch aaai_ugc/jobs/shift_dryrun.sh             # validate every path at 10 rows
sbatch aaai_ugc/jobs/shift_study.sh              # E1 -> E2 -> E3
repro/bin/python aaai_ugc/code/analyse_shift.py  # tables + figures (safe to run on partial results)
```
