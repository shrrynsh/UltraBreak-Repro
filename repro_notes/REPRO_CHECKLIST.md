# Reproducibility checklist — UltraBreak (arXiv 2602.01025) vs. this repo

A living tracker of **every experiment, ablation, and figure in the paper**, against what this
reproduction has actually run. Companion to [`DISCREPANCIES.md`](DISCREPANCIES.md) (code-vs-paper defects)
and [`FINDINGS.md`](FINDINGS.md) (the reproduction narrative). Update the status cells as jobs land.

**Detailed topic notes:** [`transfer_experiment/`](transfer_experiment/README.md) ·
[`reproducibility/`](reproducibility/README.md) · [`author_code_defects/`](author_code_defects/README.md) ·
[`ext_benchmarks/`](ext_benchmarks/README.md) · [`ablations/`](ablations/README.md) (Table 3 + Table 6) ·
[`paper_specs.md`](paper_specs.md).

**Legend:** ✅ reproduced · 🟡 partial · ⏳ in flight (job#) · ⬜ not started (infra ready) · 🚫 out of scope.

**Update 2026-08-22:** the self-chaining `jobs/pipeline.sh` ran steps 1–9 to completion (jobs
880→935→939→950→982→987→994→997→1005); step 10 (B14) is running as job 1020. This cleared A1/A2/A3, the
full transfer experiment, B11, and B13, and produced the paper-faithful retrain. The exact code-vs-paper
comparison (job 1058, self-resuming) is in flight.

---

## What is ACTUALLY running right now (2026-08-22)

Only two things are live on the cluster. **Most items in the tables below have no
job** — they are done, backlog, or out-of-scope. This section is the honest map so
the tables are not mistaken for "all in progress."

| Job | Checklist item | State |
|---|---|---|
| **1020** | **B14** token-mode retrain | 🔵 running (pipeline step 10/13) |
| **1041** | **B12** no-constraints retrain | ⏳ pending (pipeline step 11) |
| *(spawned by pipeline)* | **B16a/B16b** TV-weight 0.2 & 1.0 | ⏳ not yet submitted (steps 12–13) |
| **1058** | exact code-vs-paper comparison (144-cell factorial + code-faithful train) | ⏳ pending, self-resuming |

**Done (jobs finished):** A1, A2, A3 (job 987/994) · A4 SB+AB, transfer experiment (880/935/939) ·
A7-partial ext-benchmarks (950) · paper-faithful retrain (982) · B11 (997) · B13 (1005) · D-asr.

**Now IMPLEMENTED and pipelined (auto-chained, 2026-08-22):** the runnable backlog
was built out and chained so it runs one-by-one after the ablations:
- **`jobs/pipeline2.sh`** (chained after pipeline.sh step 13): **B15** phrase sweep
  (exact paper phrases) · **A4-MM** 809 × MM-SafetyBench × 5 targets · **C2b** LLaVA-surrogate train+eval.
- **`jobs/pipeline3.sh`** (chained after pipeline2): **A10** timing (`analysis/timing_table.py`) ·
  **C6** first-token dist (`analysis/first_token_dist.py`) · **C4** loss landscape
  (`analysis/loss_landscape.py`) · **C2a** model-size transfer (surrogates 2B/7B → victims 3B/7B/32B;
  downloads the 3 missing weight sets; attack.py + optimise_proj.py extended). Specs in [[paper_specs]].

**Still no job — genuine backlog (⬜):** A5 (FigStep/VAJM/UMK baselines — no attack code) ·
D-sr / A7-native (StrongREJECT 0–1 scorer — needs API key or SR model). C3 (qualitative patch figure) — trivial, do ad-hoc.

**Out of scope (🚫):** A6, A8 (paid proprietary/frontier APIs).

**Full auto-chain now:** `pipeline.sh (…→B16) → pipeline2 (B15, A4-MM, C2b) → pipeline3
(A10, C6, C4, C2a)`, each hop handed off as the single slot frees. Job 1058 (exact
comparison) self-chains independently.

---

## A. Main results

| # | Paper item | Status | What's missing |
|---|---|---|---|
| A1 | **Table 1** — UltraBreak ASR, authors' image, 6 open models × SafeBench | ✅ 6/6 | done (Kimi-VL/SafeBench = 66.67%, job 987) |
| A2 | **Table 1** — …× AdvBench | ✅ 6/6 | done (Qwen-VL-Chat/AdvBench = 76.15%, job 987) |
| A3 | **Table 1** — …× **MM-SafetyBench** (subsampled 520) | ✅ 6/6 | done (job 994): Qwen2-VL 37.3, Qwen2.5 48.1, Qwen-VL-Chat 56.0, LLaVA 74.6, Kimi 54.4, GLM 38.1 |
| A4 | **Table 1** — transfer matrix with **our retrained** patch (809), not just the authors' image | 🟡 SB+AB | **done for SafeBench + AdvBench** (jobs 935/939 — 809 = the projoff arm × 5 targets); MM-SafetyBench transfer still open. See [[transfer_experiment/README]] |
| A5 | **Table 1** — baselines **FigStep / VAJM / UMK** | ⬜ | none implemented; the comparative claim rests on these (E4) |
| A6 | **Table 1** — proprietary GPT-4.1-nano / Gemini-2.5-flash-lite / Claude-3-haiku | 🚫 | paid API — report as untested, not omitted |
| A7 | **Table 2** — **StrongREJECT scores** on AdvBench (UltraBreak vs VAJM/UMK) | 🟡 | ext-benchmarks (job 950) scored StrongREJECT+HarmBench prompts (native+norm) with the HarmBench judge for authors' image + 4 retrains; SR *native* 0–1 scorer still absent; baselines missing |
| A8 | **Table 9** — frontier GPT-5 / Claude-Sonnet-4.5 subset | 🚫 | paid API |
| A10 | **Table 10** — compute overhead, CE vs semantic (s/iter) | 🟡 | CE run done (B13, job 1005); per-step wall-clock in `losses.csv` — comparison table not yet compiled |

## B. Ablations (Tables 3, 5, 6)

| # | Paper item | Status | What's missing / enabling flag |
|---|---|---|---|
| B11 | **Table 3** — *w/o jailbreak image* (No-Attack) | ✅ | done (job 997), 6 models × {SafeBench, AdvBench} — the transfer baseline. LLaVA is weakly aligned (57.1/48.7 with no attack). See [[transfer_experiment/README]] |
| B12 | **Table 3** — *w/o constraints* (transforms + TV off) | ⏳ | pipeline step 11 (pending) — `--no_transforms --tv_weight 0` |
| B13 | **Table 3** — *w/o semantic loss* (CE baseline) | ✅ | done (job 1005): SafeBench 67.94%, AdvBench 15.77% (v3). Also yields A10 timing |
| B14 | **Table 3** — *w/o attention weighting* (token mode, τ→0) | ⏳ 1020 | pipeline step 10, **running** — `--loss_mode token` |
| B15 | **Table 5** — affirming-phrase sensitivity (3 phrases, GLM, SafeBench) | ⬜ | `--phrase` (cheap); not in the current pipeline |
| B16 | **Table 6** — **TV weight** {0, 0.2, 0.5, 1} on Qwen-VL + GLM | ⏳ | pipeline steps 12–13 (pending), TV 0.2 & 1.0 (0.5 ≈ 809 already); **tests [[DISCREPANCIES#D10]]** |

## C. Analysis figures (2–6)

| # | Paper item | Status | What's missing |
|---|---|---|---|
| C2a | **Fig 2a** — model-size transfer (Qwen 2B/7B/32B) | ⬜ | needs 2B & 32B Qwen weights |
| C2b | **Fig 2b** — surrogate-model choice (train on other surrogates) | ⬜ | LLaVA adapter exists + D11 fix; not trained (E7) |
| C3 | **Fig 3** — transforms/TV emergent text-like patterns (qualitative) | 🟡 | patch PNGs on disk; no side-by-side figure |
| C4 | **Fig 4/5** — loss landscapes, CE vs semantic vs τ | ⬜ | visualisation; lower priority |
| C6 | **Fig 6** — first-token prob distribution by τ | ⬜ | not done |

## D. Metrics

| # | Item | Status |
|---|---|---|
| D-asr | ASR via HarmBench (v1/v2/v3) | ✅ + [[DISCREPANCIES#D12]] judge-robustness study |
| D-sr | StrongREJECT native scorer (0–1, willingness×ability) | ⬜ needs OpenAI key or the finetuned SR evaluator model (blocks A7) |

---

## In flight
⏳ **1020** — pipeline step 10, B14 token-mode retrain (running).
⏳ **1058** — exact code-vs-paper comparison (self-resuming until its 144-cell matrix completes): the
code-faithful **retrain** (`optimise_exact.py`, authors' verbatim method) vs the paper-faithful retrain vs
the released image, across SafeBench{315,350} × AdvBench{raw,norm} × judge{v1,v3} × 6 models × {ASR, NRR}.
See [[reproducibility/README]].

## `jobs/pipeline.sh` — actual run order (self-chaining, one job at a time)
Cluster limits: MaxJobs=1, MaxSubmit=3, 24h/job. Status reflects the 2026-08-22 run.

| step | job | checklist item | kind | status |
|---|---|---|---|---|
| 1 | 880 | **TRANSFER control-find** — 809 checkpoint matched to 843's white-box (~42.5%) → step 1000 (43.17%) | score | ✅ |
| 2 | 935 | **TRANSFER SafeBench** — PROJ-OFF (809) / PROJ-ON (843) / CONTROL × 5 targets | score | ✅ |
| 3 | 939 | **TRANSFER AdvBench** — same 3 arms × 5 targets | score | ✅ |
| 4 | 950 | **ext-benchmarks** — StrongREJECT + HarmBench (native+norm), authors' image + 4 retrains | score | ✅ |
| 5 | 982 | **REPRO paper-faithful** retrain @1300 → SafeBench 64.13%, AdvBench 10.19% (v3) | train | ✅ |
| 6 | 987 | **A1 + A2** — Kimi/SafeBench, Qwen-VL-Chat/AdvBench (authors' image) | score | ✅ |
| 7 | 994 | **A3** — authors' image × MM-SafetyBench-520 × 6 models | score | ✅ |
| 8 | 997 | **B11** — No-Attack × 6 × {SafeBench, AdvBench} | score | ✅ |
| 9 | 1005 | **B13** — CE-loss retrain (`--loss ce`) — also A10 timing | train | ✅ |
| 10 | 1020 | **B14** — token-mode retrain (`--loss_mode token`) | train | ⏳ running |
| 11 | — | **B12** — no-constraints (`--no_transforms --tv_weight 0`) | train | ⏳ pending |
| 12–13 | — | **B16** — TV sweep 0.2, 1.0 (0.5 ≈ 809 already) | train | ⏳ pending |

**Steps 1–3 are the transferability verification** (does the §3.2 projection, done the right way, improve
black-box transfer?). **Result: no** — projon is the best arm only on Kimi-VL; elsewhere projoff ≈ control ≥
projon. Full analysis in [[transfer_experiment/README]].

Each training step is the 809 recipe (D11 fix, no projection, 3000 steps, seed 0) with one knob flipped,
so every ablation is single-variable against the reproduced baseline.

## Not tracked here (our contribution, in `FINDINGS.md`)
Authors' image reproduces SafeBench **exactly** (257/315 = 81.59%) and AdvBench ≈ (74.62% v1 vs 72.69%);
our retrain reproduces SafeBench (**809 = 83.49%**); training gap root-caused to [[DISCREPANCIES#D11]].
This checklist tracks only *paper experiments not yet reproduced*.

---

## Prioritised path (ties to the E1–E8 program in the `repro-tmlr-plan` memory)

1. **A1 / A2 / A3** — 2 missing authors'-image cells + MM-SafetyBench × 6. Cheap, no training (E1).
2. **B11–B16** white-box ablations — every flag exists; Table 3 + Table 6 are the highest-value
   "paper gave no numbers" content, and B16 sharpens [[DISCREPANCIES#D10]] (E5/E6).
3. **A4** — transfer the 809 patch across 5 targets × 3 datasets (true transfer reproduction).
4. **A5** baselines (FigStep/VAJM/UMK) — implement + train (E4).
5. **C2b** surrogate (E7); **C2a** needs extra Qwen weights.
6. **D-sr / A7** — implement the StrongREJECT native scorer (needs the API/model decision).
7. 🚫 **A6 / A8** — proprietary & frontier: report as untested (API-gated).

**Tally (updated 2026-08-22):** paper items = 22. Done ✅ 7 (A1, A2, A3, B11, B13, D-asr, + the transfer
experiment) · partial 🟡 4 (A4, A7, A10, C3) · in-flight ⏳ 3 (B12, B14, B16) · not-started ⬜ 4 (A5, B15,
C-figs, C2) · out-of-scope 🚫 4 (A6, A8, frontier).
The **18 authors'-image Table-1 cells** (6 models × 3 datasets) are now **all 18 done** (SafeBench + AdvBench
+ MM-SafetyBench complete). The remaining gaps are baselines (A5), the native StrongREJECT scorer (A7/D-sr),
figures, and the two pending training ablations.
