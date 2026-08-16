# Reproducibility checklist — UltraBreak (arXiv 2602.01025) vs. this repo

A living tracker of **every experiment, ablation, and figure in the paper**, against what this
reproduction has actually run. Companion to [`DISCREPANCIES.md`](DISCREPANCIES.md) (code-vs-paper defects)
and [`FINDINGS.md`](FINDINGS.md) (the reproduction narrative). Update the status cells as jobs land.

**Legend:** ✅ reproduced · 🟡 partial · ⏳ in flight (job#) · ⬜ not started (infra ready) · 🚫 out of scope.

**Coverage baseline** (re-derived from `ls results/`, 2026-08-16). Authors' released `ultrabreak.png` is
scored on — SafeBench: {Qwen2-VL, Qwen2.5-VL, Qwen-VL-Chat, LLaVA-1.6, GLM-4.1V}; AdvBench: {Qwen2-VL,
Qwen2.5-VL, Kimi-VL, LLaVA-1.6, GLM-4.1V}. Our retrains (622/625/639/640/773/808/809) are **Qwen2-VL only**.
No MM-SafetyBench / baseline / proprietary / frontier results exist; StrongREJECT native scorer is absent.

---

## A. Main results

| # | Paper item | Status | What's missing |
|---|---|---|---|
| A1 | **Table 1** — UltraBreak ASR, authors' image, 6 open models × SafeBench | 🟡 5/6 | **Kimi-VL** on SafeBench |
| A2 | **Table 1** — …× AdvBench | 🟡 5/6 | **Qwen-VL-Chat** on AdvBench |
| A3 | **Table 1** — …× **MM-SafetyBench** (1680, or subsampled 520) | ⬜ 0/6 | loader + `--subsample` built (`create_attack_configs.load_mm_safetybench`), never scored |
| A4 | **Table 1** — transfer matrix with **our retrained** patch (809), not just the authors' image | ⬜ | our patch scored on Qwen2-VL only; 5 transfer targets × 3 datasets untried — the *true* transfer reproduction |
| A5 | **Table 1** — baselines **FigStep / VAJM / UMK** | ⬜ | none implemented; the comparative claim rests on these (E4) |
| A6 | **Table 1** — proprietary GPT-4.1-nano / Gemini-2.5-flash-lite / Claude-3-haiku | 🚫 | paid API — report as untested, not omitted |
| A7 | **Table 2** — **StrongREJECT scores** on AdvBench (UltraBreak vs VAJM/UMK) | ⬜ | SR *native* 0–1 scorer absent (job 846 scores SR *prompts* with the HarmBench judge — related, not Table 2); baselines missing |
| A8 | **Table 9** — frontier GPT-5 / Claude-Sonnet-4.5 subset | 🚫 | paid API |
| A10 | **Table 10** — compute overhead, CE vs semantic (s/iter) | ⬜ | per-step wall-clock already logged in `losses.csv`; needs a CE run (B13) to compare |

## B. Ablations (Tables 3, 5, 6)

| # | Paper item | Status | What's missing / enabling flag |
|---|---|---|---|
| B11 | **Table 3** — *w/o jailbreak image* (No-Attack) | 🟡 | `attack_configs/{advbench,safebench}_no_attack.csv` built; **not scored** |
| B12 | **Table 3** — *w/o constraints* (transforms + TV off) | ⬜ | `--no_transforms` / `--tv_weight 0` |
| B13 | **Table 3** — *w/o semantic loss* (CE baseline) | ⬜ | `--loss ce` (also yields A10 timing) |
| B14 | **Table 3** — *w/o attention weighting* (token mode, τ→0) | ⬜ | `--loss_mode token` |
| B15 | **Table 5** — affirming-phrase sensitivity (3 phrases, GLM, SafeBench) | ⬜ | `--phrase` (cheap) |
| B16 | **Table 6** — **TV weight** {0, 0.2, 0.5, 1} on Qwen-VL + GLM | ⬜ | `--tv_weight`; **high value — directly tests [[DISCREPANCIES#D10]]**; paper claims peak at 0.2 |

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
⏳ **843** projfix · **844** doublesquash · **846** ext-benchmarks (StrongREJECT + HarmBench prompts,
native+norm, our 5 patches, HarmBench-judged → not the same as A7's native SR scorer).

## Scheduled — `jobs/pipeline.sh` (self-chaining, one job at a time)
A self-chaining pipeline works these items in priority order within the cluster limits (MaxJobs=1,
MaxSubmit=3, 24h/job). Seeded by `jobs/seed_pipeline.sh` when a slot frees.

| step | checklist item | kind |
|---|---|---|
| 1 | **A1 + A2** — Kimi/SafeBench, Qwen-VL-Chat/AdvBench (authors' image) | score |
| 2 | **A3** — authors' image × MM-SafetyBench-520 × 6 models | score |
| 3 | **B11** — No-Attack (white.jpeg, no phrase) × 6 × {SafeBench, AdvBench} | score |
| 4 | **B13** — CE-loss retrain (`--loss ce`) — also A10 timing | train |
| 5 | **B14** — token-mode retrain (`--loss_mode token`) | train |
| 6 | **B12** — no-constraints (`--no_transforms --tv_weight 0`) | train |
| 7–8 | **B16** — TV sweep 0.2, 1.0 (0.5 ≈ 809 already have) | train |
| 9 | **A4** — transfer 809 patch × 5 targets × {SafeBench, AdvBench} | score |

Each training step is the 809 recipe (D11 fix, no projection, 3000 steps, seed 0) with one knob flipped,
so every ablation is single-variable against the reproduced baseline. Flip the status cells above to ✅/🟡
as steps land.

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

**Tally:** paper items = 22. Done ✅ 1 · partial 🟡 5 · not-started ⬜ 12 · out-of-scope 🚫 4.
Of the 18 authors'-image Table-1 cells (6 models × 3 datasets), **10 done, 8 missing** (2 open cells + all
6 MM-SafetyBench).
