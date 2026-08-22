# External Benchmarks — Review & Analysis

**Date: 2026-08-22.** Plain-language write-up of the out-of-distribution
generalization test (pipeline step 4, job 950). All numbers are Attack Success
Rate (ASR %) under our v3 judge, white-box on Qwen2-VL-7B.

---

## 1. What this experiment asks

The patches were trained on SafeBench-Tiny. Do they still work on **completely
different harmful-prompt benchmarks they never saw during training?** We test two:

| Benchmark | Size | What it is |
|---|---|---|
| **StrongREJECT** | 313 prompts | a curated, hard jailbreak benchmark |
| **HarmBench-standard** | 200 behaviors | the standard HarmBench behavior set |

Each is run in two prompt forms: **native** (original wording) and **norm**
(normalized to "Steps to …" form). This is a **generalization** check — a good
attack should carry beyond the exact benchmark it was tuned on.

---

## 2. The five patches being compared

| Tag | Patch file | Plain description | Headline SafeBench |
|---|---|---|---|
| **authors** | `ultrabreak.png` | the released image (reference) | 81.59% |
| **622** | `safebench_tiny_full50_repro/1300.png` | buggy baseline (D11 band, no projection) | weak |
| **625** | `safebench_tiny_full50_proj_repro/1300.png` | projection variant @1300 | weak |
| **773** | `full50_normfix_noproj_seed0/1300.png` | D11-fixed but only @1300 (under-trained) | 23% |
| **809** | `full50_normfix_lr0p01_s3000/3000.png` | **D11-fixed @3000 — our SafeBench reproduction** | 83.49% |

---

## 3. Results

### StrongREJECT ASR % (n=313)
| Patch | native | norm |
|---|---|---|
| **authors** | **29.71** | **35.46** |
| 809 | 21.73 | 26.20 |
| 625 | 29.39 | 23.00 |
| 773 | 8.63 | 6.07 |
| 622 | 6.71 | 7.35 |

### HarmBench-standard ASR % (n=200)
| Patch | native | norm |
|---|---|---|
| **authors** | **28.5** | **42.5** |
| 809 | 22.0 | 36.0 |
| 625 | 26.5 | 31.5 |
| 773 | 18.0 | 8.0 |
| 622 | 17.5 | 6.5 |

---

## 4. What it tells us (in simple words)

1. **The authors' released image generalizes best** on both benchmarks. Its power
   is real and carries beyond SafeBench — consistent with the reproducibility story
   that their *artifact* is genuinely strong.

2. **Our best retrain (809) generalizes second-best, and respectably** — about
   6–10 points behind the authors' image (e.g. HarmBench 22/36 vs 28.5/42.5). This
   is a **positive** result: the patch that reproduces SafeBench is *not* overfit to
   SafeBench; it transfers to unseen benchmarks too.

3. **Generalization tracks training quality.** The two weakest patches — **622**
   (buggy baseline) and **773** (D11-fix but under-trained @1300) — collapse here
   too (~6–18%). The band bug and under-training that hurt the headline numbers hurt
   generalization the same way. Nothing surprising, but it confirms the pattern.

4. **625 is a mild outlier.** A @1300 projection patch that generalizes better than
   expected (29.4 SR-native, even edging 809). It's a single-seed point, worth a
   footnote, not a conclusion.

5. **Prompt form ("norm") is a real confound — again.** Normalization *helps* the
   strong patches (authors HarmBench 28.5 → 42.5; 809 22 → 36) but *hurts* the weak
   ones (773 18 → 8). A strong patch benefits from the cleaner "Steps to …"
   phrasing; a weak patch fails either way and the rephrasing just adds noise. This
   is the same prompt-form sensitivity flagged in [[../reproducibility/README]].

---

## 5. Bottom line

**Generalization ranks the same as reproduction quality:**
`authors' image > 809 (our reproduction) > under-trained / buggy retrains`.

Because 809 lands within ~6–10 points of the authors' image on two benchmarks it
never trained on, the reproduction gap is one of **degree, not a wholesale failure**
— our retrained patch really does learn a transferable jailbreak pattern, just a
somewhat weaker one than the released artifact.

**How it was produced:** `ext_benchmarks/run_ext_benchmarks.sh` (pipeline step 4,
job 950), Qwen2-VL-7B white-box, v3 HarmBench judge. Note this is *not* the paper's
Table 2 — that needs the native StrongREJECT 0–1 scorer (willingness × ability),
which is still absent (see [[REPRO_CHECKLIST]] item A7 / D-sr).
