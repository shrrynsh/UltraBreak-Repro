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

## 5. Consistency with SafeBench / AdvBench

Do these external scores agree with the headline SafeBench/AdvBench numbers for the
same patches? Here is every benchmark side by side (ASR %, v3 judge, white-box
Qwen2-VL):

| Patch | SafeBench | AdvBench | StrongREJECT nat / norm | HarmBench nat / norm |
|---|---|---|---|---|
| **809** | **83.49** | 39.04 | 21.7 / 26.2 | 22.0 / 36.0 |
| **625** | 48.25 | 22.31 | 29.4 / 23.0 | 26.5 / 31.5 |
| **622** | 29.84 | 1.73 | 6.7 / 7.4 | 17.5 / 6.5 |
| **773** | 23.49 | 1.35 | 8.6 / 6.1 | 18.0 / 8.0 |
| **authors** | 78.73* | 67.69* | 29.7 / 35.5 | 28.5 / 42.5 |

\*authors' white-box v3 from `FINDINGS.md` (clean-315 SafeBench / normalized
AdvBench). On the `_ultrabreak` configs on disk, authors' AdvBench reads only ~9% —
a different prompt set/form, i.e. the raw-vs-norm confound, not a contradiction.

### Ranking table (1 = best / highest ASR, per benchmark)
| Patch | SafeBench | AdvBench | SR nat | SR norm | HB nat | HB norm | **avg rank** |
|---|---|---|---|---|---|---|---|
| **authors** | 2 | 1 | 1 | 1 | 1 | 1 | **1.2** |
| **809** | 1 | 2 | 3 | 2 | 3 | 2 | **2.2** |
| **625** | 3 | 3 | 2 | 3 | 2 | 3 | **2.7** |
| **622** | 4 | 4 | 5 | 4 | 5 | 5 | **4.5** |
| **773** | 5 | 5 | 4 | 5 | 4 | 4 | **4.5** |

**Overall generalization order:** authors > 809 > 625 > (622 ≈ 773).

### Verdict: broadly consistent, with two informative exceptions
**Consistent ✅** — the strong/weak tiering holds on all six benchmarks. authors and
809 are top; 622 and 773 are bottom everywhere. Nothing flips a weak patch to the
top or vice-versa.

**Two real deviations ⚠️**
1. **625 over-performs externally.** Mid-pack on white-box AdvBench (22.3) but rank
   2 on StrongREJECT-native and HarmBench-native — it rivals the authors' image and
   beats 809 there. It generalizes *better* than its AdvBench score predicts.
2. **809's lead is SafeBench-specific.** It is rank 1 on white-box SafeBench (even
   above authors), but rank 2–3 everywhere else and **below the authors' image on
   every external benchmark**. Its SafeBench peak does not carry over — exactly the
   documented "809 overfits SafeBench-Tiny" finding ([[../DISCREPANCIES#D11]]).

Both exceptions are already part of the reproduction story, so the scores hang
together — the external benchmarks confirm rather than contradict the headline
numbers.

## 6. Bottom line

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
