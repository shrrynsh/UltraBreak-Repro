# Authors' Released Code — Defects Review & Analysis

**Date: 2026-08-22.** Plain-language catalogue of the bugs in the authors' released
code that affect the paper's numbers. Two sources: (a) our own reproduction study
(commit `c4c276d`), and (b) an independent code review of the live released repo
(HEAD `c024bb4`). Where both agree, the finding is doubly confirmed.

---

## 1. The three that matter most

| # | Defect | Where | Effect in plain words |
|---|---|---|---|
| **D11** | **The grey-band bug** | `qwen2_adapter.py` `compute_loss` | The training image is optimised inside a tiny 27%-wide grey band, but shipped spanning full color. **The image trained and the image shipped are not the same image.** |
| **D1** | **Dead projection** | `utils.py` / `optimise.py` | The paper's Section 3.2 projection function `project_patch()` is **never called** — it is dead code. Only `clamp(x,0,1)` runs. |
| **D12** | **Judge contamination** | `evaluate.py` line 30 | The judge accidentally reads the **harmful prompt back into the model's "answer,"** so it mis-scores refusals as successes and **inflates ASR**. |

### D11 — the grey-band bug (the big one)
- **What happens:** the adversarial patch is injected **raw** (values 0–1) into an
  image tensor that has *already been CLIP-normalised* (values roughly −2 to +2).
  It is a silent units mismatch — nothing errors.
- **Why it's fatal:** with the code's `clamp(x,0,1)`, training can only ever make
  the model perceive pixels in the range **[0.449, 0.718]** — a 27%-wide grey band.
  But the saved PNG spans the full [0,255] range.
- **The proof:** a patch scoring 67.69% loses **87% of its power** (down to 8.85%)
  when squeezed into the band — pattern untouched, only the value-range changed.
  **Conclusion: the released training code cannot produce the released image.**
  Either the authors trained with normalization in place, or the artifact came from
  a different pipeline.
- **Confirmed fixed:** job 809 (fix + 3000 steps) → SafeBench **83.49%**, reproducing
  the paper. So the band was the blocker.

### D1 — the projection is dead code
The paper's central novelty (Section 3.2 projection) has **zero call sites** in the
released code. Our transfer experiment then tested whether turning it on even helps
— and it mostly does not (see [[../transfer_experiment/README]]).

### D12 — the judge reads the prompt back in
The judge's `extract_assistant_text` looks for the word `assistant` to find where
the model's reply starts. But Qwen's system prompt already says *"You are a helpful
**assistant**."* — so it grabs the wrong spot and includes the **user's harmful
instruction** in the text sent to the classifier. This inflates ASR. This is exactly
why the authors' judge (v1) scores higher than our fixed judge (v3).

---

## 2. Independent code-review findings (live repo, HEAD `c024bb4`)

A separate detailed review of the released repository confirmed D11/D12 and found
more. Ranked by how much they affect results:

| # | Defect | File | Plain-language effect |
|---|---|---|---|
| 1 | **Judge reads prompt back in** (= D12) | `evaluate.py:30` | Inflates ASR for the main Qwen models. **Confirmed against their own test CSV.** |
| 2 | **LLaVA adapter has no `return`** | `llava16_adapter.py:71` | `process_target` returns `None`, so **any run that includes the LLaVA surrogate crashes immediately** on unpack. |
| 3 | **General judge is inverted** | `general_judge.py:68` | Returns "success" for **any** output that isn't exactly the token `no`. "No, this is a refusal." counts as a success. **Systematically inflates ASR.** |
| 4 | **Causal mask flipped** | `utils.py:210` | The core semantic loss uses `torch.triu` (future tokens) where the comment says lower-triangle (past tokens). Each prediction is aligned to the **wrong** target positions. |
| 5 | **TV/L2 penalties counted N times** | `optimise.py:129` | The regularizers are added inside the per-row loop, so their strength **secretly scales with dataset size** — the optimiser minimises a different objective than the one printed. |
| 6 | **Loss log off-by-one / empty** | `optimise.py:156` | `losses.csv` is written before the current epoch is recorded — empty at epoch 0, always up to 20 epochs stale. |
| 7 | **Hardcoded `device='cuda'`** | `utils.py:134` | The positional-encoding tensor is forced to CUDA, so the default loss **crashes on CPU/MPS** even though the code tries to select those devices. |
| 8 | **Refusal check on raw text** | `evaluate.py:87` | Refusal/NRR is computed over the contaminated response text (same root cause as D12), so NRR is also unreliable. |
| 9 | **GLM judged differently** | `evaluate.py:259` | GLM's output keeps `<think>` reasoning and special tokens while other models are cleaned — so **GLM's ASR is not comparable** to the rest. |
| 10 | **Dead early-stopping + wasted clone** | `optimise.py:102` | `patience == num_epochs`, so early-stopping never fires; a full patch is cloned every improving epoch for nothing. |

---

## 3. What these defects mean for the paper

| Claim in paper | Affected by | Verdict |
|---|---|---|
| Headline ASR (e.g. 81.59%) | D12, #1, #3, #8 | Judge bugs inflate ASR; real numbers under a clean judge are a few points lower. |
| "Section 3.2 projection drives transfer" | D1 | The projection is dead code in the release; our test shows it does not drive transfer anyway. |
| Retrained numbers reproduce | D11 | Not from the released code — the band bug makes it impossible; a fix is needed to reproduce. |
| LLaVA used as a surrogate | #2 | The LLaVA path crashes as shipped — that surrogate could not have been run as released. |
| Semantic loss as described | #4, #5 | The implemented objective differs from the described one (flipped mask, size-scaled penalties). |

---

## 4. Severity summary

- **Blockers (make results impossible to reproduce as-is):** D11 (band), #2 (LLaVA crash).
- **Metric-inflating (make numbers look better than reality):** D12/#1, #3, #8, #9.
- **Method-altering (the code does something different from the paper):** D1, #4, #5.
- **Cosmetic / robustness:** #6, #7, #10.

Full audit trail and paper quotes are in [`../DISCREPANCIES.md`](../DISCREPANCIES.md).
Questions to raise with the authors are in [`../author_email_draft.md`](../author_email_draft.md).
