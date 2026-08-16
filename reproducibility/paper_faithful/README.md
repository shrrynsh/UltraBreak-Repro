# `paper_faithful/` — the paper's described method, implemented cleanly

A self-contained implementation of UltraBreak **as the paper describes it**, isolated from the working
tree. `optimise_paper.py` pins the validated engine (`optimisation/optimise_proj.py`, copied here) to
Table 4 / §3.2 / Eq 8-10, with a hard assertion (`_assert_table4`) that fails if any value drifts.

```
optimisation/     frozen copies: optimise_proj.py (engine), qwen2_adapter.py, llava16_adapter.py,
                  base_adapter.py, utils.py  (AblationConfig defaults ARE Table 4)
optimise_paper.py paper-exact entry point (Qwen2-VL surrogate, 1300 iterations)
```

Isolation is verified: `optimise_paper.py` puts this folder's `optimisation/` first on `sys.path`, so it
imports the frozen engine, never `../../optimisation/`. Editing the working tree cannot change this run.

## Table 4 / equations → code (every value traced)

| paper | value | where |
|---|---|---|
| Iterations | **1300** | `optimise_paper.py` `num_epochs=1301` (0..1300) |
| Learning rate η, optimiser | 0.01, Adam | `optimise_paper.py` `lr`; `optimise_proj.py` |
| λ_TV | 0.5 on raw `x` (Eq 10 `L_TV(x)`) | `optimise_paper.py` `tv_weight`; `utils.total_variation` |
| Scale / rotation | U(0.8,1.2) / U(−15,15)° | `AblationConfig` defaults; `utils.apply_random_patch` |
| Semantic loss | Eq 8, attention mode | `AblationConfig.loss='semantic', loss_mode='attention'` |
| Attention temperature τ | 0.5 | `AblationConfig.tau` |
| Embedding noise ε | 1e-4 | `AblationConfig.embed_noise` |
| §3.2 projection | `x_proj = clip(γx+β, 0,1)`, γ=CLIP_STD, β=CLIP_MEAN | `optimise_proj.project_to_constrained_space` (projection ON) |
| CLIP normalisation of the patch | present (D11 fix) | `qwen2_adapter.compute_loss` (`normalise_patch=True`) |
| Affirming phrase / TPG | `[Jailbroken Mode]` | config generation (`../../create_attack_configs.py`) |
| Surrogate | Qwen2-VL-7B-Instruct | `optimise_paper.py` `surrogates='qwen'` |

Two points where the paper is under-specified, and the faithful choice made (see DISCREPANCIES for the
full argument): **normalisation** is included because `pixel_values` are CLIP-normalised and the method is
otherwise on the wrong scale (D11); **random init** is read as a random *image* mapped through the
projection (`init_space='image'`, D-O2). Both are documented, not silent.

## The documented gap (paper-exact, per the fidelity decision)

Table 4 states **1300** iterations. Run at 1300, this configuration **under-reproduces** — the same
regime as job 773 (SafeBench 23.49%). It reproduces only near **~3000** iterations (job 809: 83.49%).
Per the agreed decision, this folder runs the paper's **1300** for fidelity and records the gap here; it
does **not** bump to 3000 (that would stop being paper-exact). The budget finding is DISCREPANCIES D11;
the 3000-iteration result is produced by the working-tree ablation pipeline, not here.

## Run

```bash
cd <repo-root>
python reproducibility/paper_faithful/optimise_paper.py     # or: sbatch jobs/run_repro_paper_faithful.sh
# -> outputs/repro_paper_faithful_1300/1300.png
```
