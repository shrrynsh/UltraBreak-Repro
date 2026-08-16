"""
optimise_paper.py — paper-faithful UltraBreak training (Table 4, paper-exact).
==============================================================================
Self-contained: imports ONLY the frozen modules in this folder's optimisation/,
never the working ../../optimisation/ (so ablation edits there cannot affect
this reproduction, and vice versa).

This pins the validated engine (optimise_proj.py, copied here) to the paper's
described configuration, with every value traced to its source. It is the
literal reading of §3.2 + Eq 8-10 + Table 4:

  - §3.2 / Eq 10 projection ON: the model sees x_proj = clip(gamma*x + beta,0,1),
    gamma = CLIP_STD, beta = CLIP_MEAN; the deployable image is x_proj.
  - patch normalised into CLIP space before the vision encoder (DISCREPANCIES
    D11 fix) — REQUIRED for the method to be on the scale the processor expects;
    the released code omits it, which is the units bug the study documents.
  - semantic attention loss (Eq 8), tau = 0.5, embedding noise eps = 1e-4.
  - random transforms ON: scale U(0.8,1.2), rotation U(-15,15) deg.
  - lambda_TV = 0.5 on the raw latent x (Eq 10 L_TV(x)).
  - Adam, lr = 0.01, **1300 iterations** (Table 4), Qwen2-VL-7B surrogate.
  - random *image* initialisation mapped back through the projection.

DOCUMENTED GAP (per the user's decision, run paper-exact and record the gap):
Table 4 says 1300 iterations. We found 1300 UNDER-reproduces (job 773: SafeBench
23.49%) while ~3000 reproduces (job 809: 83.49%). This script runs the paper's
1300 for fidelity; see reproducibility/README.md and repro_notes/DISCREPANCIES.md
(D11) for the budget finding. Do NOT silently bump to 3000 here — that would stop
being paper-exact.

  cd <repo-root> && python reproducibility/paper_faithful/optimise_paper.py
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer THIS folder's frozen modules over the working tree.
sys.path.insert(0, os.path.join(HERE, "optimisation"))
import optimise_proj  # noqa: E402  (the frozen engine copied beside this file)
from utils import AblationConfig  # noqa: E402  (frozen; its defaults ARE Table 4)


# ---- paper-exact configuration, every value traced to its source -------------
PAPER = argparse.Namespace(
    train_config="train_configs/safebench-tiny_jailbroken_mode_full50.csv",  # SafeBench-Tiny, 50 (D2-corrected)
    exp_name="repro_paper_faithful_1300",
    num_epochs=1301,        # epochs 0..1300 -> saves 1300.png; Table 4 "Iterations = 1300"
    base_epoch=0,
    init_space="image",     # random IMAGE init mapped through the projection ("random initialisation")
    no_projection=False,    # §3.2 projection ON
    clamp_latent=False,     # exactly ONE constraint, inside the projection (not an extra latent clamp)
    raw_patch_injection=False,  # D11 fix ON: patch normalised into CLIP space
    seed=0,                 # paper does not seed (D7); we fix one for a reproducible artifact
    lr=0.01,                # Table 4
    tv_weight=0.5,          # Table 4 lambda_TV
    l2_weight=0.0,          # not in the paper; inert
    surrogates="qwen",      # Qwen2-VL-7B-Instruct, the paper's surrogate
    loss="semantic",        # Eq 8
    loss_mode="attention",  # Table 4
    tau=0.5,                # Table 4
    embed_noise=1e-4,       # Table 4 epsilon
    pos_alpha=0.01,         # positional-encoding weight
    no_transforms=False,    # transforms ON (Table 4: scale 0.8-1.2, rotation +-15)
)


def _assert_table4():
    """Fail loud if anyone edits a value away from the paper's Table 4."""
    cfg = AblationConfig()  # frozen defaults
    assert (cfg.tau, cfg.embed_noise, cfg.loss, cfg.loss_mode) == (0.5, 1e-4, "semantic", "attention")
    assert cfg.scale_range == (0.8, 1.2) and cfg.rotation_range == (-15.0, 15.0)
    assert cfg.normalise_patch is True and cfg.transforms is True
    assert (PAPER.lr, PAPER.tv_weight, PAPER.num_epochs) == (0.01, 0.5, 1301)
    assert PAPER.no_projection is False and PAPER.raw_patch_injection is False
    print("[paper-faithful] Table-4 / §3.2 config verified against the paper.")


if __name__ == "__main__":
    _assert_table4()
    print("[paper-faithful] NOTE: running the paper-exact 1300 iterations. This "
          "under-reproduces (job 773 ~23%); ~3000 reproduces (job 809 ~83%). "
          "This gap is the documented finding, not a bug — see reproducibility/README.md.")
    optimise_proj.main(PAPER)
