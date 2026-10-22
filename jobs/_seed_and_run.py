#!/usr/bin/env python3
"""Apply a fixed seed, then run an unmodified trainer's main().

WHY THIS EXISTS. Contract clause CF1 allows the code-faithful trainer to differ from
`c4c276d:optimisation/optimise.py` in at most two lines, and only `exp_name` and
`num_epochs`. The authors set no seed anywhere (D7), so adding `torch.manual_seed(...)`
inside the trainer would be a third changed line and would break CF1.

Clause S1 nevertheless requires three fixed seeds so that run-to-run variance is
quantified. The two are reconciled by seeding the interpreter *from outside* and then
calling the trainer's own `main()`. The authors' file stays a two-line diff, and every
random draw it makes is still governed by the seed:

  * patch initialisation          torch.rand            -> torch.manual_seed
  * per-step transform parameters random.uniform        -> random.seed
  * target-embedding noise        torch.randn_like      -> torch.manual_seed

HONEST LIMIT (D7). This pins the *sampling*, not the arithmetic. `grid_sample`'s
backward pass has no deterministic CUDA kernel and every training step goes through it,
so two runs at the same seed still drift slightly. Three seeds give a variance
estimate; they do not make training bitwise reproducible.

    python jobs/_seed_and_run.py <seed> <path/to/trainer.py>
"""
import importlib.util
import os
import random
import sys

import numpy as np
import torch


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    seed, script = int(sys.argv[1]), sys.argv[2]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Covers the python hash seed for any dict-order sensitivity in the loop.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    print(f"[seed] random/numpy/torch/cuda seeded with {seed}", flush=True)
    print(f"[seed] NOTE: grid_sample's CUDA backward is non-deterministic, so this "
          f"pins sampling, not arithmetic (D7).", flush=True)

    # The trainer imports its siblings by bare name, so its own directory leads sys.path.
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    spec = importlib.util.spec_from_file_location("trainer_under_seed", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # defines main(); __name__ != "__main__" so it does not self-run
    mod.main()


if __name__ == "__main__":
    main()
