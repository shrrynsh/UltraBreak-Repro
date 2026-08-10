"""
test_reproducibility.py
-----------------------
Check that --seed actually pins the three stochastic parts of training.

The released code seeds nothing, so every run differs in its patch
initialisation, its per-step transform schedule (angle / scale / placement) and
its target-embedding noise.  That makes run-to-run variance unquantifiable and
every reported delta n=1.  This asserts the seed controls all three.

Runs on CPU in a second; no model download, no GPU.

    python optimisation/test_reproducibility.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from optimise_proj import initialise_patch
from utils import apply_random_patch, set_seed


def patch_init(seed, init_space="unit"):
    set_seed(seed)
    return initialise_patch("cpu", 32, None, init_space).detach().clone()


def transform_schedule(seed, steps=8):
    """The sequence of composites a run would produce, as a fingerprint."""
    set_seed(seed)
    canvas = torch.full((3, 96, 96), -1.0)
    patch = torch.rand(3, 32, 32)
    return [
        float(apply_random_patch(canvas, patch).sum())
        for _ in range(steps)
    ]


def embedding_noise(seed, n=4):
    set_seed(seed)
    return [float(torch.randn(16).sum()) for _ in range(n)]


def check(name, same_seed_equal, diff_seed_equal):
    ok = same_seed_equal and not diff_seed_equal
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not same_seed_equal:
        print("         same seed did not reproduce")
    if diff_seed_equal:
        print("         different seeds produced identical output — seed is not reaching this")
    return ok


def main():
    print("Seed control over the three stochastic sources:")
    results = []

    a, b, c = patch_init(0), patch_init(0), patch_init(1)
    results.append(check("patch initialisation", torch.equal(a, b), torch.equal(a, c)))

    a, b, c = patch_init(0, "image"), patch_init(0, "image"), patch_init(1, "image")
    results.append(check("patch initialisation (--init_space image)",
                         torch.equal(a, b), torch.equal(a, c)))

    a, b, c = transform_schedule(0), transform_schedule(0), transform_schedule(1)
    results.append(check("transform schedule (angle / scale / placement)", a == b, a == c))

    a, b, c = embedding_noise(0), embedding_noise(0), embedding_noise(1)
    results.append(check("target-embedding noise", a == b, a == c))

    # --no_transforms must be deterministic by construction, not merely seeded.
    canvas, patch = torch.full((3, 96, 96), -1.0), torch.rand(3, 32, 32)
    set_seed(0)
    fixed_a = apply_random_patch(canvas, patch, transforms_enabled=False)
    set_seed(999)
    fixed_b = apply_random_patch(canvas, patch, transforms_enabled=False)
    results.append(check("--no_transforms is seed-independent",
                         torch.equal(fixed_a, fixed_b), False))

    print()
    if all(results):
        print("All checks passed.")
        print(
            "\nNote: this pins the sampling, not the arithmetic. grid_sample's backward\n"
            "pass has no deterministic CUDA kernel, so two seeded GPU runs will still\n"
            "drift slightly. Seeding removes the dominant variance; it does not make\n"
            "training bitwise reproducible, and the write-up should not claim it does."
        )
        return 0
    print("FAILURES — seed is not reaching every stochastic source.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
