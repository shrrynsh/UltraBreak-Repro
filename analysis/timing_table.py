#!/usr/bin/env python
"""A10 / Table 10 — compute overhead: CE loss vs semantic (UltraBreak) loss.

Paper reports, on an H100, averaged over 5 iterations:  CE = 6.44s, UltraBreak = 7.55s
(ratio 1.17x). Absolute times depend on the GPU; the REPRODUCIBLE quantity is the
ratio. This times a full training step (forward + backward) for both losses on our
GPU, mirroring the adapter's own compute_loss path.

    python analysis/timing_table.py [--iters 5] [--warmup 2]
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "optimisation"))
import torch, pandas as pd
from qwen2_adapter import Qwen2Adapter
from utils import AblationConfig

def time_loss(adapter, row, patch, custom_loss, iters, warmup):
    times = []
    for i in range(warmup + iters):
        patch_i = patch.clone().detach().requires_grad_(True)
        torch.cuda.synchronize()
        t0 = time.time()
        loss = adapter.compute_loss(row, patch_i, custom_loss=custom_loss)
        loss.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        if i >= warmup:
            times.append(dt)
    return sum(times) / len(times)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--corpus", default="train_configs/safebench-tiny_jailbroken_mode_full50.csv")
    args = ap.parse_args()

    cfg = AblationConfig()  # paper defaults: semantic, attention, tau=0.5
    adapter = Qwen2Adapter("Qwen/Qwen2-VL-7B-Instruct", patch_only=False, cfg=cfg)
    row = pd.read_csv(args.corpus).fillna("").iloc[0]
    patch = torch.rand(3, 224, 224, device=adapter.device)

    ce = time_loss(adapter, row, patch, custom_loss=False, iters=args.iters, warmup=args.warmup)
    sem = time_loss(adapter, row, patch, custom_loss=True, iters=args.iters, warmup=args.warmup)

    print("\n===== A10 / Table 10 — compute overhead (this GPU) =====")
    print(f"{'Method':26s} {'Time (s)':>10s}")
    print(f"{'Cross-Entropy Loss':26s} {ce:>10.2f}")
    print(f"{'UltraBreak (semantic)':26s} {sem:>10.2f}")
    print(f"\nratio semantic/CE = {sem/ce:.2f}x   (paper: 7.55/6.44 = 1.17x on H100)")
    print(f"(mean of {args.iters} iters, {args.warmup} warmup; forward+backward)")

if __name__ == "__main__":
    main()
