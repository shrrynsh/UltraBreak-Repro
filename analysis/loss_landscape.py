#!/usr/bin/env python
"""C4 / Fig 4-5 — loss landscape along two random image-space directions.

Paper: "sample the loss along two random directions ... in image space ... two
normalised directional vectors matching the image dimension and compute losses on a
30x30 grid over a 20x20 range" (Fig 4, 100-epoch images); Fig 5 uses fully-trained
images on a 200x200 grid. Compares CE vs semantic loss at tau=0, 0.5, ->inf.

    python analysis/loss_landscape.py --image outputs/<run>/<ckpt>.png \
        --loss semantic --tau 0.5 --grid 30 --range 20 --out analysis/out/land_<tag>

Writes <out>.npy (the loss grid) and <out>.png (contour). Deterministic directions
via --seed so CE/semantic/tau panels share the same axes.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "optimisation"))
import numpy as np, torch, pandas as pd
from PIL import Image
import torchvision.transforms as T
from qwen2_adapter import Qwen2Adapter
from utils import AblationConfig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_patch(path, device):
    img = Image.open(path).convert("RGB")
    return T.Compose([T.Resize((224, 224)), T.ToTensor()])(img).to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--loss", choices=["semantic", "ce"], default="semantic")
    ap.add_argument("--tau", type=float, default=0.5, help="use a large value (e.g. 1e6) for tau->inf")
    ap.add_argument("--grid", type=int, default=30)          # 30 (Fig4) or 200 (Fig5)
    ap.add_argument("--range", type=float, default=20.0)     # 20x20 range
    ap.add_argument("--nrows", type=int, default=3, help="avg loss over the first N corpus rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default="train_configs/safebench-tiny_jailbroken_mode_full50.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = AblationConfig(loss=args.loss, loss_mode="attention", tau=args.tau)
    adapter = Qwen2Adapter("Qwen/Qwen2-VL-7B-Instruct", patch_only=False, cfg=cfg)
    base = load_patch(args.image, adapter.device)
    rows = [r for _, r in pd.read_csv(args.corpus).fillna("").head(args.nrows).iterrows()]

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    d1 = torch.randn(base.shape, generator=g); d1 = (d1 / d1.norm()).to(adapter.device)
    d2 = torch.randn(base.shape, generator=g); d2 = (d2 / d2.norm()).to(adapter.device)

    half = args.range / 2.0
    coords = np.linspace(-half, half, args.grid)
    Z = np.zeros((args.grid, args.grid), dtype=np.float32)
    custom = (args.loss == "semantic")
    with torch.no_grad():
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                patch = torch.clamp(base + a * d1 + b * d2, 0, 1)
                tot = 0.0
                for row in rows:
                    tot += float(adapter.compute_loss(row, patch, custom_loss=custom).item())
                Z[i, j] = tot / len(rows)
            print(f"[landscape] row {i+1}/{args.grid}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out + ".npy", Z)
    plt.figure(figsize=(5, 4))
    cs = plt.contourf(coords, coords, Z, levels=30, cmap="viridis")
    plt.colorbar(cs, label="loss")
    tau_lbl = "inf" if args.tau >= 1e5 else str(args.tau)
    plt.title(f"{args.loss} loss" + (f" (tau={tau_lbl})" if custom else ""))
    plt.xlabel("direction 1"); plt.ylabel("direction 2")
    plt.tight_layout(); plt.savefig(args.out + ".png", dpi=130)
    print(f"[landscape] wrote {args.out}.npy / .png  (min={Z.min():.3f} max={Z.max():.3f})")

if __name__ == "__main__":
    main()
