#!/usr/bin/env python
"""C3 / Figure 3 — do random transforms + TV really produce structured, text-like patterns?

The paper shows adversarial patches side by side and asserts qualitatively that its
input-space constraints (random scale/rotation/location transforms + a total-variation
penalty) yield "structured" and "text-like" patterns that generalise, where
unconstrained optimisation yields high-frequency noise.

A contact sheet alone cannot settle that, so this script pairs the images with three
measured texture statistics, computed with numpy only (the pinned `repro` env has no
scipy and must not gain packages):

  TV/px        mean absolute first difference. Low = locally smooth.
  mid-band     fraction of FFT power in the mid radial band (8-32 cycles/image).
               Glyph-like strokes live here; pixel noise is flat-to-high.
  edge density fraction of pixels whose Sobel magnitude exceeds 10% of the max.

Written so the claim is falsifiable: if the constrained patches are not smoother and
not more mid-band-concentrated than the unconstrained one, the figure says so.

    repro/bin/python analysis/fig3_patterns.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "analysis", "out")
FIGS = os.path.join(REPO, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# (label, path, short note) — constrained runs first, then the ablations that remove pieces
PATCHES = [
    ("authors\nultrabreak.png", "outputs/ultrabreak.png", "released artifact"),
    ("ours 809\nsem+constraints", "outputs/full50_normfix_lr0p01_s3000/3000.png", "D11 fix, 3000 steps"),
    ("no constraints\n(B12)", "outputs/full50_abl_noconstraints/3000.png", "no TV, no transforms"),
    ("TV 0.2\n(B16)", "outputs/full50_abl_tv0p2/3000.png", "weaker TV"),
    ("TV 1.0\n(B16)", "outputs/full50_abl_tv1p0/3000.png", "stronger TV"),
    ("CE loss\n(B13)", "outputs/full50_abl_ce/3000.png", "cross-entropy"),
    ("token mode\n(B14)", "outputs/full50_abl_token/3000.png", "semantic, token"),
]

# Single-hue magnitude encoding (the metric panels show magnitude, not identity),
# so no categorical palette is involved and no CVD pair-separation check applies.
HUE = "#1f6feb"
HUE_HI = "#d1495b"   # reserved for the one unconstrained reference bar
INK = "#24292f"
MUTED = "#6e7781"


def sobel(g: np.ndarray) -> np.ndarray:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    ky = kx.T
    p = np.pad(g, 1, mode="edge")
    gx = sum(kx[i, j] * p[i:i + g.shape[0], j:j + g.shape[1]]
             for i in range(3) for j in range(3))
    gy = sum(ky[i, j] * p[i:i + g.shape[0], j:j + g.shape[1]]
             for i in range(3) for j in range(3))
    return np.hypot(gx, gy)


def stats(path: str) -> dict | None:
    full = os.path.join(REPO, path)
    if not os.path.exists(full):
        return None
    rgb = np.asarray(Image.open(full).convert("RGB"), dtype=np.float64) / 255.0
    g = rgb.mean(axis=2)

    tv = float((np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2)

    f = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    power = np.abs(f) ** 2
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    total = power.sum()
    mid = float(power[(r >= 8) & (r < 32)].sum() / total) if total > 0 else float("nan")
    high = float(power[r >= 32].sum() / total) if total > 0 else float("nan")

    s = sobel(g)
    edge = float((s > 0.1 * s.max()).mean()) if s.max() > 0 else float("nan")

    return {"tv_per_px": tv, "mid_band_frac": mid, "high_band_frac": high,
            "edge_density": edge, "std": float(g.std()), "img": rgb}


def main() -> None:
    rows, imgs, labels = [], [], []
    for label, path, note in PATCHES:
        st = stats(path)
        if st is None:
            print(f"  MISSING {path}")
            continue
        imgs.append(st.pop("img"))
        labels.append(label)
        rows.append({"patch": label.replace("\n", " "), "path": path, "note": note, **st})
    if not rows:
        raise SystemExit("no patches found")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fig3_texture_stats.csv"), index=False)
    print(df[["patch", "tv_per_px", "mid_band_frac", "high_band_frac", "edge_density"]]
          .to_string(index=False))

    n = len(imgs)
    metrics = [("tv_per_px", "TV per pixel\n(lower = smoother)"),
               ("mid_band_frac", "mid-band FFT power\n(8-32 cyc/img, stroke scale)"),
               ("edge_density", "edge density\n(Sobel > 10% max)")]

    fig = plt.figure(figsize=(2.05 * n, 5.9))
    gs = fig.add_gridspec(2, len(metrics), height_ratios=[1.0, 1.08],
                          hspace=0.30, wspace=0.26)

    # ── contact sheet (images carry their own pixels; no palette applies) ──
    inner = gs[0, :].subgridspec(1, n, wspace=0.06)
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        ax = fig.add_subplot(inner[0, i])
        ax.imshow(im, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#d0d7de"); sp.set_linewidth(0.8)
        ax.set_title(lab, fontsize=8, color=INK, pad=4)

    # ── metric panels: single series each, so no legend; direct value labels ──
    unconstrained = "no constraints (B12)"
    for k, (key, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, k])
        d = df.sort_values(key)
        y = np.arange(len(d))
        colors = [HUE_HI if p == unconstrained else HUE for p in d["patch"]]
        ax.barh(y, d[key], color=colors, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels([p.replace(" (", "\n(") for p in d["patch"]], fontsize=7, color=INK)
        ax.set_title(title, fontsize=8.5, color=INK, pad=6)
        span = float(d[key].max()) if d[key].max() > 0 else 1.0
        for yi, v in zip(y, d[key]):
            ax.text(v + span * 0.02, yi, f"{v:.4f}", va="center", fontsize=6.8, color=MUTED)
        ax.set_xlim(0, span * 1.28)
        ax.grid(axis="x", alpha=0.18, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#d0d7de")
        ax.tick_params(axis="x", labelsize=7, colors=MUTED, length=2)

    fig.suptitle("C3 / Fig 3 — adversarial patch patterns and their measured texture "
                 "(red bar = the unconstrained ablation)", fontsize=10.5, color=INK, y=0.995)
    p = os.path.join(FIGS, "fig3_patch_patterns.png")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n-> {p}")
    print(f"-> {os.path.join(OUT, 'fig3_texture_stats.csv')}")

    # ── the falsifiable comparison, stated explicitly ──
    def get(name, key):
        m = df[df["patch"] == name]
        return float(m[key].iloc[0]) if len(m) else float("nan")
    nc_tv, ours_tv = get(unconstrained, "tv_per_px"), get("ours 809 sem+constraints", "tv_per_px")
    nc_mid, ours_mid = get(unconstrained, "mid_band_frac"), get("ours 809 sem+constraints", "mid_band_frac")
    auth_tv, auth_mid = get("authors ultrabreak.png", "tv_per_px"), get("authors ultrabreak.png", "mid_band_frac")
    print("\nVERDICT (constrained vs unconstrained):")
    print(f"  TV/px        ours {ours_tv:.4f} | authors {auth_tv:.4f} | unconstrained {nc_tv:.4f}")
    print(f"  mid-band     ours {ours_mid:.4f} | authors {auth_mid:.4f} | unconstrained {nc_mid:.4f}")
    smoother = ours_tv < nc_tv
    more_mid = ours_mid > nc_mid
    print(f"  constrained is smoother than unconstrained:        {smoother}")
    print(f"  constrained has more mid-band (stroke) energy:     {more_mid}")
    if not (smoother and more_mid):
        print("  -> the paper's qualitative claim is NOT supported on these statistics.")


if __name__ == "__main__":
    main()
