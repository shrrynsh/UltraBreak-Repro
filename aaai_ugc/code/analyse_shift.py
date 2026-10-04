"""
analyse_shift.py — scoring tables for E1 (format sweep), E2 (forced-prefix
ablation) and E3 (source shift).

Runs on whatever is finished: every missing cell is reported as absent rather
than silently dropped, so the script is safe to call at the end of each requeue.

All comparisons are PAIRED — conditions share the identical query list in the
identical row order, which the loaders assert — so uplift is a row-wise
difference and McNemar is exact on discordant pairs.

Emits metrics, counts and query indices only. No completion text, ever.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "aaai_ugc", "code"))

import metrics as M  # noqa: E402
import stats as S    # noqa: E402

QWEN = "Qwen/Qwen2-VL-7B-Instruct"
GEN = os.path.join(REPO, "aaai_ugc", "results", "gen")
OUT = os.path.join(REPO, "aaai_ugc", "results")
FIG = os.path.join(REPO, "aaai_ugc", "figures")
for d in (OUT, FIG):
    os.makedirs(d, exist_ok=True)

VARIANTS = ["bare", "steps", "question"]
CONDS = ["base", "shuffled", "authors", "p809", "text_only"]

# E3's `authors` arms are reused from existing runs (PLAN.md §4).
E3_REUSE = {
    "safebench": ("results/cmp_img_sb315", 315),
    "advbench": ("results/cmp_img_abnorm", 520),
    "harmbench": ("results/ext_authors_harmbench_norm_eval", 200),
}
E3_SOURCES = ["safebench", "advbench", "harmbench"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def load(cfg: str, judge: str = "v3", root: str | None = None) -> pd.DataFrame | None:
    suf = "_harmbench_v3.csv" if judge == "v3" else "_harmbench.csv"
    base = root if root else GEN
    p = os.path.join(base, cfg, QWEN + suf) if root is None \
        else os.path.join(REPO, root, QWEN + suf)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    return d if "attack_success" in d.columns else None


def succ(d: pd.DataFrame) -> np.ndarray:
    return d["attack_success"].astype(bool).values


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def cell(d: pd.DataFrame | None) -> dict | None:
    if d is None:
        return None
    s = succ(d)
    ci = S.wilson_ci(int(s.sum()), len(s))
    m = M.summarise(d)
    return {"n": len(s), "k": int(s.sum()), "asr": 100 * ci.p,
            "lo": 100 * ci.lo, "hi": 100 * ci.hi,
            "nrr": 100 * m["nrr"], "prefix_rate": 100 * m["prefix_rate"],
            "list_rate": 100 * m["list_rate"],
            "ptr": 100 * m["prefix_then_refuse_rate"],
            "medlen": m["median_answer_len"]}


# ══════════════════════════════════════════════════════════════════════════════
def e1() -> dict:
    banner("E1 — format sweep: does the patch's contribution depend on prompt format?")
    rows, mats = [], {}
    for judge in ("v3", "v1"):
        for v in VARIANTS:
            for c in CONDS:
                d = load(f"E1_{v}_{c}", judge)
                cl = cell(d)
                if cl is None:
                    continue
                mats[(judge, v, c)] = succ(d)
                rows.append({"experiment": "E1", "judge": judge, "variant": v,
                             "condition": c, **cl})
    df = pd.DataFrame(rows)
    if df.empty:
        print("  no E1 results yet")
        return {"status": "pending"}
    df.to_csv(os.path.join(OUT, "E1_asr.csv"), index=False)

    for judge in ("v3", "v1"):
        sub = df[df["judge"] == judge]
        if sub.empty:
            continue
        print(f"\n-- ASR % [Wilson 95% CI], judge {judge} --")
        hdr = f"  {'variant':10s}" + "".join(f"{c:>22s}" for c in CONDS)
        print(hdr)
        for v in VARIANTS:
            line = f"  {v:10s}"
            for c in CONDS:
                r = sub[(sub["variant"] == v) & (sub["condition"] == c)]
                line += f"{'—':>22s}" if r.empty else \
                    f"{r.iloc[0]['asr']:>9.2f} [{r.iloc[0]['lo']:4.1f},{r.iloc[0]['hi']:5.1f}]"
            print(line)

    # ── uplift over base, paired, + exact McNemar ────────────────────────────
    up = []
    print(f"\n-- uplift = ASR(cond) - ASR(base), paired on the same 150 IDs, judge v3 --")
    print(f"  {'variant':10s}" + "".join(f"{c:>26s}" for c in CONDS if c != "base"))
    for v in VARIANTS:
        if ("v3", v, "base") not in mats:
            continue
        b = mats[("v3", v, "base")]
        line = f"  {v:10s}"
        for c in CONDS:
            if c == "base":
                continue
            if ("v3", v, c) not in mats:
                line += f"{'—':>26s}"
                continue
            a = mats[("v3", v, c)]
            mc = S.mcnemar_exact(a, b)
            d_pts = 100 * (a.mean() - b.mean())
            up.append({"variant": v, "condition": c, "judge": "v3",
                       "asr_cond": 100 * a.mean(), "asr_base": 100 * b.mean(),
                       "uplift_pts": d_pts, "mcnemar_b": mc.b, "mcnemar_c": mc.c,
                       "mcnemar_p": mc.p_value, "n_pairs": mc.n_pairs})
            star = "***" if mc.p_value < 1e-3 else "**" if mc.p_value < 1e-2 \
                else "*" if mc.p_value < 0.05 else "ns"
            line += f"{d_pts:>+14.2f} p={mc.p_value:<7.2g}{star:>4s}"[:26].rjust(26)
        print(line)
    updf = pd.DataFrame(up)
    if not updf.empty:
        updf.to_csv(os.path.join(OUT, "E1_uplift_mcnemar.csv"), index=False)
        print("\n  (b = cond success & base failure; c = cond failure & base success;")
        print("   p = two-sided exact McNemar. Full counts in E1_uplift_mcnemar.csv)")

    # ── the central contrast: uplift(steps) vs uplift(bare) ──────────────────
    key = {}
    print(f"\n-- KEY CONTRAST: uplift(steps) - uplift(bare), judge v3 --")
    for c in ("authors", "p809", "shuffled", "text_only"):
        ok = all(("v3", v, x) in mats for v in ("bare", "steps") for x in (c, "base"))
        if not ok:
            continue
        u_bare = mats[("v3", "bare", c)].mean() - mats[("v3", "bare", "base")].mean()
        u_step = mats[("v3", "steps", c)].mean() - mats[("v3", "steps", "base")].mean()
        key[c] = {"uplift_bare_pts": 100 * u_bare, "uplift_steps_pts": 100 * u_step,
                  "difference_pts": 100 * (u_step - u_bare)}
        print(f"  {c:10s}  uplift(bare)={100*u_bare:+6.2f}   "
              f"uplift(steps)={100*u_step:+6.2f}   difference={100*(u_step-u_bare):+6.2f} pts")
    if key:
        with open(os.path.join(OUT, "E1_key_contrast.json"), "w") as f:
            json.dump(key, f, indent=2)
        print("\n  A large positive difference = the patch helps far more when the prompt")
        print("  already matches the training format -> FORMAT-BOUND TRIGGER.")
        print("  A difference near zero = the patch acts as a format-independent switch.")

    # ── response-shape diagnostics ───────────────────────────────────────────
    print(f"\n-- response shape, judge v3 (NRR / prefix / list / prefix-then-refuse / median len) --")
    print(f"  {'variant':10s} {'cond':10s} {'NRR':>6s} {'pfx%':>6s} {'list%':>6s} {'PTR%':>6s} {'medlen':>7s}")
    for v in VARIANTS:
        for c in CONDS:
            r = df[(df["judge"] == "v3") & (df["variant"] == v) & (df["condition"] == c)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"  {v:10s} {c:10s} {r['nrr']:>6.1f} {r['prefix_rate']:>6.1f} "
                  f"{r['list_rate']:>6.1f} {r['ptr']:>6.1f} {r['medlen']:>7.0f}")

    _fig_e1(df)
    return {"status": "ok", "cells": len(df), "key_contrast": key}


def _fig_e1(df: pd.DataFrame) -> None:
    sub = df[df["judge"] == "v3"]
    if sub.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"base": "#8b949e", "shuffled": "#c9a227", "authors": "#1f6feb",
              "p809": "#d1495b", "text_only": "#2d6a4f"}
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    width = 0.16
    xs = np.arange(len(VARIANTS))
    for i, c in enumerate(CONDS):
        vals, los, his = [], [], []
        for v in VARIANTS:
            r = sub[(sub["variant"] == v) & (sub["condition"] == c)]
            if r.empty:
                vals.append(np.nan); los.append(0); his.append(0)
            else:
                r = r.iloc[0]
                vals.append(r["asr"]); los.append(r["asr"] - r["lo"]); his.append(r["hi"] - r["asr"])
        ax.bar(xs + (i - 2) * width, vals, width, label=c, color=colors[c],
               yerr=[los, his], capsize=2.5, error_kw={"lw": 0.9})
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v}" for v in VARIANTS])
    ax.set_xlabel("prompt variant (same 150 AdvBench intents, intent held constant)")
    ax.set_ylabel("ASR % (judge v3, Wilson 95% CI)")
    ax.set_title("E1 — format sweep on Qwen2-VL-7B: ASR by prompt format x image condition")
    ax.legend(frameon=False, ncol=5, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    p = os.path.join(FIG, "E1_format_sweep.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"\n  figure -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
def e2() -> dict:
    banner("E2 — forced-prefix ablation: how much survives without the TPG instruction?")
    rows, mats = [], {}
    for judge in ("v3", "v1"):
        for tpg in ("on", "off"):
            for c in ("base", "authors"):
                d = load(f"E2_tpg{tpg}_{c}", judge)
                cl = cell(d)
                if cl is None:
                    continue
                mats[(judge, tpg, c)] = succ(d)
                rows.append({"experiment": "E2", "judge": judge, "tpg": tpg,
                             "condition": c, **cl})
    df = pd.DataFrame(rows)
    if df.empty:
        print("  no E2 results yet")
        return {"status": "pending"}
    df.to_csv(os.path.join(OUT, "E2_asr.csv"), index=False)

    for judge in ("v3", "v1"):
        sub = df[df["judge"] == judge]
        if sub.empty:
            continue
        print(f"\n-- SafeBench-315 ASR % [Wilson 95% CI], judge {judge} --")
        print(f"  {'TPG':6s} {'base':>24s} {'authors':>24s} {'uplift':>10s}")
        for tpg in ("on", "off"):
            cells = {}
            line = f"  {tpg:6s}"
            for c in ("base", "authors"):
                r = sub[(sub["tpg"] == tpg) & (sub["condition"] == c)]
                if r.empty:
                    line += f"{'—':>24s}"
                else:
                    cells[c] = r.iloc[0]
                    line += f"{r.iloc[0]['asr']:>11.2f} [{r.iloc[0]['lo']:4.1f},{r.iloc[0]['hi']:5.1f}]"
            if len(cells) == 2:
                line += f"{cells['authors']['asr'] - cells['base']['asr']:>+10.2f}"
            print(line)

    out = {}
    print(f"\n-- uplift (authors - base) and exact McNemar, judge v3 --")
    for tpg in ("on", "off"):
        if ("v3", tpg, "base") in mats and ("v3", tpg, "authors") in mats:
            a, b = mats[("v3", tpg, "authors")], mats[("v3", tpg, "base")]
            mc = S.mcnemar_exact(a, b)
            u = 100 * (a.mean() - b.mean())
            out[f"tpg_{tpg}"] = {"uplift_pts": u, "mcnemar_b": mc.b, "mcnemar_c": mc.c,
                                 "mcnemar_p": mc.p_value, "n_pairs": mc.n_pairs}
            print(f"  TPG {tpg:3s}: uplift = {u:+6.2f} pts   {mc}")
    if "tpg_on" in out and "tpg_off" in out:
        d = out["tpg_off"]["uplift_pts"] - out["tpg_on"]["uplift_pts"]
        out["uplift_off_minus_on_pts"] = d
        print(f"\n  uplift(TPG off) - uplift(TPG on) = {d:+.2f} pts")
        print("  Uplift collapsing when the forced prefix is removed means the image's")
        print("  standalone contribution is small - it needs the text scaffold.")
    with open(os.path.join(OUT, "E2_uplift_mcnemar.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n-- response shape, judge v3 --")
    print(f"  {'TPG':5s} {'cond':8s} {'NRR':>6s} {'pfx%':>6s} {'list%':>6s} {'PTR%':>6s} {'medlen':>7s}")
    for _, r in df[df["judge"] == "v3"].iterrows():
        print(f"  {r['tpg']:5s} {r['condition']:8s} {r['nrr']:>6.1f} {r['prefix_rate']:>6.1f} "
              f"{r['list_rate']:>6.1f} {r['ptr']:>6.1f} {r['medlen']:>7.0f}")
    return {"status": "ok", "cells": len(df), **out}


# ══════════════════════════════════════════════════════════════════════════════
def e3() -> dict:
    banner("E3 — source shift with format held constant (all prompts in `steps` form)")
    rows, mats = [], {}
    for judge in ("v3", "v1"):
        for src in E3_SOURCES:
            d = load(f"E3_{src}_base", judge)
            if d is not None:
                mats[(judge, src, "base")] = succ(d)
                rows.append({"experiment": "E3", "judge": judge, "source": src,
                             "condition": "base", "reused": False, **cell(d)})
            path, n_exp = E3_REUSE[src]
            da = load("", judge, root=path)
            if da is not None:
                if len(da) != n_exp:
                    print(f"  WARNING reused {path} has {len(da)} rows, expected {n_exp}")
                mats[(judge, src, "authors")] = succ(da)
                rows.append({"experiment": "E3", "judge": judge, "source": src,
                             "condition": "authors", "reused": True, **cell(da)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("  no E3 results yet")
        return {"status": "pending"}
    df.to_csv(os.path.join(OUT, "E3_asr.csv"), index=False)

    for judge in ("v3", "v1"):
        sub = df[df["judge"] == judge]
        if sub.empty:
            continue
        print(f"\n-- ASR % [Wilson 95% CI] and uplift, judge {judge} --")
        print(f"  {'source':12s} {'n':>5s} {'base':>24s} {'authors':>24s} {'uplift':>9s} "
              f"{'NRR base':>9s} {'NRR auth':>9s}")
        for src in E3_SOURCES:
            b = sub[(sub["source"] == src) & (sub["condition"] == "base")]
            a = sub[(sub["source"] == src) & (sub["condition"] == "authors")]
            nb = f"{int(b.iloc[0]['n'])}" if not b.empty else (
                f"{int(a.iloc[0]['n'])}" if not a.empty else "?")
            fb = f"{b.iloc[0]['asr']:>11.2f} [{b.iloc[0]['lo']:4.1f},{b.iloc[0]['hi']:5.1f}]" \
                if not b.empty else f"{'—':>24s}"
            fa = f"{a.iloc[0]['asr']:>11.2f} [{a.iloc[0]['lo']:4.1f},{a.iloc[0]['hi']:5.1f}]" \
                if not a.empty else f"{'—':>24s}"
            up = f"{a.iloc[0]['asr'] - b.iloc[0]['asr']:>+9.2f}" \
                if (not a.empty and not b.empty) else f"{'—':>9s}"
            nrb = f"{b.iloc[0]['nrr']:>9.1f}" if not b.empty else f"{'—':>9s}"
            nra = f"{a.iloc[0]['nrr']:>9.1f}" if not a.empty else f"{'—':>9s}"
            print(f"  {src:12s} {nb:>5s} {fb} {fa} {up} {nrb} {nra}")

    out = {}
    print(f"\n-- exact McNemar, authors vs base, per source (judge v3) --")
    for src in E3_SOURCES:
        if ("v3", src, "base") in mats and ("v3", src, "authors") in mats:
            a, b = mats[("v3", src, "authors")], mats[("v3", src, "base")]
            if len(a) != len(b):
                print(f"  {src:12s} SKIPPED - unpaired lengths {len(a)} vs {len(b)}")
                continue
            mc = S.mcnemar_exact(a, b)
            out[src] = {"uplift_pts": 100 * (a.mean() - b.mean()), "mcnemar_b": mc.b,
                        "mcnemar_c": mc.c, "mcnemar_p": mc.p_value, "n_pairs": mc.n_pairs}
            print(f"  {src:12s} uplift={100*(a.mean()-b.mean()):+6.2f} pts   {mc}")
    with open(os.path.join(OUT, "E3_uplift_mcnemar.json"), "w") as f:
        json.dump(out, f, indent=2)
    if len(out) > 1:
        u = {k: v["uplift_pts"] for k, v in out.items()}
        print(f"\n  uplift spread across sources at FIXED format: "
              f"{min(u.values()):+.1f} to {max(u.values()):+.1f} pts "
              f"({max(u.values())-min(u.values()):.1f} pts)")
        print("  A large spread = a genuine source effect beyond phrasing.")
    return {"status": "ok", "cells": len(df), **{f"uplift_{k}": v["uplift_pts"]
                                                 for k, v in out.items()}}


def main() -> None:
    t0 = datetime.now(timezone.utc)
    man = {"git_commit": git_commit(), "model": QWEN,
           "run_utc": t0.isoformat(), "judges": ["v3", "v1"]}
    man["E1"] = e1()
    man["E2"] = e2()
    man["E3"] = e3()
    man["elapsed_s"] = (datetime.now(timezone.utc) - t0).total_seconds()
    with open(os.path.join(OUT, "shift_analysis.manifest.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    banner("analysis written to aaai_ugc/results/ (E1_*.csv, E2_*.csv, E3_*.csv)")


if __name__ == "__main__":
    main()
