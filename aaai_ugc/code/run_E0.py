"""
run_E0.py — E0: re-analysis of existing scored outputs. CPU only, no GPU.

Asks, at zero GPU cost, whether universality already falls off for queries
unlike the 50-row training corpus:

  E0a  harm-category analysis. As specified (seen vs unseen categories) the
       split is DEGENERATE — every SafeBench-315 category is present in
       training — so the degeneracy is derived and reported, and the analysis is
       redefined to (i) per-category ASR within the 7 seen categories and
       (ii) a cross-source comparison where categories genuinely do differ.
  E0b  ASR vs max cosine similarity to the training corpus (quartile bins,
       pooled and per-benchmark, plus a logistic fit with benchmark covariate).
  E0c  response-format analysis for successes vs failures.

Writes aaai_ugc/results/E0/*.csv and aaai_ugc/figures/E0_*.png.
Emits metrics, counts, category names and query indices only — never completion text.
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

from create_attack_configs import load_safebench  # noqa: E402
import metrics as M                               # noqa: E402
import stats as S                                 # noqa: E402

OUT = os.path.join(REPO, "aaai_ugc", "results", "E0")
FIG = os.path.join(REPO, "aaai_ugc", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

QWEN = "Qwen/Qwen2-VL-7B-Instruct"

# (label, config dir, benchmark, prompt-format) -> already-scored runs, from PLAN.md §1
RUNS = {
    ("authors", "safebench"): ("cmp_img_sb315", "steps"),
    ("authors", "advbench"): ("cmp_img_abnorm", "steps"),
    ("p809", "safebench"): ("full50_normfix_lr0p01_s3000_safebench_eval", "steps"),
    ("p809", "advbench"): ("full50_normfix_lr0p01_s3000_advbench_eval", "steps"),
}
# Extra runs used only by the cross-source table (E0a-ii).
XSOURCE = [
    ("authors", "safebench-315", "cmp_img_sb315"),
    ("authors", "advbench-520-steps", "cmp_img_abnorm"),
    ("authors", "advbench-520-raw", "cmp_img_abraw"),
    ("authors", "harmbench-200-raw", "ext_authors_harmbench_eval"),
    ("authors", "harmbench-200-steps", "ext_authors_harmbench_norm_eval"),
    ("authors", "strongreject-313-raw", "ext_authors_strongreject_eval"),
    ("p809", "safebench-315", "full50_normfix_lr0p01_s3000_safebench_eval"),
    ("p809", "advbench-520-steps", "full50_normfix_lr0p01_s3000_advbench_eval"),
    ("p809", "harmbench-200-raw", "ext_809_harmbench_eval"),
    ("p809", "harmbench-200-steps", "ext_809_harmbench_norm_eval"),
    ("p809", "strongreject-313-raw", "ext_809_strongreject_eval"),
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def load_scored(cfg: str, judge: str = "v3") -> pd.DataFrame | None:
    suf = "_harmbench_v3.csv" if judge == "v3" else "_harmbench.csv"
    p = os.path.join(REPO, "results", cfg, QWEN + suf)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ══════════════════════════════════════════════════════════════════════════════
# E0a — categories
# ══════════════════════════════════════════════════════════════════════════════
def e0a() -> dict:
    banner("E0a — harm-category analysis")

    tiny = pd.read_csv(os.path.join(REPO, "datasets", "SafeBench-Tiny.csv"))
    full = pd.read_csv(os.path.join(REPO, "datasets", "safebench.csv"))
    ev = load_safebench(os.path.join(REPO, "datasets", "safebench.csv"), None,
                        os.path.join(REPO, "datasets", "SafeBench-Tiny.csv"),
                        exclude_categories=True)

    train_cats = sorted(set(tiny["category_name"].str.strip()))
    eval_cats = sorted(set(ev["category_name"].str.strip()))
    unseen = sorted(set(eval_cats) - set(train_cats))
    train_only = sorted(set(train_cats) - set(eval_cats))

    print(f"training corpus  : {len(tiny)} rows, {len(train_cats)} categories")
    print(f"eval set (315)   : {len(ev)} rows, {len(eval_cats)} categories")
    print(f"UNSEEN categories in eval : {unseen if unseen else '[] (none)'}")
    print(f"train-only categories     : {train_only}")
    print("\n>>> The seen/unseen split is DEGENERATE: every eval category is in training.")
    print(">>> 'Held out' in this protocol means held-out QUERIES, never held-out CATEGORIES.")

    degeneracy = {
        "train_rows": int(len(tiny)),
        "train_categories": train_cats,
        "eval_rows": int(len(ev)),
        "eval_categories": eval_cats,
        "unseen_eval_categories": unseen,
        "train_only_categories": train_only,
        "degenerate": len(unseen) == 0,
        "full_safebench_rows": int(len(full)),
    }

    # ── E0a-i: per-category ASR inside the 7 seen categories ──────────────────
    # results carry `target` (clean instruction, period-stripped) but no category,
    # so join back to safebench.csv on the normalised instruction string.
    key = lambda s: s.astype(str).str.strip().str.rstrip(".").str.lower()
    cat_map = dict(zip(key(full["instruction"]), full["category_name"].str.strip()))

    rows = []
    for patch in ("authors", "p809"):
        cfg, _ = RUNS[(patch, "safebench")]
        for judge in ("v3", "v1"):
            d = load_scored(cfg, judge)
            if d is None:
                continue
            d = d.copy()
            d["category"] = key(d["target"]).map(cat_map)
            miss = int(d["category"].isna().sum())
            if miss:
                print(f"  WARNING {patch}/{judge}: {miss} rows failed the category join")
            for cat, sub in d.groupby("category"):
                k = int(sub["attack_success"].astype(bool).sum())
                ci = S.wilson_ci(k, len(sub))
                rows.append({"patch": patch, "judge": judge, "category": cat,
                             "n": len(sub), "n_success": k, "asr": 100 * ci.p,
                             "ci_lo": 100 * ci.lo, "ci_hi": 100 * ci.hi})
    percat = pd.DataFrame(rows)
    percat.to_csv(os.path.join(OUT, "E0a_per_category_asr.csv"), index=False)

    print("\n-- per-category ASR, SafeBench-315 (all 7 categories SEEN in training) --")
    for judge in ("v3", "v1"):
        sub = percat[percat["judge"] == judge]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="category", columns="patch", values="asr")
        print(f"\n  judge {judge}:")
        for cat, r in piv.iterrows():
            cells = "  ".join(
                f"{p}={r[p]:5.1f}" for p in piv.columns if not pd.isna(r.get(p))
            )
            n = int(sub[sub["category"] == cat]["n"].iloc[0])
            print(f"    {cat:22s} n={n:3d}  {cells}")
        if len(piv) > 1:
            for p in piv.columns:
                v = piv[p].dropna()
                if len(v) > 1:
                    print(f"    -> {p}: spread {v.min():.1f}-{v.max():.1f} "
                          f"({v.max()-v.min():.1f} pts across seen categories)")

    # ── E0a-ii: cross-source comparison ───────────────────────────────────────
    xrows = []
    for patch, source, cfg in XSOURCE:
        for judge in ("v3", "v1"):
            d = load_scored(cfg, judge)
            if d is None:
                continue
            k = int(d["attack_success"].astype(bool).sum())
            ci = S.wilson_ci(k, len(d))
            m = M.summarise(d)
            xrows.append({"patch": patch, "source": source, "judge": judge,
                          "config": cfg, "n": len(d), "n_success": k,
                          "asr": 100 * ci.p, "ci_lo": 100 * ci.lo, "ci_hi": 100 * ci.hi,
                          "nrr": 100 * m["nrr"], "prefix_rate": 100 * m["prefix_rate"],
                          "list_rate": 100 * m["list_rate"]})
    xdf = pd.DataFrame(xrows)
    xdf.to_csv(os.path.join(OUT, "E0a_cross_source_asr.csv"), index=False)

    print("\n-- cross-source ASR (where harm categories genuinely DO differ) --")
    v3 = xdf[xdf["judge"] == "v3"]
    print(f"  {'source':24s} {'patch':8s} {'n':>5s} {'ASR v3 [95% CI]':>24s} {'NRR':>7s}")
    for _, r in v3.sort_values(["source", "patch"]).iterrows():
        print(f"  {r['source']:24s} {r['patch']:8s} {int(r['n']):>5d} "
              f"{r['asr']:>9.2f} [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]  {r['nrr']:>6.1f}")

    return {"degeneracy": degeneracy,
            "per_category_rows": len(percat),
            "cross_source_rows": len(xdf)}


# ══════════════════════════════════════════════════════════════════════════════
# E0b — ASR vs semantic distance to the training corpus
# ══════════════════════════════════════════════════════════════════════════════
def e0b() -> dict:
    banner("E0b — ASR vs semantic similarity to the training corpus")

    from embed import embed_texts, max_similarity_to_train

    tiny = pd.read_csv(os.path.join(REPO, "datasets", "SafeBench-Tiny.csv"))
    train_q = tiny["instruction"].astype(str).str.strip().tolist()
    print(f"training queries embedded: {len(train_q)}")

    # Build the evaluation query lists from the scored files themselves, so the
    # row order matches the success vector exactly.
    frames = {}
    for (patch, bench), (cfg, _) in RUNS.items():
        d = load_scored(cfg, "v3")
        if d is None:
            print(f"  MISSING {patch}/{bench} ({cfg}) — skipped")
            continue
        frames[(patch, bench)] = d

    # queries per benchmark (identical across patches — assert it)
    bench_q = {}
    for (patch, bench), d in frames.items():
        q = d["target"].astype(str).str.strip().tolist()
        if bench in bench_q:
            assert bench_q[bench] == q, f"{bench}: query order differs between patches"
        else:
            bench_q[bench] = q
    for b, q in bench_q.items():
        print(f"  {b}: {len(q)} eval queries")

    all_eval = [(b, i, q) for b, qs in bench_q.items() for i, q in enumerate(qs)]
    print(f"embedding {len(all_eval)} eval + {len(train_q)} train queries (CPU)...")
    emb_eval = embed_texts([q for _, _, q in all_eval], verbose=False)
    emb_train = embed_texts(train_q, verbose=False)
    sim, nn_idx = max_similarity_to_train(emb_eval, emb_train)
    print(f"max-sim to training corpus: min={sim.min():.3f} "
          f"median={np.median(sim):.3f} max={sim.max():.3f}")

    simdf = pd.DataFrame({
        "benchmark": [b for b, _, _ in all_eval],
        "row_id": [i for _, i, _ in all_eval],
        "max_cos_sim": sim,
        "nearest_train_row": nn_idx,
    })
    # attach success per patch
    for (patch, bench), d in frames.items():
        succ = d["attack_success"].astype(bool).values
        mask = simdf["benchmark"] == bench
        simdf.loc[mask, f"success_{patch}"] = succ[simdf.loc[mask, "row_id"].values]
    simdf.to_csv(os.path.join(OUT, "E0b_similarity_per_query.csv"), index=False)
    print(f"  per-query similarities -> E0b_similarity_per_query.csv")

    # ── quartile bins, pooled and per-benchmark ───────────────────────────────
    def binned(df: pd.DataFrame, scope: str) -> list:
        out = []
        # quartiles by rank so bins are equal-sized regardless of distribution shape
        q = pd.qcut(df["max_cos_sim"], 4, labels=["Q1 (farthest)", "Q2", "Q3", "Q4 (nearest)"],
                    duplicates="drop")
        for patch in ("authors", "p809"):
            col = f"success_{patch}"
            if col not in df.columns:
                continue
            sub = df.dropna(subset=[col])
            if sub.empty:
                continue
            qq = q.loc[sub.index]
            for b, g in sub.groupby(qq, observed=True):
                k = int(g[col].astype(bool).sum())
                ci = S.wilson_ci(k, len(g))
                out.append({"scope": scope, "patch": patch, "bin": str(b), "n": len(g),
                            "n_success": k, "asr": 100 * ci.p,
                            "ci_lo": 100 * ci.lo, "ci_hi": 100 * ci.hi,
                            "sim_min": g["max_cos_sim"].min(),
                            "sim_max": g["max_cos_sim"].max()})
        return out

    brows = binned(simdf, "pooled")
    for b in sorted(simdf["benchmark"].unique()):
        brows += binned(simdf[simdf["benchmark"] == b].copy(), b)
    bdf = pd.DataFrame(brows)
    bdf.to_csv(os.path.join(OUT, "E0b_asr_by_similarity_bin.csv"), index=False)

    for scope in bdf["scope"].unique():
        print(f"\n-- ASR by similarity quartile ({scope}), judge v3 --")
        sub = bdf[bdf["scope"] == scope]
        print(f"  {'bin':16s} {'patch':8s} {'n':>4s} {'ASR [95% CI]':>24s} {'sim range':>16s}")
        for _, r in sub.iterrows():
            print(f"  {r['bin']:16s} {r['patch']:8s} {int(r['n']):>4d} "
                  f"{r['asr']:>9.2f} [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}] "
                  f"  {r['sim_min']:.3f}-{r['sim_max']:.3f}")

    # ── logistic regression: success ~ sim + benchmark ────────────────────────
    lrows = []
    for patch in ("authors", "p809"):
        col = f"success_{patch}"
        if col not in simdf.columns:
            continue
        d = simdf.dropna(subset=[col]).copy()
        y = d[col].astype(bool).astype(float).values
        is_adv = (d["benchmark"] == "advbench").astype(float).values
        X = np.column_stack([np.ones(len(d)), d["max_cos_sim"].values, is_adv])
        fit = S.logistic_fit(X, y, ["intercept", "max_cos_sim", "benchmark=advbench"])
        print(f"\n-- logistic: success ~ max_cos_sim + benchmark   [{patch}, judge v3] --")
        print(fit.table())
        if fit.separated:
            print(f"   NOTE quasi-separated terms (coef not interpretable): {fit.separated}")
        for i, nm in enumerate(fit.names):
            lrows.append({"patch": patch, "term": nm, "coef": fit.coef[i],
                          "se": fit.se[i], "z": fit.z[i], "p": fit.p[i],
                          "n": fit.n, "converged": fit.converged})
        # and within each benchmark alone, where format is constant
        for b in sorted(d["benchmark"].unique()):
            db = d[d["benchmark"] == b]
            if db[col].nunique() < 2:
                continue
            Xb = np.column_stack([np.ones(len(db)), db["max_cos_sim"].values])
            fb = S.logistic_fit(Xb, db[col].astype(bool).astype(float).values,
                                ["intercept", "max_cos_sim"])
            print(f"   within {b} only: max_cos_sim coef={fb.coef[1]:+.3f} "
                  f"se={fb.se[1]:.3f} p={fb.p[1]:.3g}")
            lrows.append({"patch": patch, "term": f"max_cos_sim|{b}", "coef": fb.coef[1],
                          "se": fb.se[1], "z": fb.z[1], "p": fb.p[1],
                          "n": fb.n, "converged": fb.converged})
    pd.DataFrame(lrows).to_csv(os.path.join(OUT, "E0b_logistic.csv"), index=False)

    _figure_e0b(bdf)
    return {"n_eval_embedded": int(len(simdf)), "n_train_embedded": len(train_q),
            "sim_median": float(np.median(sim))}


def _figure_e0b(bdf: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scopes = [s for s in ("safebench", "advbench", "pooled") if s in set(bdf["scope"])]
    if not scopes:
        return
    order = ["Q1 (farthest)", "Q2", "Q3", "Q4 (nearest)"]
    colors = {"authors": "#1f6feb", "p809": "#d1495b"}
    fig, axes = plt.subplots(1, len(scopes), figsize=(4.6 * len(scopes), 4.0), sharey=True)
    if len(scopes) == 1:
        axes = [axes]
    for ax, scope in zip(axes, scopes):
        sub = bdf[bdf["scope"] == scope]
        for patch in ("authors", "p809"):
            s = sub[sub["patch"] == patch]
            if s.empty:
                continue
            s = s.set_index("bin").reindex([b for b in order if b in set(s["bin"])]).reset_index()
            x = np.arange(len(s))
            ax.errorbar(x, s["asr"],
                        yerr=[s["asr"] - s["ci_lo"], s["ci_hi"] - s["asr"]],
                        marker="o", capsize=3, lw=1.8, color=colors[patch], label=patch)
            ax.set_xticks(x)
            ax.set_xticklabels(s["bin"], rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{scope}", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("ASR % (judge v3, Wilson 95% CI)")
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle("E0b — ASR vs semantic similarity to the 50-query training corpus",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG, "E0b_asr_vs_similarity.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"\n  figure -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
# E0c — response format
# ══════════════════════════════════════════════════════════════════════════════
def e0c() -> dict:
    banner("E0c — response-format analysis, successes vs failures")
    rows = []
    for (patch, bench), (cfg, fmt) in RUNS.items():
        d = load_scored(cfg, "v3")
        if d is None:
            continue
        sp = M.split_summary(d)
        for _, r in sp.iterrows():
            rows.append({"patch": patch, "benchmark": bench, "format": fmt,
                         "judge": "v3", **r.to_dict()})
    fdf = pd.DataFrame(rows)
    fdf.to_csv(os.path.join(OUT, "E0c_response_format.csv"), index=False)

    print(f"  {'patch':8s} {'benchmark':10s} {'group':8s} {'n':>5s} {'prefix%':>8s} "
          f"{'list%':>7s} {'medlen':>7s} {'refus%':>7s} {'pfx-then-refuse%':>17s}")
    for _, r in fdf.iterrows():
        if r.get("n", 0) == 0:
            continue
        print(f"  {r['patch']:8s} {r['benchmark']:10s} {r['group']:8s} {int(r['n']):>5d} "
              f"{100*r['prefix_rate']:>7.1f} {100*r['list_rate']:>6.1f} "
              f"{r['median_answer_len']:>7.0f} {100*r['refusal_rate']:>6.1f} "
              f"{100*r['prefix_then_refuse_rate']:>16.1f}")

    # whole-condition summary too (NRR etc.)
    srows = []
    for (patch, bench), (cfg, fmt) in RUNS.items():
        for judge in ("v3", "v1"):
            d = load_scored(cfg, judge)
            if d is None:
                continue
            srows.append({"patch": patch, "benchmark": bench, "format": fmt,
                          "judge": judge, "config": cfg, **M.summarise(d)})
    sdf = pd.DataFrame(srows)
    sdf.to_csv(os.path.join(OUT, "E0c_condition_summary.csv"), index=False)
    print("\n-- condition summary (metrics on the EXTRACTED answer, not the raw cell) --")
    print(f"  {'patch':8s} {'benchmark':10s} {'judge':6s} {'n':>5s} {'ASR':>7s} "
          f"{'NRR':>7s} {'prefix%':>8s} {'list%':>7s} {'medlen':>7s}")
    for _, r in sdf.iterrows():
        print(f"  {r['patch']:8s} {r['benchmark']:10s} {r['judge']:6s} {int(r['n']):>5d} "
              f"{100*r['asr']:>6.2f} {100*r['nrr']:>6.1f} {100*r['prefix_rate']:>7.1f} "
              f"{100*r['list_rate']:>6.1f} {r['median_answer_len']:>7.0f}")
    return {"format_rows": len(fdf), "summary_rows": len(sdf)}


def main() -> None:
    t0 = datetime.now(timezone.utc)
    man = {"experiment": "E0", "git_commit": git_commit(),
           "start_utc": t0.isoformat(), "judge": "v3 (+v1 where available)",
           "model": QWEN, "gpu": False}
    man["E0a"] = e0a()
    man["E0b"] = e0b()
    man["E0c"] = e0c()
    man["end_utc"] = datetime.now(timezone.utc).isoformat()
    man["elapsed_s"] = (datetime.now(timezone.utc) - t0).total_seconds()
    with open(os.path.join(OUT, "E0.manifest.json"), "w") as f:
        json.dump(man, f, indent=2, default=str)
    banner(f"E0 complete in {man['elapsed_s']:.0f}s — outputs in aaai_ugc/results/E0/")


if __name__ == "__main__":
    main()
