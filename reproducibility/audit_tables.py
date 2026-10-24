#!/usr/bin/env python3
"""Per-table CF/PF/seed audit. Encodes BRANCH_CONTRACT.md at TABLE level.

check_branch.py answers "may I submit this job?". This answers the other question:
"does the number already printed in a table satisfy CF or PF, on all three seeds?"

A table's branch row is SATISFIED only if, for every seed in {0,1,2}, a config exists
whose patch is that branch's contract patch for that seed, evaluated on every model the
table reports, under both judges (E5).

    python reproducibility/audit_tables.py            # verdict per table
    python reproducibility/audit_tables.py --why      # also explain every rejected patch

Exit 0 = every branch row of every table satisfied. Non-zero = re-reproduction needed.
"""
import argparse, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUTH = "c4c276d"
SEEDS = (0, 1, 2)
CKPT = "1300.png"

SIX_MODELS = ["Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen-VL-Chat", "Qwen/Qwen2.5-VL-7B-Instruct",
              "llava-hf/llava-v1.6-mistral-7b-hf", "moonshotai/Kimi-VL-A3B-Instruct",
              "THUDM/GLM-4.1V-9B-Thinking"]
FIVE_MODELS = [m for m in SIX_MODELS if "Kimi" not in m]        # paper's Table 3 omits Kimi
TWO_VICTIMS = ["Qwen/Qwen-VL-Chat", "THUDM/GLM-4.1V-9B-Thinking"]


def contract_patch(branch, seed):
    return f"outputs/{branch}_strict_test50_1300_s{seed}/{CKPT}"


# ---- table registry: the branch rows actually printed in repro_notes/TABLES.md ----
# (table, sub, models, kind, {branch: [configs that currently back that row]})
TABLES = [
    ("1a", "SafeBench 315",   SIX_MODELS, "safebench", {"cf": ["cmp_cf_sb315"],  "pf": ["cmp_pf_sb315"]}),
    ("1b", "AdvBench 520",    SIX_MODELS, "advbench",  {"cf": ["cmp_cf_abnorm"], "pf": ["cmp_pf_abnorm"]}),
    ("1c", "MM-Safety",       SIX_MODELS, "mmsafety",  {"cf": ["t1mm_cf_mmsafety"], "pf": ["t1mm_pf_mmsafety"]}),
    ("3",  "ablations",       FIVE_MODELS, "safebench",
     {"cf": [], "pf": ["pf3_noconstraints_safebench", "pf3_ce_safebench", "pf3_token_safebench"]}),
    ("6",  "TV sweep",        TWO_VICTIMS, "safebench",
     {"cf": [], "pf": ["pf6_tv0_safebench", "pf6_tv0p2_safebench", "pf6_tv1p0_safebench"]}),
]

# A PF ablation row is NOT built on one of the six anchor patches: it is a PF-style
# retrain (projection ON, authors' corpus, Table 4) with exactly ONE knob moved. The
# moved knob is the experiment, so it is an INTENDED deviation from PF4, not a failure.
# Expected artifact names, one per variant per seed, and the knob each is allowed to move.
PF_VARIANTS = {
    "3": {"pf_abl_noconstraints": ("tv_weight", 0.0),   # + transforms off
          "pf_abl_ce":            ("loss", "ce"),
          "pf_abl_token":         ("loss_mode", "token")},
    "6": {"pf_tv0p2":             ("tv_weight", 0.2),
          "pf_tv1p0":             ("tv_weight", 1.0)},
}
# the unablated row of Table 3, and TV=0.5 of Table 6, ARE the PF anchor itself
PF_ANCHOR_ROW = {"3": "Proposed", "6": "TV=0.5"}
# tables with no branch row at all, and why
NO_BRANCH = [("2", "StrongREJECT", "blocked — native 0-1 scorer unavailable"),
             ("4", "hyperparameters", "descriptive, no patch"),
             ("5", "affirming phrase", "authors' released image, no retrain"),
             ("9", "closed models", "blocked — paid API"),
             ("10", "overhead", "per-step ratio, branch-invariant")]
# CF is impossible for these, by CF1's two-line limit
CF_IMPOSSIBLE = {"3", "6"}


def patches_of(cfg):
    import pandas as pd
    p = REPO / "attack_configs" / f"{cfg}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if "image" not in d.columns:
        return set()
    return set(d["image"].astype(str))


def cells_of(cfg, models):
    """models fully judged under BOTH judges (E5)."""
    out = []
    for m in models:
        d = REPO / "results" / cfg / m
        v3 = d.parent / f"{d.name}_harmbench_v3.csv"
        v1 = d.parent / f"{d.name}_harmbench.csv"
        out.append((m, v3.exists(), v1.exists()))
    return out


def explain(patch):
    """Why is this patch not a contract patch? Checked, not assumed."""
    if patch.startswith("outputs/ultrabreak"):
        return "E7 — the authors' RELEASED image; a branch must be a retrain"
    if patch.startswith("images/") or patch.startswith("baselines/"):
        return "not a branch row (control or baseline)"
    exp = patch.split("/")[1] if "/" in patch else patch
    rc = REPO / "outputs" / exp / "run_config.json"
    if not rc.exists():
        return (f"no run_config.json in outputs/{exp} — provenance unverifiable; "
                "cannot satisfy PF, and CF is judged on its trainer")
    c = json.loads(rc.read_text())
    bad = []
    tc = c.get("train_config", "")
    if "test.csv" not in tc:
        bad.append(f"PF5 corpus={tc}")
    if c.get("projection") is not True:
        bad.append(f"PF1 projection={c.get('projection')}")
    if c.get("num_epochs") != 1301:
        bad.append(f"PF3 num_epochs={c.get('num_epochs')}")
    if c.get("surrogates") != ["qwen"]:
        bad.append(f"PF6 surrogates={c.get('surrogates')}")
    if c.get("seed") not in SEEDS:
        bad.append(f"PF7 seed={c.get('seed')}")
    a = c.get("ablation", {})
    moved = []
    for k, want in (("tau", 0.5), ("loss", "semantic"), ("loss_mode", "attention")):
        if a.get(k) != want:
            moved.append(f"{k}={a.get(k)}")
    if c.get("tv_weight") != 0.5:
        moved.append(f"tv_weight={c.get('tv_weight')}")
    # ONE moved knob on an ablation artifact is the experiment, not a defect
    if moved and exp.startswith(("a_", "pf_abl_", "pf_tv")):
        note = f"ablated knob (intended): {', '.join(moved)}"
    elif moved:
        note = "fails PF4 " + ", ".join(moved)
        bad.append(note); note = None
    else:
        note = None
    msg = ("fails " + ", ".join(bad)) if bad else "no clause violation found (unexpected)"
    return msg + (f"   [{note}]" if note else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--why", action="store_true", help="explain every rejected patch")
    args = ap.parse_args()

    print("TABLE-LEVEL CF/PF/SEED AUDIT")
    print(f"contract: reproducibility/BRANCH_CONTRACT.md   authors' commit {AUTH}")
    print(f"a branch row passes only with all of seeds {set(SEEDS)}, every model, both judges\n")

    # which contract patches even exist
    print("Contract patches on disk:")
    for b in ("cf", "pf"):
        have = [s for s in SEEDS if (REPO / contract_patch(b, s)).exists()]
        print(f"  {b.upper()}: seeds {have} trained ({len(have)}/3)")
    print()

    bad_tables = []
    rejected = {}
    hdr = f"{'table':6} {'row':14} {'branch':7} {'seeds':7} {'cells':9} verdict"
    print(hdr); print("-" * len(hdr))
    for tid, sub, models, kind, rows in TABLES:
        for branch in ("cf", "pf"):
            cfgs = rows[branch]
            if branch == "cf" and tid in CF_IMPOSSIBLE:
                print(f"{tid:6} {sub:14} {'CF':7} {'-':7} {'-':9} n/a BY DEFINITION (CF1 allows 2 lines; "
                      f"an ablation needs a 3rd)")
                continue
            # find, per seed, a config on that seed's contract patch
            seeds_ok, seeds_part = [], []
            for s in SEEDS:
                want = contract_patch(branch, s)
                hit = [c for c in cfgs if (patches_of(c) or set()) == {want}]
                if not hit:
                    continue
                full = all(v3 and v1 for c in hit for _, v3, v1 in cells_of(c, models))
                (seeds_ok if full else seeds_part).append(s)
            # record what the row IS built on
            for c in cfgs:
                for p in (patches_of(c) or set()):
                    if p not in {contract_patch(b, s) for b in ("cf", "pf") for s in SEEDS}:
                        rejected.setdefault(p, set()).add(c)
            n = len(seeds_ok)
            if n == 3:
                v = "SATISFIED"
            elif n or seeds_part:
                v = f"PARTIAL — seeds {seeds_ok} complete, {seeds_part} incomplete"
                bad_tables.append((tid, branch.upper(), v))
            else:
                if tid in PF_VARIANTS and branch == "pf":
                    want = [f"{v}_s{s}" for v in PF_VARIANTS[tid] for s in SEEDS]
                    have = sum((REPO / "outputs" / w / CKPT).exists() for w in want)
                    v = (f"NOT SATISFIED — needs {len(want)} PF-style variant retrains "
                         f"(projection ON); {have}/{len(want)} exist")
                else:
                    v = "NOT SATISFIED — no cell on any contract patch"
                bad_tables.append((tid, branch.upper(), v))
            ncell = sum(1 for c in cfgs for _, v3, v1 in cells_of(c, models) if v3 and v1)
            print(f"{tid:6} {sub:14} {branch.upper():7} {f'{n}/3':7} {f'{ncell}/{len(models)*3}':9} {v}")
    print()
    for tid, sub, why in NO_BRANCH:
        print(f"{tid:6} {sub:14} {'-':7} {'-':7} {'-':9} no branch row — {why}")

    if args.why:
        print("\nWhat the current table numbers are actually built on, and why each is not a branch:")
        for p, cfgs in sorted(rejected.items()):
            print(f"\n  {p}")
            print(f"    used by: {', '.join(sorted(cfgs))}")
            print(f"    {explain(p)}")

    print()
    if bad_tables:
        print(f"RESULT: {len(bad_tables)} branch row(s) need re-reproduction:")
        for tid, b, v in bad_tables:
            print(f"  Table {tid} / {b}: {v}")
        return 1
    print("RESULT: every branch row satisfied on all three seeds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
