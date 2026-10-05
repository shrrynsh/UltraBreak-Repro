"""
build_E4_configs.py — configs for E4 (transfer targets), reusing the E1/E2 builders.

E4 asks whether the format/template dependence found on the white-box surrogate
survives on black-box transfer targets. Design per target:

  E4a  the SAME 150 AdvBench IDs as E1 x {bare, steps} x {base, shuffled, authors}
  E4b  a fixed 150-row subset of SafeBench-315 x {TPG on, off} x {base, authors}

Four of the ten cells per target have a BYTE-IDENTICAL existing run from the
transfer study, so those are reused rather than regenerated (see REUSE below) —
which both saves the time box and removes any risk of disagreement:

  E4a bare  x authors  ==  cmp_img_abraw                (advbench raw,  TPG on)
  E4a steps x authors  ==  cmp_img_abnorm               (advbench steps, TPG on)
  E4b on    x authors  ==  cmp_img_sb315                (safebench315,  TPG on)
  E4b off   x base     ==  noattack_safebench_matrix    (safebench315,  TPG off)

Equality is asserted row-by-row here, not assumed.

Everything is built with the imported E1/E2 primitives (`build_variant_config`,
`normalize_instruction`, `make_tpg_text`), so prompts are identical in
construction to the pilot's.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from build_configs import (  # noqa: E402  — reuse, do not reimplement
    CFG_DIR,
    PATCHES,
    PHRASE,
    SEED,
    build_variant_config,
    git_commit,
    md5,
    safebench_315,
    write_cfg,
)

E4_N = 150
CONDS = ["base", "shuffled", "authors"]

# cell -> (existing config dir, dataset the ids index into)
REUSE = {
    ("E4a", "bare", "authors"): "cmp_img_abraw",
    ("E4a", "steps", "authors"): "cmp_img_abnorm",
    ("E4b", "on", "authors"): "cmp_img_sb315",
    ("E4b", "off", "base"): "noattack_safebench_matrix",
}


def advbench_e1_subset() -> tuple[pd.DataFrame, list]:
    """The exact 150 AdvBench IDs E1 used — read from the frozen file, never redrawn."""
    ids_path = os.path.join(CFG_DIR, "E1_ids.txt")
    ids = [int(x) for x in open(ids_path).read().split()]
    assert len(ids) == E4_N, f"E1_ids.txt has {len(ids)} ids, expected {E4_N}"
    full = pd.read_csv(os.path.join(REPO, "datasets", "adv_bench.csv"))
    sub = full.iloc[ids].copy().rename(columns={"goal": "clean_target"})
    sub["category_name"] = "AdvBench"
    return sub[["clean_target", "category_name"]].reset_index(drop=True), ids


def safebench_e4b_subset() -> tuple[pd.DataFrame, list]:
    """
    Fixed 150-row subset of the canonical SafeBench-315, seed 0, frozen to file.

    Positions index into load_safebench(...315) row order, which is the row order
    of every existing SafeBench-315 result file — so the same positions select the
    same queries in the reused runs and in the new ones.
    """
    sb = safebench_315()
    rng = np.random.RandomState(SEED)
    ids = sorted(rng.choice(len(sb), size=E4_N, replace=False).tolist())
    p = os.path.join(CFG_DIR, "E4b_ids.txt")
    if os.path.exists(p):
        prev = [int(x) for x in open(p).read().split()]
        assert prev == ids, "E4b_ids.txt disagrees with the seeded draw — refusing to change IDs"
    else:
        with open(p, "w") as f:
            f.write("\n".join(str(i) for i in ids) + "\n")
    return sb.iloc[ids].reset_index(drop=True), ids


def assert_reuse_identical(cell, cfg_name, built_rows: pd.DataFrame, ids: list) -> dict:
    """
    Prove the reused run's config rows are byte-identical to the cell we would
    have built, on the selected IDs. Compares text/image/target exactly.
    """
    p = os.path.join(REPO, "attack_configs", f"{cfg_name}.csv")
    ref = pd.read_csv(p, keep_default_na=False)
    sel = ref.iloc[ids].reset_index(drop=True)
    mine = built_rows.reset_index(drop=True)
    for col in ("text", "image", "target"):
        bad = (sel[col].astype(str).values != mine[col].astype(str).values)
        if bad.any():
            i = int(np.argmax(bad))
            raise SystemExit(
                f"REUSE MISMATCH for {cell} vs {cfg_name}, column {col}, first at row {i}\n"
                f"  existing: {sel[col].iloc[i]!r}\n  built:    {mine[col].iloc[i]!r}"
            )
    return {"cell": "/".join(cell), "reused_config": cfg_name,
            "rows": len(sel), "identical": True, "n_full": len(ref)}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-rows", type=int, default=None)
    args = ap.parse_args()
    rows, pre = args.dry_run_rows, ("dry_" if args.dry_run_rows else "")

    print("=" * 78)
    print(f"build_E4_configs.py{'  (dry run)' if rows else ''}")
    print("=" * 78)

    adv, adv_ids = advbench_e1_subset()
    sb, sb_ids = safebench_e4b_subset()
    print(f"  E4a: {len(adv)} AdvBench ids (reused from E1_ids.txt)")
    print(f"  E4b: {len(sb)} SafeBench-315 positions, seed {SEED} -> E4b_ids.txt")

    built, reuse_log = [], []

    # ── E4a: {bare, steps} x {base, shuffled, authors} ───────────────────────
    for variant in ("bare", "steps"):
        for cond in CONDS:
            frame = build_variant_config(adv, variant, PATCHES[cond], tpg=True)
            cell = ("E4a", variant, cond)
            if cell in REUSE and not rows:
                reuse_log.append(assert_reuse_identical(cell, REUSE[cell], frame, adv_ids))
                print(f"  [reuse] E4a_{variant}_{cond:8s} == {REUSE[cell]} (verified identical)")
                continue
            built.append(write_cfg(
                f"{pre}E4a_{variant}_{cond}", frame,
                {"experiment": "E4a", "variant": variant, "image_condition": cond,
                 "image_path": PATCHES[cond], "image_md5": md5(PATCHES[cond]),
                 "dataset": "advbench", "subset_ids_file": "E1_ids.txt",
                 "subset_n": E4_N, "tpg": True, "phrase": PHRASE}, rows))

    # ── E4b: {tpg on, off} x {base, authors} ─────────────────────────────────
    for tpg_on in (True, False):
        tag = "on" if tpg_on else "off"
        for cond in ("base", "authors"):
            frame = build_variant_config(sb, "bare", PATCHES[cond], tpg=tpg_on)
            cell = ("E4b", tag, cond)
            if cell in REUSE and not rows:
                reuse_log.append(assert_reuse_identical(cell, REUSE[cell], frame, sb_ids))
                print(f"  [reuse] E4b_tpg{tag}_{cond:8s} == {REUSE[cell]} (verified identical)")
                continue
            built.append(write_cfg(
                f"{pre}E4b_tpg{tag}_{cond}", frame,
                {"experiment": "E4b", "variant": "safebench-native",
                 "image_condition": cond, "image_path": PATCHES[cond],
                 "image_md5": md5(PATCHES[cond]), "dataset": "safebench-315-sub150",
                 "subset_ids_file": "E4b_ids.txt", "subset_n": E4_N,
                 "tpg": tpg_on, "phrase": PHRASE}, rows))

    tot = sum(b["rows"] for b in built)
    print(f"\n  NEW configs: {len(built)}, {tot} generations per target")
    print(f"  REUSED cells: {len(reuse_log)} (byte-identical, verified row by row)")
    for b in built:
        print(f"    {b['config']:26s} {b['rows']:4d}")

    idx = {"new_configs": built, "reused": reuse_log,
           "e4a_ids_file": "E1_ids.txt", "e4b_ids_file": "E4b_ids.txt",
           "new_generations_per_target": tot, "git_commit": git_commit()}
    with open(os.path.join(CFG_DIR, f"{pre}E4_INDEX.json"), "w") as f:
        json.dump(idx, f, indent=2)
    print(f"  index -> aaai_ugc/configs/{pre}E4_INDEX.json")


if __name__ == "__main__":
    main()
