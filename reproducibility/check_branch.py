#!/usr/bin/env python3
"""Branch contract gate. See reproducibility/BRANCH_CONTRACT.md — this file encodes it.

No job that trains or evaluates a patch for a paper table may be submitted until the
relevant check passes. Every clause prints PASS/FAIL with its contract id.

    # training side
    python reproducibility/check_branch.py cf --trainer reproducibility/authors_code/optimisation/optimise_cf_strict.py
    python reproducibility/check_branch.py pf --exp repro_paper_faithful_1300

    # evaluation side (either branch)
    python reproducibility/check_branch.py eval --config cfs_sb315   --kind safebench
    python reproducibility/check_branch.py eval --config cfs_abnorm  --kind advbench
    python reproducibility/check_branch.py eval --config mm1680_pf   --kind mmsafety

Exit status 0 = contract satisfied. Non-zero = do not submit.
"""
import argparse, difflib, inspect, json, subprocess, sys
from pathlib import Path

AUTH = "c4c276d"
REPO = Path(__file__).resolve().parent.parent
FAILED = []


def check(cid, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}: {detail}")
    if not ok:
        FAILED.append(cid)
    return ok


def blob(path):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{AUTH}:{path}"],
                       capture_output=True)
    if r.returncode:
        raise SystemExit(f"cannot read {AUTH}:{path} — is this the right repo?")
    return r.stdout


# ---------------------------------------------------------------- CF
def check_cf(trainer):
    print(f"CF — code-faithful  (trainer: {trainer})")
    tp = REPO / trainer
    if not tp.exists():
        return check("CF1", False, f"{trainer} does not exist")
    src = blob("optimisation/optimise.py").decode()
    out = tp.read_text()

    ch = [l for l in difflib.unified_diff(src.splitlines(), out.splitlines(), lineterm="", n=0)
          if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    allowed = all(("exp_name" in l) or ("num_epochs" in l) for l in ch)
    check("CF1", len(ch) <= 4 and allowed,
          f"{len(ch)//2} line(s) changed vs {AUTH}:optimisation/optimise.py"
          + ("" if allowed else " — a line other than exp_name/num_epochs was touched"))

    check("CF2", "num_epochs = 1301" in out, "num_epochs = 1301 (saves epoch 1300)")

    corpus_line = 'train_config = "./train_configs/test.csv"'
    has_corpus = corpus_line in out
    check("CF3a", has_corpus, "trainer reads the authors' ./train_configs/test.csv")
    same = blob("train_configs/test.csv") == (REPO / "train_configs/test.csv").read_bytes()
    check("CF3b", same, f"train_configs/test.csv byte-identical to {AUTH}")

    tree = ("__init__.py", "base_adapter.py", "llava16_adapter.py",
            "optimise.py", "qwen2_adapter.py", "utils.py")
    bad = [f for f in tree
           if blob(f"optimisation/{f}") !=
           (REPO / "reproducibility/authors_code/optimisation" / f).read_bytes()]
    check("CF4", not bad, f"authors' tree pristine ({len(tree)-len(bad)}/{len(tree)} files)"
          + (f" — modified: {bad}" if bad else ""))

    sys.path.insert(0, str(REPO / "reproducibility/authors_code/optimisation"))
    from qwen2_adapter import Qwen2Adapter  # noqa: E402
    s = inspect.getsource(Qwen2Adapter.compute_loss)
    raw = ("if self.patch_only:" in s and
           "image_grid_thw[0], patch)" in s and "patch_only=False" in out)
    check("CF5", raw, "raw patch injection — the D11 bug is PRESENT (as the release has it)")

    check("CF6", "project_patch" not in out, "no §3.2 projection on this path")

    seeded = any(k in out for k in ("manual_seed", "random.seed", "np.random.seed"))
    check("CF7", not seeded, "no seed set (the authors set none)")

    hp = all(x in out for x in ("lr = 0.01", "tv_weight = 0.5", "l2_weight = 0.0",
                                "custom_loss = True", "torch.clamp(adv_patch.data, 0, 1)"))
    check("CF8", hp, "lr 0.01 / TV 0.5 / L2 0.0 / semantic loss / clamp per step, as shipped")


# ---------------------------------------------------------------- PF
def check_pf(exp):
    print(f"PF — paper-faithful, §3.2 = Reading A  (run: {exp})")
    rc = REPO / "outputs" / exp / "run_config.json"
    if not rc.exists():
        return check("PF1", False, f"outputs/{exp}/run_config.json does not exist")
    c = json.loads(rc.read_text())
    a = c.get("ablation", {})

    check("PF1", c.get("projection") is True,
          f"§3.2 projection ACTIVE (projection={c.get('projection')})"
          + ("" if c.get("projection") else "  <-- Reading B is NOT paper-faithful; see BEYOND_REPRO §B25"))
    np_ = a.get("normalise_patch")
    check("PF2", np_ is True,
          f"patch normalised into CLIP space (normalise_patch={np_})"
          + ("  <-- field ABSENT from this run's dump, so it cannot be verified; it is not "
             "evidence the fix was off (the field was added to run_config.json later)"
             if np_ is None else ""))
    check("PF3", c.get("num_epochs") == 1301, f"1300 iterations (num_epochs={c.get('num_epochs')})")

    t4 = {"lr": (c.get("lr"), 0.01), "tv_weight": (c.get("tv_weight"), 0.5),
          "tau": (a.get("tau"), 0.5), "embed_noise": (a.get("embed_noise"), 1e-4),
          "loss": (a.get("loss"), "semantic"), "loss_mode": (a.get("loss_mode"), "attention"),
          "transforms": (a.get("transforms"), True),
          "scale_range": (tuple(a.get("scale_range", ())), (0.8, 1.2)),
          "rotation_range": (tuple(a.get("rotation_range", ())), (-15.0, 15.0))}
    off = {k: v for k, (v, want) in t4.items() if v != want}
    check("PF4", not off, "Table 4 verbatim" + (f" — off: {off}" if off else ""))

    import pandas as pd
    tc = c.get("train_config", "")
    corpus = REPO / tc
    n = len(pd.read_csv(corpus)) if corpus.exists() else -1
    is_authors = corpus.exists() and blob("train_configs/test.csv") == corpus.read_bytes()
    check("PF5", n == 50 and is_authors,
          f"corpus = {tc} ({n} rows), authors' test.csv: {is_authors}"
          + ("" if is_authors else
             "  <-- must be ./train_configs/test.csv, so CF and PF differ ONLY in the §3.2 treatment"))
    check("PF6", c.get("surrogates") == ["qwen"], f"surrogate = {c.get('surrogates')}")
    check("PF7", c.get("seed") == 0, f"seed = {c.get('seed')} [declared deviation: the paper seeds nothing]")


# ---------------------------------------------------------------- eval
def check_eval(cfg, kind):
    print(f"EVAL — paper protocol  (config: {cfg}, kind: {kind})")
    import pandas as pd
    p = REPO / "attack_configs" / f"{cfg}.csv"
    if not p.exists():
        return check("E0", False, f"attack_configs/{cfg}.csv does not exist")
    d = pd.read_csv(p)
    want = {"safebench": 315, "advbench": 520, "mmsafety": 1680}[kind]
    cid = {"safebench": "E1", "advbench": "E2", "mmsafety": "E3"}[kind]
    check(cid, len(d) == want, f"{len(d)} rows (contract: {want})")
    if kind == "advbench":
        frac = d["text"].astype(str).str.startswith("Steps to").mean()
        check("E2b", frac > 0.9, f"{frac:.1%} of rows in the §3.3 'Steps to' form")
    imgs = sorted(set(d["image"].astype(str)))
    check("E4", len(imgs) == 1, f"single patch across the config: {imgs if len(imgs) < 4 else len(imgs)}")
    leak = [i for i in imgs if "ultrabreak" in i]
    check("E7", not leak,
          "no released image in this branch config"
          + (f"  <-- {leak} is the AUTHORS' RELEASED IMAGE; a branch must be a retrain" if leak else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("branch", choices=["cf", "pf", "eval"])
    ap.add_argument("--trainer", help="CF: path to the trainer script")
    ap.add_argument("--exp", help="PF: outputs/<exp> whose run_config.json to check")
    ap.add_argument("--config", help="eval: attack_configs/<config>.csv")
    ap.add_argument("--kind", choices=["safebench", "advbench", "mmsafety"])
    s = ap.parse_args()

    print(f"Branch contract: reproducibility/BRANCH_CONTRACT.md (authors' commit {AUTH})\n")
    if s.branch == "cf":
        check_cf(s.trainer or "reproducibility/authors_code/optimisation/optimise_cf_strict.py")
    elif s.branch == "pf":
        if not s.exp:
            ap.error("pf needs --exp")
        check_pf(s.exp)
    else:
        if not (s.config and s.kind):
            ap.error("eval needs --config and --kind")
        check_eval(s.config, s.kind)

    print()
    if FAILED:
        print(f"CONTRACT NOT SATISFIED — failed: {', '.join(FAILED)}")
        print("DO NOT SUBMIT.")
        return 1
    print("CONTRACT SATISFIED — safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
