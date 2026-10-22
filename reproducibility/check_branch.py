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
def check_cf(trainer, seed=None):
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
    w = REPO / "jobs/_seed_and_run.py"
    wrap_ok = w.exists() and all(k in w.read_text()
                                 for k in ("random.seed", "np.random.seed",
                                           "torch.manual_seed", "torch.cuda.manual_seed_all"))
    check("CF7", (not seeded) and wrap_ok,
          "no seed inside the authors' file; applied externally by jobs/_seed_and_run.py"
          + ("" if wrap_ok else "  <-- wrapper missing or does not seed all four RNGs"))

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
    check("PF7", c.get("seed") in (0, 1, 2),
          f"seed = {c.get('seed')} in S1 set {{0,1,2}} [declared deviation: the paper seeds nothing]")


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


def check_seeds(require_trained=False):
    """S1 — both branches on the identical seed set, and the paper has no seeds to adopt."""
    print("S1 — seeds")
    import pandas as pd  # noqa: F401  (kept for symmetry with the other checks)
    cf = [REPO / f"reproducibility/authors_code/optimisation/optimise_cf_strict_s{s}.py" for s in (0, 1, 2)]
    pf = [REPO / f"reproducibility/paper_faithful/optimise_paper_strict_s{s}.py" for s in (0, 1, 2)]
    check("S1b-cf", all(p.exists() for p in cf),
          f"CF trainers present: {sum(p.exists() for p in cf)}/3")
    check("S1b-pf", all(p.exists() for p in pf),
          f"PF trainers present: {sum(p.exists() for p in pf)}/3")

    cf_names = [f"cf_strict_test50_1300_s{s}" for s in (0, 1, 2)]
    pf_names = [f"pf_strict_test50_1300_s{s}" for s in (0, 1, 2)]
    ok = all(n in p.read_text() for n, p in zip(cf_names, cf) if p.exists()) and \
         all(n in p.read_text() for n, p in zip(pf_names, pf) if p.exists())
    check("S1a", ok, "both branches target the identical seed set {0,1,2}")

    pf_seeds = []
    for sd, p in zip((0, 1, 2), pf):
        if p.exists():
            pf_seeds.append(f"seed={sd}," in p.read_text())
    check("S1a-pf", all(pf_seeds) and len(pf_seeds) == 3,
          "each PF trainer carries its own seed value")

    # the authors have no seeds to adopt — assert that, do not assume it
    import subprocess
    hits = []
    for f in ("optimise.py", "utils.py", "qwen2_adapter.py", "llava16_adapter.py", "base_adapter.py"):
        t = blob(f"optimisation/{f}").decode(errors="replace")
        if any(k in t for k in ("manual_seed", "random.seed", "np.random.seed")):
            hits.append(f)
    check("S1-authors", not hits,
          "the authors' tree sets no seed, so there are none to adopt (D7)"
          + (f"  <-- found seeding in {hits}: USE THEIRS" if hits else ""))

    trained = [(n, (REPO / "outputs" / n / "1300.png").exists()) for n in cf_names + pf_names]
    done = sum(v for _, v in trained)
    msg = (f"trained patches on disk: {done}/6"
           + ("" if done == 6 else "  (missing: " + ", ".join(n for n, v in trained if not v) + ")"))
    if require_trained:
        check("S1b-trained", done == 6, msg)
    else:
        print(f"  [INFO] S1b-trained: {msg} — a post-condition, not a pre-submit gate")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("branch", choices=["cf", "pf", "eval", "seeds"])
    ap.add_argument("--trainer", help="CF: path to the trainer script")
    ap.add_argument("--exp", help="PF: outputs/<exp> whose run_config.json to check")
    ap.add_argument("--config", help="eval: attack_configs/<config>.csv")
    ap.add_argument("--kind", choices=["safebench", "advbench", "mmsafety"])
    ap.add_argument("--seed", type=int, choices=[0, 1, 2])
    ap.add_argument("--require-trained", action="store_true",
                    help="seeds: also require all 6 patches to exist (post-run check)")
    s = ap.parse_args()

    print(f"Branch contract: reproducibility/BRANCH_CONTRACT.md (authors' commit {AUTH})\n")
    if s.branch == "cf":
        default = ("reproducibility/authors_code/optimisation/"
                   f"optimise_cf_strict_s{s.seed if s.seed is not None else 0}.py")
        check_cf(s.trainer or default, s.seed)
    elif s.branch == "seeds":
        check_seeds(s.require_trained)
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
