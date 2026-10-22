#!/usr/bin/env bash
#SBATCH --job-name=ub_retrain6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@1200
#
# TRAINING ONLY. Six patches, no evaluation.  Contract: reproducibility/BRANCH_CONTRACT.md
#
#   CF  code-faithful  : authors' optimise.py, 2-line diff, authors' ./train_configs/test.csv,
#                        1300 iters, raw patch injection (the D11 bug is PRESENT, by design),
#                        seeded EXTERNALLY by jobs/_seed_and_run.py so CF1's 2-line limit holds.
#   PF  paper-faithful : literal section 3.2 (Reading A, projection ACTIVE), Table 4, same
#                        corpus, same seeds.
#   S1  seeds {0,1,2}  : the identical set on both branches, so a CF-vs-PF difference can
#                        never be a difference of seeds. The paper states no seeds - verified
#                        by clause S1-authors, not assumed.
#
# Evaluation is deliberately NOT here. It is a separate job submitted once all six patches
# exist, because the patches are the irreplaceable part: an evaluation can always be redone
# from a saved PNG, a training that never ran cannot. Nothing is chained after this job.
#
# ---- NOTHING LOST, NO TIME WASTED ----
# 1. A partial training cannot be resumed: --base_epoch restores patch pixels but NOT Adam
#    state, so a resumed run is a different run. GUARD: a partial is discarded and restarted,
#    and a training is only STARTED if its branch reserve fits in the remaining wall.
# 2. The reserve is PER BRANCH, from measured times (see below). A flat reserve would start a
#    PF training with ~7h left and lose all 7.24h at the wall.
# 3. A bash trap does NOT interrupt a running foreground command - it is deferred until that
#    command returns, which is why job 9138's pre-wall handoff never fired. GUARD: elapsed
#    time is checked EXPLICITLY before every training; the USR1 trap is only a backstop.
# 4. A resume must not spin forever making no progress. GUARD: one pending resume at a time,
#    and the latch counts FINISHED 1300.png files - not directories, which os.makedirs
#    creates before the first epoch, so a failed training cannot masquerade as progress.
#
# MEASURED training wall times at 1300 iterations on the 50-row corpus, taken from the
# existing artifacts' own 0.png -> 1300.png timestamps:
#   no projection (CF path)      4.39 / 4.64 / 4.68 / 4.69 h   -> ~4.7h  (12.7 s/epoch)
#   projection ACTIVE (PF path)  7.24 h                        -> ~7.3h  (20.0 s/epoch)
# PF is 1.5x slower because project() runs per row per epoch.
#
# BUDGET: 3 x 4.7h + 3 x 7.3h = 36.0 GPU-h. Two windows: CF s0,s1,s2 + PF s0 in the first
# (22.2h), PF s1,s2 in the second (14.6h).
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate || { echo "[rt6] repro venv missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1
L="[rt6]"
ATTEMPT="${1:-1}"; MAX=8
SEEDS=(0 1 2); CKPT=1300
mkdir -p logs

echo "$L start $(date) attempt $ATTEMPT/$MAX"

# ── Stage 0: contract gate, before the GPU is touched ─────────────────────────
python reproducibility/check_branch.py seeds || exit 1
for s in "${SEEDS[@]}"; do
  python reproducibility/check_branch.py cf --seed "$s" || exit 1
done
python - <<'PY' || exit 1
import importlib.metadata as m, importlib.util, torch, sys, os
assert torch.__version__.startswith("2.5.1"), torch.__version__
assert m.version("transformers").startswith("4.51"), m.version("transformers")
HERE="reproducibility/paper_faithful"; sys.path.insert(0, os.path.join(HERE,"optimisation"))
for s in (0,1,2):
    spec=importlib.util.spec_from_file_location(f"pf{s}", f"{HERE}/optimise_paper_strict_s{s}.py")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod._assert_table4()
    P=mod.PAPER
    assert P.seed==s and P.exp_name==f"pf_strict_test50_1300_s{s}"
    assert P.no_projection is False, "PF1: the projection must be ACTIVE (Reading A)"
    assert P.raw_patch_injection is False, "PF2: patch normalised into CLIP space"
    assert P.train_config=="./train_configs/test.csv" and P.num_epochs==1301
print("[rt6] PF guards OK for all three seeds (Reading A, Table 4, authors' corpus)")
PY
echo "$L CONTRACT GATE PASSED"

# ── wall accounting ───────────────────────────────────────────────────────────
WALL_LIMIT=86400
RESERVE_CF=21600          # 6.0h for a ~4.7h CF training
RESERVE_PF=32400          # 9.0h for a ~7.3h PF training
reserve_for(){ [[ "$1" == pf ]] && echo $RESERVE_PF || echo $RESERVE_CF; }
wall_left(){ echo $(( WALL_LIMIT - SECONDS )); }

trained_n(){ ls outputs/{cf,pf}_strict_test50_1300_s*/${CKPT}.png 2>/dev/null | wc -l; }
START=$(trained_n); RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d; d=$(trained_n)
  (( d >= 6 )) && { echo "$L COMPLETE $d/6"; return; }
  (( ATTEMPT >= MAX )) && { echo "$L MAX attempts at $d/6 - stopping rather than looping" >&2; return; }
  if [[ "${1:-}" == end ]] && (( d <= START )); then
    echo "$L no training finished this window - NOT resubmitting (would loop)" >&2; return
  fi
  local n; n=$(sbatch --parsable jobs/retrain_6patches.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n ($d/6 trained)" \
    || echo "$L resubmit FAILED: $n" >&2; }
trap 'echo "$L USR1 backstop (the explicit guard should have handed off already)"; resub; exit 0' USR1

# ── train ─────────────────────────────────────────────────────────────────────
train_one(){   # branch seed
  local br="$1" sd="$2" exp patch rsv
  exp="${br}_strict_test50_1300_s${sd}"; patch="outputs/${exp}/${CKPT}.png"
  [[ -f "$patch" ]] && { echo "$L $exp already trained - skipping"; return 0; }
  rsv=$(reserve_for "$br")
  if (( SECONDS > WALL_LIMIT - rsv )); then
    echo "$L wall: $(( $(wall_left) / 60 ))m left, a $br training reserves $((rsv/60))m - handing off"
    resub; exit 0
  fi
  if compgen -G "outputs/${exp}/*.png" >/dev/null; then
    echo "$L $exp is PARTIAL with no ${CKPT}.png - discarding and restarting from scratch,"
    echo "$L because --base_epoch restores patch pixels but not Adam state."
    rm -f outputs/${exp}/*.png outputs/${exp}/losses.csv
  fi
  echo ""; echo "$L ===== train $exp  (seed $sd, reserve $((rsv/3600))h, $(( $(wall_left)/3600 ))h left) ====="
  if [[ "$br" == cf ]]; then
    python jobs/_seed_and_run.py "$sd" \
      "reproducibility/authors_code/optimisation/optimise_cf_strict_s${sd}.py" \
      || { echo "$L TRAIN FAILED $exp" >&2; return 1; }
  else
    python "reproducibility/paper_faithful/optimise_paper_strict_s${sd}.py" \
      || { echo "$L TRAIN FAILED $exp" >&2; return 1; }
  fi
  [[ -f "$patch" ]] || { echo "$L $exp produced no ${CKPT}.png" >&2; return 1; }
  python - "$patch" "$exp" <<'PY'
import sys, numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(float)/255.0
print(f"[rt6] {sys.argv[2]}: std={a.std():.4f} range=[{a.min():.3f},{a.max():.3f}]")
PY
  echo "$L trained $exp  ($(trained_n)/6)"
}

for br in cf pf; do
  for sd in "${SEEDS[@]}"; do
    train_one "$br" "$sd" || { echo "$L stopping after a failed training" >&2; break 2; }
  done
done

# ── report ────────────────────────────────────────────────────────────────────
echo ""; echo "$L ===== patches on disk ====="
python - <<'PY'
import os, numpy as np
from PIL import Image
print(f"  {'artifact':36s} {'std':>8s} {'min':>7s} {'max':>7s}")
for br in ("cf","pf"):
    for s in (0,1,2):
        p=f"outputs/{br}_strict_test50_1300_s{s}/1300.png"
        if os.path.exists(p):
            a=np.asarray(Image.open(p).convert("RGB")).astype(float)/255.0
            print(f"  {br}_strict_test50_1300_s{s:<20d} {a.std():>8.4f} {a.min():>7.3f} {a.max():>7.3f}")
        else:
            print(f"  {br}_strict_test50_1300_s{s:<20d} {'--':>8s} {'--':>7s} {'--':>7s}")
print("\n  reference: released ultrabreak.png std 0.1636, range [0.000,1.000]")
PY

echo ""; echo "$L trainings: $(trained_n)/6"
if (( $(trained_n) >= 6 )); then
  echo "$L ===== ALL SIX PATCHES TRAINED ====="
  python reproducibility/check_branch.py seeds --require-trained || true
  echo "$L Evaluation is a SEPARATE job and is deliberately not chained here."
else
  resub end
fi
echo "$L finished $(date)"
