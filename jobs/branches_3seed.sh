#!/usr/bin/env bash
#SBATCH --job-name=ub_branches
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@1200
#
# BOTH BRANCHES, THREE SEEDS.  Contract: reproducibility/BRANCH_CONTRACT.md
#
#   CF  code-faithful  : authors' optimise.py, 2-line diff, authors' test.csv corpus,
#                        1300 iters, raw patch injection (D11 bug PRESENT), seeded
#                        EXTERNALLY by jobs/_seed_and_run.py so CF1's 2-line limit holds.
#   PF  paper-faithful : literal section 3.2 (Reading A, projection ACTIVE), Table 4,
#                        same corpus, same seeds.
#   S1  seeds {0,1,2}  : identical set on both branches, so a CF-vs-PF difference can
#                        never be a difference of seeds. The paper states no seeds
#                        (verified: check_branch.py seeds, clause S1-authors).
#   Eval: paper protocol - SafeBench 315, AdvBench 520 in the section 3.3 form, BOTH
#         scorers (v1 is the paper's own and leads; v3 beside it).
#
# ONE job, one marker directory, one chain. Two interleaved chains with a SLURM
# dependency would have been fragile: a self-resume gets a NEW job id, so an
# afterany: dependency on the FIRST attempt releases early and the two branches
# interleave unpredictably. One chain removes that entirely.
#
# ---- NOTHING LOST, NO TIME WASTED: the four failure modes and the guard for each ----
#
# 1. attack.py writes its CSV only after a WHOLE config finishes, so a cell killed
#    mid-generation loses every row and repeats identically. GUARD: cell_fits() starts a
#    cell only if the whole thing fits in the remaining wall, using rates recalibrated
#    from job 9187's MEASURED per-cell times (see rate_for). The old table was 2.6-5.4x
#    optimistic for LLaVA and GLM, which is exactly how a window gets burned for zero rows.
# 2. A bash trap does NOT interrupt a running foreground command - it is deferred until
#    that command returns. So the USR1 pre-wall trap cannot be relied on (it is what
#    failed in job 9138). GUARD: elapsed time is checked EXPLICITLY between every stage
#    and every cell; the trap is only a backstop.
# 3. A partial training cannot be resumed: --base_epoch restores patch pixels but NOT
#    Adam state, so a resumed run is not the same run. GUARD: a partial training is
#    discarded and restarted, and is only STARTED if RESERVE_TRAIN fits.
# 4. A self-resume must not be refused by the submit limit, and must not spin forever
#    making no progress. GUARD: exactly one pending resume at a time (RESUB latch), and
#    `resub end` refuses to resubmit a window that completed no new unit of work.
#
# ORDER MATTERS FOR PARTIAL COMPLETION. Cells are ordered branch-major, then seed, then
# cheapest model first. So if the budget runs out we hold COMPLETE rows for the seeds
# that finished rather than three half-filled rows: seed 0 of both branches is finished
# before seed 1 starts, and every seed's cheap models land before its GLM cells.
#
# Budget: 6 trainings x ~5.2h = ~31h, plus ~10h eval per patch = ~60h. ~4 windows.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate || { echo "[br] repro venv missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1
L="[br]"; AUTH="c4c276d"
MARK="results/.branches3"; mkdir -p "$MARK" logs
STAGE="${SLURM_TMPDIR:-/tmp}/br_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE"
ATTEMPT="${1:-1}"; MAX=12
SEEDS=(0 1 2)
CKPT=1300
# cheapest first, GLM last: a window boundary then costs the least re-work
MODELS=(Qwen/Qwen2-VL-7B-Instruct Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen-VL-Chat
        moonshotai/Kimi-VL-A3B-Instruct llava-hf/llava-v1.6-mistral-7b-hf
        THUDM/GLM-4.1V-9B-Thinking)
TOTAL_CELLS=72   # 2 branches x 3 seeds x 6 models x 2 benchmarks

echo "$L start $(date) attempt $ATTEMPT/$MAX"

# ── Stage 0: contract gate. Every clause, before the GPU is touched. ──────────
python reproducibility/check_branch.py seeds   || exit 1
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
    assert P.no_projection is False and P.raw_patch_injection is False
    assert P.train_config=="./train_configs/test.csv" and P.num_epochs==1301
print("[br] PF guards OK for all three seeds (Reading A, Table 4, authors' corpus)")
PY
echo "$L CONTRACT GATE PASSED"

# ── wall accounting ───────────────────────────────────────────────────────────
WALL_LIMIT=86400
RESERVE_TRAIN=25200       # 7h; a 1300-iteration training measured ~5.2h
wall_left(){ echo $(( WALL_LIMIT - SECONDS )); }

# Rates RECALIBRATED from job 9187's measured per-cell wall times on these exact
# benchmarks (SafeBench-315 / AdvBench-520), then given ~1.5x headroom. The previous
# table was badly optimistic - measured s/row: LLaVA 16.5 (table said 2.2), GLM 26.4
# and one cold-start cell at 74.6 (table said 29), Qwen-VL-Chat 6.6 (table said 1.4).
rate_for(){ case "$1" in
    *GLM-4.1V*)      echo 40.0 ;;   # measured 12.3-26.4, cold start 74.6
    *llava*)         echo 20.0 ;;   # measured 11.8-16.5
    *Kimi-VL*)       echo 12.0 ;;   # not measured on these sets; conservative
    *Qwen-VL-Chat*)  echo 8.0  ;;   # measured 1.8-6.6
    *)               echo 6.0  ;;   # Qwen2-VL / Qwen2.5-VL, measured 2.5-4.3
  esac; }
cell_fits(){   # cfg model -> 0 if the WHOLE cell fits in what is left
  local cfg="$1" m="$2"
  local rows; rows=$(( $(wc -l < "attack_configs/${cfg}.csv") - 1 ))
  local est; est=$(python -c "print(int($rows*$(rate_for "$m")+1800))")   # +30m model load
  if (( SECONDS + est > WALL_LIMIT - 600 )); then
    echo "$L wall: ${cfg}/${m} needs ~$((est/60))m, only $(( (WALL_LIMIT-SECONDS)/60 ))m left - handing off"
    return 1
  fi
  return 0
}

done_n(){ ls "$MARK" 2>/dev/null | wc -l; }
units_done(){ echo $(( $(done_n) + $(ls -d outputs/{cf,pf}_strict_test50_1300_s*/ 2>/dev/null | wc -l) )); }
START=$(units_done); RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d; d=$(done_n)
  (( d >= TOTAL_CELLS )) && { echo "$L COMPLETE $d/$TOTAL_CELLS"; return; }
  (( ATTEMPT >= MAX )) && { echo "$L MAX attempts at $d/$TOTAL_CELLS - stopping rather than looping" >&2; return; }
  if [[ "${1:-}" == end ]] && (( $(units_done) <= START )); then
    echo "$L no new unit of work completed this window - NOT resubmitting (would loop)" >&2; return
  fi
  local n; n=$(sbatch --parsable jobs/branches_3seed.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n ($d/$TOTAL_CELLS cells)" \
    || echo "$L resubmit FAILED: $n" >&2; }
trap 'echo "$L USR1 backstop (explicit guards should have handed off already)"; resub; exit 0' USR1

# ── training ──────────────────────────────────────────────────────────────────
train_one(){   # branch seed
  local br="$1" sd="$2" exp patch
  exp="${br}_strict_test50_1300_s${sd}"; patch="outputs/${exp}/${CKPT}.png"
  [[ -f "$patch" ]] && return 0
  if (( SECONDS > WALL_LIMIT - RESERVE_TRAIN )); then
    echo "$L wall: $((wall_left)/60)m left, a training needs ~$((RESERVE_TRAIN/60))m - handing off"
    resub; exit 0
  fi
  if compgen -G "outputs/${exp}/*.png" >/dev/null; then
    echo "$L $exp PARTIAL and no ${CKPT}.png - discarding (--base_epoch loses Adam state)"
    rm -f outputs/${exp}/*.png outputs/${exp}/losses.csv
  fi
  echo ""; echo "$L ===== train $exp (seed $sd) ====="
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
a=np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(float)/255.0
print(f"[br] {sys.argv[2]}: std={a.std():.4f} range=[{a.min():.3f},{a.max():.3f}]")
PY
  echo "$L trained $exp"
}

# ── evaluation ────────────────────────────────────────────────────────────────
git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py" || exit 1
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT
GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

mkcfg(){   # cfg dataset patch
  local cfg="$1" ds="$2" patch="$3"
  [[ -s "attack_configs/${cfg}.csv" ]] && return 0
  if [[ "$ds" == safebench ]]; then
    python create_attack_configs.py --dataset safebench --config-type attack \
      --exclude-train datasets/SafeBench-Tiny.csv --image "$patch" \
      --output "attack_configs/${cfg}.csv" || return 1
    python reproducibility/check_branch.py eval --config "$cfg" --kind safebench || return 1
  else
    python create_attack_configs.py --dataset advbench --config-type attack --normalize \
      --image "$patch" --output "attack_configs/${cfg}.csv" || return 1
    python reproducibility/check_branch.py eval --config "$cfg" --kind advbench || return 1
  fi
}

score(){   # cfg model
  local cfg="$1" m="$2"
  local MK="$MARK/$(echo "${cfg}_${m}" | tr '/' '_')"; [[ -f "$MK" ]] && return 0
  cell_fits "$cfg" "$m" || { resub; exit 0; }
  local RES="results/${cfg}/${m}.csv"
  if [[ ! -f "$RES" ]]; then
    local gpy; gpy="$(py_for_gen "$m")"; echo "$L gen $cfg on $m"
    "$gpy" evaluation/attack.py --model_name "$m" --attack_config "$cfg" || return 1
  fi
  [[ -f "$RES" ]] || return 1
  [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] || \
    python evaluation/evaluate.py --attack_result "$RES" --output_suffix "_harmbench_v3.csv" >/dev/null || true
  [[ -f "results/${cfg}/${m}_harmbench.csv" ]] || \
    ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$RES" ) >/dev/null || true
  [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] && { touch "$MK"; echo "$L cell $(done_n)/$TOTAL_CELLS done: $cfg/$m"; }
}

# branch-major, then seed, then cheapest model: partial completion leaves COMPLETE rows
for br in cf pf; do
  for sd in "${SEEDS[@]}"; do
    exp="${br}_strict_test50_1300_s${sd}"; patch="outputs/${exp}/${CKPT}.png"
    train_one "$br" "$sd" || continue
    if [[ "$br" == pf ]]; then
      python reproducibility/check_branch.py pf --exp "$exp" || {
        echo "$L CONTRACT VIOLATED by $exp - refusing to evaluate it" >&2; continue; }
    fi
    SB="${br}s_sb315_s${sd}"; AB="${br}s_abnorm_s${sd}"
    mkcfg "$SB" safebench "$patch" || continue
    mkcfg "$AB" advbench  "$patch" || continue
    for m in "${MODELS[@]}"; do
      score "$SB" "$m"
      score "$AB" "$m"
    done
  done
done

# ── report: mean +/- spread across the three seeds ────────────────────────────
echo ""; echo "$L ===== BOTH BRANCHES, 3 SEEDS, mean +/- spread ====="
python - <<'PY'
import os, pandas as pd, numpy as np
M=[("Qwen2-VL*","Qwen/Qwen2-VL-7B-Instruct"),("Qwen-VL-Chat","Qwen/Qwen-VL-Chat"),
   ("Qwen2.5-VL","Qwen/Qwen2.5-VL-7B-Instruct"),("LLaVA-1.6","llava-hf/llava-v1.6-mistral-7b-hf"),
   ("Kimi-VL","moonshotai/Kimi-VL-A3B-Instruct"),("GLM-4.1V","THUDM/GLM-4.1V-9B-Thinking")]
paper={"SafeBench":[81.59,72.70,60.32,88.25,67.94,66.03],
       "AdvBench":[72.69,71.92,35.77,92.88,30.38,30.00]}
def asr(cfg,m,suf):
    p=f"results/{cfg}/{m}_{suf}.csv"
    if not os.path.exists(p): return None
    d=pd.read_csv(p); return 100*d.attack_success.astype(bool).mean()
for bench,tag in (("SafeBench","sb315"),("AdvBench","abnorm")):
    print(f"\n  ===== {bench} =====")
    for suf,sn in (("harmbench","v1 (the paper's scorer)"),("harmbench_v3","v3 (ours)")):
        print(f"\n  -- {sn} --")
        print(f"  {'':16s}" + "".join(f"{n:>22s}" for n,_ in M))
        for br,lbl in (("cf","code-faithful"),("pf","paper-faithful")):
            cells=[]
            for _,m in M:
                vs=[asr(f"{br}s_{tag}_s{s}",m,suf) for s in (0,1,2)]
                vs=[v for v in vs if v is not None]
                if not vs: cells.append(f"{'.':>22s}")
                elif len(vs)==1: cells.append(f"{vs[0]:>14.2f} (n=1)")
                else:
                    cells.append(f"{np.mean(vs):>10.2f} +/-{(max(vs)-min(vs))/2:>5.2f} (n={len(vs)})")
            print(f"  {lbl:16s}" + "".join(cells))
        print(f"  {'paper':16s}" + "".join(f"{v:>22.2f}" for v in paper[bench]))
PY

echo ""; echo "$L cells: $(done_n)/$TOTAL_CELLS   trainings: $(ls -d outputs/{cf,pf}_strict_test50_1300_s*/ 2>/dev/null | wc -l)/6"
if (( $(done_n) >= TOTAL_CELLS )); then
  echo "$L COMPLETE - both branches, three seeds, both scorers."
  python reproducibility/check_branch.py seeds --require-trained || true
else
  resub end
fi
echo "$L finished $(date)"
