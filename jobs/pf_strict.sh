#!/usr/bin/env bash
#SBATCH --job-name=ub_pfstrict
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@1200
#
# STRICT PAPER-FAITHFUL retrain + paper-protocol evaluation.  §3.2 = Reading A.
#
# WHY. `repro_paper_faithful_1300` is correct on PF1-PF4, PF6 and PF7 but was trained
# on `safebench-tiny_jailbroken_mode_full50.csv`, a corpus WE generated. The strict CF
# run (job 9243) trains on the authors' own `./train_configs/test.csv`. Leaving PF on
# our file would make the two branches differ on TWO axes - the §3.2 treatment AND a
# `keywords` column that disagrees on 47/50 rows and drives the semantic-ATTENTION
# loss - so a CF-vs-PF comparison would be confounded. PF5 of the contract now
# requires the authors' corpus on both branches, and `test.csv` IS "SafeBench-Tiny,
# 50 queries", so this satisfies the paper as well as de-confounding the comparison.
#
#   AFTER THIS JOB: CF and PF differ in exactly ONE thing - whether §3.2's projection
#   is an active compression step. That is the only axis the two branches exist to test.
#
# CONTRACT: reproducibility/BRANCH_CONTRACT.md (PF = Reading A, the literal §3.2).
#   PF1 projection ACTIVE      PF5 corpus = authors' ./train_configs/test.csv
#   PF2 normalise_patch ON     PF6 surrogate Qwen2-VL-7B
#   PF3 1300 iterations        PF7 seed 0 (declared deviation: the paper seeds none)
#   PF4 Table 4 verbatim
# Evaluation: E1 315 / E2 520 §3.3 form / E4 one patch / E5 both scorers / E7 no
# released image. `check_branch.py` is run as a HARD GATE after training and before
# any evaluation - if the trained run violates the contract the job stops.
#
# NOTE ON WHAT THIS SUPERSEDES. Table 1's PF rows currently come from
# `repro_paper_faithful_1300`. They must be re-read from `pf_strict_test50_1300`
# once this lands; the old numbers stay in TABLES.md marked superseded, not deleted.
#
# Cost: ~5.2 GPU-h training + ~9 GPU-h eval = ~15h, one window.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate || { echo "[pfs] repro venv missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1
L="[pfs]"; AUTH="c4c276d"
EXP="pf_strict_test50_1300"; CKPT=1300
PATCH="outputs/${EXP}/${CKPT}.png"
MARK="results/.pfstrict"; mkdir -p "$MARK" logs
STAGE="${SLURM_TMPDIR:-/tmp}/pfs_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE"
ATTEMPT="${1:-1}"; MAX=6
SB="pfs_sb315"; AB="pfs_abnorm"
MODELS=(Qwen/Qwen2-VL-7B-Instruct Qwen/Qwen-VL-Chat Qwen/Qwen2.5-VL-7B-Instruct
        llava-hf/llava-v1.6-mistral-7b-hf moonshotai/Kimi-VL-A3B-Instruct
        THUDM/GLM-4.1V-9B-Thinking)
TOTAL_CELLS=12

echo "$L start $(date) attempt $ATTEMPT/$MAX"

# ── Stage 0: pre-training guards ──────────────────────────────────────────────
python - <<'PY' || exit 1
import importlib.metadata as m, importlib.util, torch, subprocess, sys, os
assert torch.__version__.startswith("2.5.1"), torch.__version__
assert m.version("transformers").startswith("4.51"), m.version("transformers")
HERE="reproducibility/paper_faithful"
sys.path.insert(0, os.path.join(HERE,"optimisation"))
spec=importlib.util.spec_from_file_location("ops", f"{HERE}/optimise_paper_strict.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod._assert_table4()          # PF1-PF4, PF6, PF7 + the PF5 blob check
P=mod.PAPER
assert P.exp_name=="pf_strict_test50_1300" and P.train_config=="./train_configs/test.csv"
assert P.no_projection is False, "PF1: the projection must be ACTIVE (Reading A)"
assert P.raw_patch_injection is False, "PF2: the patch must be normalised into CLIP space"
assert P.num_epochs==1301 and P.seed==0
print("[pfs] PRE-TRAIN GUARDS OK: Reading A, Table 4, authors' corpus")
PY

WALL_LIMIT=86400
RESERVE_TRAIN=23400
rate_for(){ case "$1" in
    *GLM-4.1V*)      echo 29.0 ;;
    *Kimi-VL*)       echo 2.5 ;;
    *llava*)         echo 2.2 ;;
    *Qwen-VL-Chat*)  echo 1.4 ;;
    *)               echo 1.6 ;;
  esac; }
cell_fits(){
  local cfg="$1" m="$2"
  local rows; rows=$(( $(wc -l < "attack_configs/${cfg}.csv") - 1 ))
  local est; est=$(python -c "print(int($rows*$(rate_for "$m")+900))")
  if (( SECONDS + est > WALL_LIMIT - 600 )); then
    echo "$L wall: ${cfg}/${m} needs ~$((est/60))m, only $(( (WALL_LIMIT-SECONDS)/60 ))m left - handing off"
    return 1
  fi
  return 0
}
done_n(){ ls "$MARK" 2>/dev/null | wc -l; }
START=$(done_n); RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d; d=$(done_n)
  (( d >= TOTAL_CELLS )) && { echo "$L COMPLETE $d/$TOTAL_CELLS"; return; }
  (( ATTEMPT >= MAX )) && { echo "$L MAX attempts at $d/$TOTAL_CELLS"; return; }
  [[ "${1:-}" == end ]] && (( d <= START )) && [[ -f "$PATCH" ]] && { echo "$L no progress"; return; }
  local n; n=$(sbatch --parsable jobs/pf_strict.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n ($d/$TOTAL_CELLS)" || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1 pre-wall handoff"; resub; exit 0' USR1

# ── Stage 1: train ────────────────────────────────────────────────────────────
if [[ ! -f "$PATCH" ]]; then
  if (( SECONDS > WALL_LIMIT - RESERVE_TRAIN )); then
    echo "$L wall: not enough left to train - handing off"; resub; exit 0
  fi
  if compgen -G "outputs/${EXP}/*.png" >/dev/null; then
    echo "$L PARTIAL training and no ${CKPT}.png - discarding (--base_epoch loses Adam state)"
    rm -f outputs/${EXP}/*.png outputs/${EXP}/losses.csv
  fi
  echo ""; echo "$L --- train $EXP : Reading A, authors' corpus, 1300 iters ---"
  python reproducibility/paper_faithful/optimise_paper_strict.py \
    || { echo "$L TRAIN FAILED" >&2; exit 1; }
  [[ -f "$PATCH" ]] || { echo "$L training produced no ${CKPT}.png" >&2; exit 1; }
  python - "$PATCH" <<'PY'
import sys, numpy as np
from PIL import Image
a=np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(float)/255.0
print(f"[pfs] patch: std={a.std():.4f} range=[{a.min():.3f},{a.max():.3f}]")
print("[pfs] EXPECTED under Reading A: range inside gamma*[0,1]+beta = [0.478,0.741], std ~0.025.")
print("[pfs] The released ultrabreak.png spans [0,1] with std ~0.165 - 6.6x more contrast.")
print("[pfs] That bound is ARITHMETIC, not a bug: it is the finding the literal reading produces.")
PY
else
  echo "$L training already done ($PATCH)"
fi

# ── HARD GATE: the trained run must satisfy the contract before we evaluate ───
python reproducibility/check_branch.py pf --exp "$EXP" || {
  echo "$L CONTRACT VIOLATED by the trained run - refusing to evaluate" >&2; exit 1; }

# ── Stage 2: evaluation configs, PAPER protocol ───────────────────────────────
[[ -s "attack_configs/${SB}.csv" ]] || python create_attack_configs.py \
  --dataset safebench --config-type attack --exclude-train datasets/SafeBench-Tiny.csv \
  --image "$PATCH" --output "attack_configs/${SB}.csv" || exit 1
[[ -s "attack_configs/${AB}.csv" ]] || python create_attack_configs.py \
  --dataset advbench --config-type attack --normalize \
  --image "$PATCH" --output "attack_configs/${AB}.csv" || exit 1
python reproducibility/check_branch.py eval --config "$SB" --kind safebench || exit 1
python reproducibility/check_branch.py eval --config "$AB" --kind advbench  || exit 1

# ── Stage 3+4: generate and score under BOTH scorers ──────────────────────────
git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py" || exit 1
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT
GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

score(){
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
  [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] && { touch "$MK"; echo "$L cell done $cfg/$m ($(done_n)/$TOTAL_CELLS)"; }
}

for m in "${MODELS[@]}"; do
  echo ""; echo "$L ########## $m ##########"
  score "$SB" "$m"
  score "$AB" "$m"
done

# ── Stage 5: report ───────────────────────────────────────────────────────────
echo ""; echo "$L ===== STRICT paper-faithful (Reading A), paper-protocol eval ====="
python - <<PY
import os, pandas as pd
M=[("Qwen2-VL*","Qwen/Qwen2-VL-7B-Instruct"),("Qwen-VL-Chat","Qwen/Qwen-VL-Chat"),
   ("Qwen2.5-VL","Qwen/Qwen2.5-VL-7B-Instruct"),("LLaVA-1.6","llava-hf/llava-v1.6-mistral-7b-hf"),
   ("Kimi-VL","moonshotai/Kimi-VL-A3B-Instruct"),("GLM-4.1V","THUDM/GLM-4.1V-9B-Thinking")]
paper={"SafeBench":[81.59,72.70,60.32,88.25,67.94,66.03],
       "AdvBench":[72.69,71.92,35.77,92.88,30.38,30.00]}
sup={"SafeBench":[64.13,68.89,26.03,87.62,60.95,60.95],   # repro_paper_faithful_1300, v3
     "AdvBench":[10.19,75.00,1.73,91.15,29.62,19.81]}
def asr(cfg,m,suf):
    p=f"results/{cfg}/{m}_{suf}.csv"
    if not os.path.exists(p): return None
    d=pd.read_csv(p); return 100*d.attack_success.astype(bool).mean()
for bench,cfg in (("SafeBench","${SB}"),("AdvBench","${AB}")):
    print(f"\n  -- {bench} --   (v1 = the paper's own scorer)")
    print(f"  {'':26s}" + "".join(f"{n:>15s}" for n,_ in M))
    for lbl,suf in (("strict PF v1","harmbench"),("strict PF v3","harmbench_v3")):
        row=f"  {lbl:26s}"
        for _,m in M:
            v=asr(cfg,m,suf); row+=f"{(f'{v:.2f}' if v is not None else '.'):>15s}"
        print(row)
    print(f"  {'superseded PF (our corpus)':26s}" + "".join(f"{v:>15.2f}" for v in sup[bench]))
    print(f"  {'paper':26s}" + "".join(f"{v:>15.2f}" for v in paper[bench]))
PY

echo ""; echo "$L markers: $(done_n)/$TOTAL_CELLS"
if (( $(done_n) >= TOTAL_CELLS )); then
  echo "$L COMPLETE - strict paper-faithful branch done."
else
  resub end
fi
echo "$L finished $(date)"
