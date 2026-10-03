#!/usr/bin/env bash
#SBATCH --job-name=ub_glmre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@900
#
# GLM re-run at the MEASURED token budget (4096; probe max was 2372 on FigStep).
# Defect D14: GLM-4.1V-Thinking was truncated mid-<think> at 512 tokens, so every
# GLM number is invalid. This regenerates + re-judges (v1 and v3) every GLM cell.
# Priority-ordered: conclusion-critical configs first. Resume-guarded via markers,
# self-chaining across 24h windows.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate; export PYTHONUNBUFFERED=1
GLM="THUDM/GLM-4.1V-9B-Thinking"; GLM_PY="repro_glm/bin/python"
AUTH="c4c276d"; L="[glmre]"; MARK="results/.glm_rerun4096"; mkdir -p "$MARK" logs
STAGE="${SLURM_TMPDIR:-/tmp}/glmre_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE"
ATTEMPT="${1:-1}"; MAX=20
echo "$L start $(date) attempt $ATTEMPT/$MAX (GLM budget 4096)"

# TIER 1 = conclusion-critical, TIER 2 = completeness
ORDER=(
  xfer_projoff_safebench xfer_projon_safebench xfer_control_safebench
  noattack_safebench_matrix
  t1mm_cf_mmsafety t1mm_pf_mmsafety
  figstep_safebench
  b15_jbm_safebench b15_access_safebench b15_classified_safebench
  xfer_projoff_advbench xfer_projon_advbench xfer_control_advbench
  noattack_advbench_matrix
  cmp_cf_sb315 cmp_pf_sb315 cmp_img_sb315
  cmp_cf_abraw cmp_pf_abraw cmp_img_abraw
  cmp_cf_abnorm cmp_pf_abnorm cmp_img_abnorm
  cmp_cf_sb350 cmp_pf_sb350 cmp_img_sb350
  a4mm_809_mmsafety authors_mmsafety_matrix
  safebench_jailbroken_mode safebench_jailbroken_mode_1300
  safebench_jailbroken_mode_5000 advbench_jailbroken_mode_1300
)
TOTAL=${#ORDER[@]}
# Wall guard: configs take 3-6h and resume is per-config, so do NOT start one we
# cannot finish in this 24h window - hand off instead of losing partial work.
WALL_LIMIT=86400   # 24h
RESERVE=25200      # 7h headroom for the slowest single config
done_n(){ ls "$MARK" 2>/dev/null | wc -l; }
START_DONE=$(done_n); echo "$L progress $START_DONE/$TOTAL configs"
if (( START_DONE >= TOTAL )); then echo "$L COMPLETE"; exit 0; fi

RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d; d=$(done_n)
  (( d >= TOTAL )) && { echo "$L COMPLETE ($d/$TOTAL)"; return; }
  (( ATTEMPT >= MAX )) && { echo "$L MAX_ATTEMPTS at $d/$TOTAL"; return; }
  [[ "$1" == end ]] && (( d <= START_DONE )) && { echo "$L no progress ($d<=$START_DONE); stopping"; return; }
  local n; n=$(sbatch --parsable jobs/glm_rerun.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n ($d/$TOTAL)" || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1 wall approaching"; resub timeout; exit 0' USR1

# authors' v1 judge, materialised from git
git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py"
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT

for cfg in "${ORDER[@]}"; do
  if (( SECONDS > WALL_LIMIT - RESERVE )); then
    echo "$L wall budget nearly spent (${SECONDS}s elapsed) - handing off before starting $cfg"
    resub timeout; exit 0
  fi
  [[ -f "$MARK/$cfg" ]] && { echo "$L skip (done) $cfg"; continue; }
  [[ -s "attack_configs/${cfg}.csv" ]] || { echo "$L SKIP $cfg: no attack_configs/${cfg}.csv"; continue; }
  echo "$L === $cfg : regenerating GLM at 4096 ==="
  res="results/${cfg}/${GLM}.csv"
  # drop stale truncated outputs for GLM only
  rm -f "$res" "results/${cfg}/${GLM}_harmbench.csv" "results/${cfg}/${GLM}_harmbench_v3.csv"
  "$GLM_PY" evaluation/attack.py --model_name "$GLM" --attack_config "$cfg" || { echo "$L gen FAILED $cfg"; continue; }
  [[ -f "$res" ]] || { echo "$L no output for $cfg"; continue; }
  # judge: v3 (ours) then v1 (authors')
  extra=""; [[ "$cfg" == *advbench* ]] && extra="--behaviors_csv datasets/adv_bench.csv --behaviors_col goal"
  python evaluation/evaluate.py --attack_result "$res" --output_suffix "_harmbench_v3.csv" $extra || echo "$L v3 judge failed $cfg"
  ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$res" ) | tail -1 || echo "$L v1 judge failed $cfg"
  touch "$MARK/$cfg"
  echo "$L done $cfg  ($(done_n)/$TOTAL)"
done

echo "$L ====== GLM re-run summary (4096) ======"
python - <<'PY'
import glob,os,pandas as pd
print(f"{'config':34s} {'n':>5s} {'valid':>6s} {'%ok':>5s} {'ASRv1':>7s} {'ASRv3':>7s}")
for p in sorted(glob.glob("results/*/THUDM/GLM-4.1V-9B-Thinking_harmbench_v3.csv")):
    cfg=os.path.basename(os.path.dirname(os.path.dirname(p)))
    if not os.path.exists(f"results/.glm_rerun4096/{cfg}"): continue
    d=pd.read_csv(p); ok=d[d.extraction_status=="ok"] if "extraction_status" in d else d
    v1p=p.replace("_harmbench_v3.csv","_harmbench.csv")
    v1=pd.read_csv(v1p).attack_success.astype(bool).mean()*100 if os.path.exists(v1p) else float('nan')
    print(f"{cfg:34s} {len(d):>5} {len(ok):>6} {100*len(ok)/max(len(d),1):>5.0f} {v1:>7.1f} {d.attack_success.astype(bool).mean()*100:>7.1f}")
PY
resub end
echo "$L done $(date)"
