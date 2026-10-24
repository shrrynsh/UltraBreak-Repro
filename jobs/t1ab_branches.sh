#!/usr/bin/env bash
#SBATCH --job-name=ub_t1ab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#
# STAGE A — Table 1a (SafeBench 315) + Table 1b (AdvBench 520), both branches, 3 seeds.
# ====================================================================================
# 72 cells = 2 branches x 3 seeds x 6 models x 2 benchmarks.  Measured 77.3 GPU-h, ~4 windows.
# EVALUATION ONLY. The six patches were trained by jobs/retrain_6patches.sh (9248 + 9271).
#
# Contract: reproducibility/BRANCH_CONTRACT.md.  Every config is gated by
#   check_branch.py eval --arm <cf|pf> --seed <n>
# BEFORE it is used, so E4b/E4c bind it to its own patch by sha256. Without that gate a
# config built on a superseded artifact passes every other check (the P1 defect, found
# 2026-10-24 and fixed the same day).
#
# The four failure modes this guards, and how:
#  1 wall clock runs out mid-cell -> attack.py writes its CSV only after a whole config,
#    so a killed cell loses ALL its rows. Guard: per-cell estimate from MEASURED rates;
#    a cell starts only if the whole cell fits. Largest cell is GLM/AdvBench at 2.5 h.
#  2 a window ends -> per-cell markers, so a resume redoes one cell, never the job.
#  3 the resume never works -> RESUB latch, and `resub end` refuses if no cell completed.
#  4 the wrong patch gets evaluated -> E4b/E4c, above.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
REPO="$PWD"; L="[t1ab]"; AUTH="c4c276d"
MARK="results/.t1ab"; mkdir -p "$MARK" logs
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
ATTEMPT="${1:-1}"; MAX=12
WALL_LIMIT=86400
echo "$L start $(date)  attempt ${ATTEMPT}/${MAX}"

source repro/bin/activate || { echo "$L repro venv missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1

SEEDS=( 0 1 2 )
ARMS=( cf pf )
# cheap models first, GLM last: GLM is ~35% of this stage on its own
MODELS=( "Qwen/Qwen2-VL-7B-Instruct" "Qwen/Qwen-VL-Chat" "Qwen/Qwen2.5-VL-7B-Instruct" \
         "llava-hf/llava-v1.6-mistral-7b-hf" "moonshotai/Kimi-VL-A3B-Instruct" \
         "THUDM/GLM-4.1V-9B-Thinking" )
TOTAL=$(( ${#ARMS[@]} * ${#SEEDS[@]} * ${#MODELS[@]} * 2 ))

# ── Stage 0: contract gate, before the GPU is touched ───────────────────────────
echo ""; echo "$L ===== stage 0: contract gate ====="
python reproducibility/check_branch.py seeds --require-trained || exit 1
for s in "${SEEDS[@]}"; do
  python reproducibility/check_branch.py cf --seed "$s" --require-trained || exit 1
  python reproducibility/check_branch.py pf --exp "pf_strict_test50_1300_s${s}" --seed "$s" --require-trained || exit 1
done
[[ -s reproducibility/patch_pins.json ]] || { echo "$L no patch_pins.json - run check_branch.py seeds --write-pins" >&2; exit 1; }

# ── config construction: one per (arm, seed, benchmark) ─────────────────────────
cfg_name(){ echo "t1_${1}_s${2}_${3}"; }   # arm seed bench
patch_for(){ echo "outputs/${1}_strict_test50_1300_s${2}/1300.png"; }

mkcfg(){   # arm seed bench
  local arm="$1" sd="$2" bench="$3"
  local cfg; cfg="$(cfg_name "$arm" "$sd" "$bench")"
  local img; img="$(patch_for "$arm" "$sd")"
  if [[ ! -s "attack_configs/${cfg}.csv" ]]; then
    if [[ "$bench" == sb315 ]]; then
      python create_attack_configs.py --dataset safebench --config-type attack \
        --exclude-train datasets/SafeBench-Tiny.csv --image "$img" \
        --output "attack_configs/${cfg}.csv" >/dev/null || return 1
    else
      python create_attack_configs.py --dataset advbench --config-type attack \
        --normalize --image "$img" --output "attack_configs/${cfg}.csv" >/dev/null || return 1
    fi
  fi
  # THE gate: binds this config to this arm+seed's patch by sha256 (E4b/E4c)
  local kind; [[ "$bench" == sb315 ]] && kind=safebench || kind=advbench
  python reproducibility/check_branch.py eval --config "$cfg" --kind "$kind" \
    --arm "$arm" --seed "$sd" >/dev/null || {
      echo "$L CONTRACT FAIL on ${cfg} - refusing to evaluate it" >&2
      python reproducibility/check_branch.py eval --config "$cfg" --kind "$kind" --arm "$arm" --seed "$sd" >&2
      return 1; }
  echo "$cfg"
}

echo ""; echo "$L ===== building and gating 12 configs ====="
declare -A CFG
for arm in "${ARMS[@]}"; do for sd in "${SEEDS[@]}"; do for bench in sb315 ab520; do
  c="$(mkcfg "$arm" "$sd" "$bench")" || exit 1
  CFG["${arm}_${sd}_${bench}"]="$c"
  echo "$L gated ${c}"
done; done; done

# ── wall-clock accounting from MEASURED rates (repro_notes / REPRO_PLAN §4) ─────
rate_for(){   # model bench -> seconds per row, incl. generation + both judges
  local m="$1" b="$2"
  case "$m" in
    *GLM-4.1V*)     [[ "$b" == sb315 ]] && echo 21.0 || echo 16.0 ;;
    *Kimi-VL*)      [[ "$b" == sb315 ]] && echo  9.0 || echo  7.0 ;;
    *llava*)        echo 10.0 ;;
    *Qwen-VL-Chat*) [[ "$b" == sb315 ]] && echo  3.5 || echo  3.6 ;;
    *Qwen2.5-VL*)   [[ "$b" == sb315 ]] && echo  5.0 || echo  4.3 ;;
    *)              [[ "$b" == sb315 ]] && echo  5.0 || echo  2.0 ;;
  esac; }
cell_fits(){   # cfg model bench
  local rows est
  rows=$(( $(wc -l < "attack_configs/${1}.csv") - 1 ))
  est=$(python -c "print(int($rows*$(rate_for "$2" "$3")+600))")
  if (( SECONDS + est > WALL_LIMIT - 600 )); then
    echo "$L wall: ${1}/${2} needs ~$((est/60))m, only $(( (WALL_LIMIT-SECONDS)/60 ))m left - handing off"
    return 1
  fi
  return 0
}

done_n(){ ls "$MARK" 2>/dev/null | wc -l; }
START=$(done_n); RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d; d=$(done_n)
  (( d >= TOTAL )) && { echo "$L COMPLETE $d/$TOTAL"; return; }
  (( ATTEMPT >= MAX )) && { echo "$L MAX attempts at $d/$TOTAL"; return; }
  [[ "${1:-}" == end ]] && (( d <= START )) && { echo "$L no progress this window - not resubmitting"; return; }
  local n; n=$(sbatch --parsable jobs/t1ab_branches.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n ($d/$TOTAL)" || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1 pre-wall handoff"; resub; exit 0' USR1

# ── judges: v3 from the tree, v1 staged from the authors' commit (E5) ───────────
git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py" || exit 1
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"; rm -rf "$STAGE"' EXIT
GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

score(){   # cfg model bench
  local cfg="$1" m="$2" bench="$3"
  local MK="$MARK/$(echo "${cfg}_${m}" | tr '/' '_')"; [[ -f "$MK" ]] && return 0
  cell_fits "$cfg" "$m" "$bench" || { resub; exit 0; }
  local RES="results/${cfg}/${m}.csv"
  if [[ ! -f "$RES" ]]; then
    local gpy; gpy="$(py_for_gen "$m")"
    echo "$L gen ${cfg} on ${m}"
    "$gpy" evaluation/attack.py --model_name "$m" --attack_config "$cfg" || return 1
  fi
  [[ -f "$RES" ]] || return 1
  [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] || \
    python evaluation/evaluate.py --attack_result "$RES" --output_suffix "_harmbench_v3.csv" >/dev/null || true
  [[ -f "results/${cfg}/${m}_harmbench.csv" ]] || \
    ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$RES" ) >/dev/null || true
  if [[ -f "results/${cfg}/${m}_harmbench_v3.csv" && -f "results/${cfg}/${m}_harmbench.csv" ]]; then
    touch "$MK"; echo "$L cell done ${cfg}/${m}  ($(done_n)/${TOTAL})"
  else
    echo "$L cell INCOMPLETE ${cfg}/${m} - one judge missing, not marking" >&2
  fi
}

echo ""; echo "$L ===== stage A: ${TOTAL} cells ====="
for m in "${MODELS[@]}"; do
  echo ""; echo "$L ########## $m ##########"
  for arm in "${ARMS[@]}"; do for sd in "${SEEDS[@]}"; do for bench in sb315 ab520; do
    score "${CFG[${arm}_${sd}_${bench}]}" "$m" "$bench" || echo "$L cell FAILED ${arm}/s${sd}/${bench}/${m}" >&2
  done; done; done
done

echo ""; echo "$L cells: $(done_n)/${TOTAL}"
if (( $(done_n) >= TOTAL )); then
  echo "$L ===== STAGE A COMPLETE ====="
  python reproducibility/audit_tables.py || true
else
  resub end
fi
echo "$L finished $(date)"
