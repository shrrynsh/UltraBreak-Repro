#!/usr/bin/env bash
#SBATCH --job-name=ub_shift
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@900
#
# Query-distribution-shift study (aaai_ugc) — E1 format sweep, E2 forced-prefix
# ablation, E3 source shift. Inference only: no training, no new patches.
#
#   E1  150 AdvBench intents x {bare, steps, question} x {base, shuffled,
#       authors, p809, text_only}                                  2250 gens
#   E2  SafeBench-315 x {TPG on, off} x {base, authors}            1260 gens
#   E3  steps form x {safebench, advbench, harmbench} x {base}     1035 gens
#                                                          total = 4545 gens
#
# The `authors` arms of E3 are REUSED from existing runs (cmp_img_sb315,
# cmp_img_abnorm, ext_authors_harmbench_norm_eval) — see aaai_ugc/PLAN.md §4.
#
# Everything is held at the job-809/683 protocol: Qwen2-VL-7B surrogate, the
# pinned `repro` venv, max_new_tokens=512 (the GLM-only 4096 bump of D14 does not
# apply here), judge v3 on every row plus v1 on the headline rows.
#
# Per-config resume markers, so a requeue never repeats finished work.
# Usage:  sbatch aaai_ugc/jobs/shift_study.sh [attempt]
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"

if [[ -f repro/bin/activate ]]; then
  source repro/bin/activate
else
  echo "[shift] repro venv missing - refusing to run under an unpinned env" >&2; exit 1
fi
export PYTHONUNBUFFERED=1

L="[shift]"; AUTH="c4c276d"; MODEL="Qwen/Qwen2-VL-7B-Instruct"
CFG_ROOT="aaai_ugc/configs"; GEN_ROOT="aaai_ugc/results/gen"
MARK="aaai_ugc/results/.markers"; mkdir -p "$MARK" logs
STAGE="${SLURM_TMPDIR:-/tmp}/shift_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE"
ATTEMPT="${1:-1}"; MAX=8

# E1 first (highest priority), then E2, then E3 — the order in PLAN.md §5.
ORDER=(
  E1_bare_base E1_bare_authors E1_bare_p809 E1_bare_shuffled E1_bare_text_only
  E1_steps_base E1_steps_authors E1_steps_p809 E1_steps_shuffled E1_steps_text_only
  E1_question_base E1_question_authors E1_question_p809 E1_question_shuffled E1_question_text_only
  E2_tpgon_base E2_tpgon_authors E2_tpgoff_base E2_tpgoff_authors
  E3_safebench_base E3_advbench_base E3_harmbench_base
)
# Configs that also get the authors' v1 judge (headline rows).
V1_SET=" E1_bare_authors E1_bare_p809 E1_steps_authors E1_steps_p809 \
E1_question_authors E1_question_p809 E2_tpgon_base E2_tpgon_authors \
E2_tpgoff_base E2_tpgoff_authors E3_safebench_base E3_advbench_base E3_harmbench_base "
TOTAL=${#ORDER[@]}

echo "$L start $(date) attempt $ATTEMPT/$MAX"
python - <<'PY'
import importlib.metadata as m
pins = ["torch","torchvision","transformers","tokenizers","qwen-vl-utils","accelerate","numpy","pandas","Pillow"]
print("[shift] Versions: " + ", ".join(f"{p}=={m.version(p)}" for p in pins))
import torch
assert torch.__version__.startswith("2.5.1"), f"torch pin broken: {torch.__version__}"
assert m.version("transformers").startswith("4.51"), "transformers pin broken"
print("[shift] env pins OK")
PY
[[ $? -eq 0 ]] || { echo "$L env assertion failed" >&2; exit 1; }

# Protocol guard: the Qwen2-VL branch must still be at 512 new tokens, i.e. the
# D14 GLM fix did not leak into this model's path and change the protocol.
python - <<'PY'
import re, sys
src = open("evaluation/attack.py").read()
i = src.index('elif model_name == "Qwen/Qwen2-VL-7B-Instruct":')
j = src.index('elif model_name.startswith("Qwen/Qwen2.5-VL-")', i)
got = re.findall(r"max_new_tokens=(\d+)", src[i:j])
assert got == ["512"], f"Qwen2-VL branch token budget changed: {got}"
print("[shift] Qwen2-VL max_new_tokens=512 confirmed (job 809/683 protocol)")
PY
[[ $? -eq 0 ]] || { echo "$L protocol assertion failed" >&2; exit 1; }

for f in images/white.jpeg outputs/ultrabreak.png \
         outputs/full50_normfix_lr0p01_s3000/3000.png \
         aaai_ugc/images/shuffled_ultrabreak_seed0.png; do
  [[ -s "$f" ]] || { echo "$L missing image $f" >&2; exit 1; }
done
echo "$L images present; md5s:"; md5sum images/white.jpeg outputs/ultrabreak.png \
  outputs/full50_normfix_lr0p01_s3000/3000.png aaai_ugc/images/shuffled_ultrabreak_seed0.png

# v1 judge, materialised from the authors' commit (identical to jobs/t1_*.sh).
git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py" || {
  echo "$L cannot materialise v1 judge" >&2; exit 1; }
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
cleanup(){ rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"; }
trap cleanup EXIT


# ── chain to the next stage ────────────────────────────────────────────────────
# The cluster allows one running job, so the night's work is a chain: each stage
# self-resumes until its own work is complete, then seeds the next stage once.
# Guarded against double-seeding by checking the queue for the successor's name.
chain_to(){   # script jobname
  local script="$1" name="$2"
  [[ -f "$script" ]] || { echo "$L chain: $script not found" >&2; return; }
  if squeue -u "${USER:-$(id -un)}" -h -o "%j" 2>/dev/null | grep -qx "$name"; then
    echo "$L chain: $name already queued/running - not seeding again"; return
  fi
  local n; n=$(sbatch --parsable "$script" 2>&1) \
    && echo "$L CHAIN -> $script as job $n" \
    || echo "$L chain submit FAILED: $n" >&2
}

done_n(){ ls "$MARK" 2>/dev/null | wc -l; }
START_DONE=$(done_n); echo "$L progress $START_DONE/$TOTAL configs done"
if (( START_DONE >= TOTAL )); then
  echo "$L COMPLETE"
  python aaai_ugc/code/analyse_shift.py || true
  chain_to jobs/t1_noattack_mmsafety.sh ub_t1nam
  exit 0
fi

RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d; d=$(done_n)
  (( d >= TOTAL )) && { echo "$L COMPLETE"; return; }
  (( ATTEMPT >= MAX )) && { echo "$L MAX attempts reached at $d/$TOTAL"; return; }
  [[ "$1" == end ]] && (( d <= START_DONE )) && { echo "$L no progress - not resubmitting"; return; }
  local n; n=$(sbatch --parsable aaai_ugc/jobs/shift_study.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n ($d/$TOTAL)" \
    || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1 - pre-wall handoff"; resub timeout; exit 0' USR1

# Wall guard: biggest single config is 520 gens (~18 min gen + ~9 min judge);
# keep 90 min so a config is never started that cannot finish.
WALL_LIMIT=57600   # 16h
RESERVE=5400       # 90 min

gen_complete(){   # cfg -> 0 if generation already complete for every config row
  python - "$CFG_ROOT/$1.csv" "$GEN_ROOT/$1/$MODEL.csv" <<'PY'
import sys, os, pandas as pd
cfg, res = sys.argv[1], sys.argv[2]
if not os.path.exists(res): sys.exit(1)
c = pd.read_csv(cfg); r = pd.read_csv(res)
if "response" not in r.columns: sys.exit(1)
ne = r["response"].astype(str).str.strip().replace("nan", "").astype(bool).sum()
sys.exit(0 if (len(r) >= len(c) and ne >= len(c)) else 1)
PY
}

for cfg in "${ORDER[@]}"; do
  [[ -f "$MARK/$cfg" ]] && { echo "$L skip $cfg (done)"; continue; }
  if (( SECONDS > WALL_LIMIT - RESERVE )); then
    echo "$L wall budget nearly spent (${SECONDS}s) - handing off before $cfg"
    resub timeout; exit 0
  fi
  [[ -s "$CFG_ROOT/$cfg.csv" ]] || { echo "$L MISSING config $CFG_ROOT/$cfg.csv" >&2; continue; }
  NROW=$(( $(wc -l < "$CFG_ROOT/$cfg.csv") - 1 ))
  echo ""; echo "$L ===== $cfg ($NROW rows) ====="
  T0=$SECONDS

  if gen_complete "$cfg"; then
    echo "$L reuse existing generations for $cfg"
  else
    python evaluation/attack.py --model_name "$MODEL" --attack_config "$cfg" \
      --attack_root "$CFG_ROOT" --save_path "$GEN_ROOT" \
      || { echo "$L generation FAILED for $cfg - leaving unmarked" >&2; continue; }
  fi
  RES="$GEN_ROOT/$cfg/$MODEL.csv"
  [[ -f "$RES" ]] || { echo "$L no result file for $cfg" >&2; continue; }

  # judge v3 (ours, corrected) on every row
  if [[ ! -f "$GEN_ROOT/$cfg/${MODEL}_harmbench_v3.csv" ]]; then
    python evaluation/evaluate.py --attack_result "$RES" \
      --output_suffix "_harmbench_v3.csv" >/dev/null \
      || echo "$L v3 judge FAILED for $cfg" >&2
  else echo "$L v3 already scored"; fi

  # judge v1 (authors') on the headline rows
  if [[ "$V1_SET" == *" $cfg "* ]]; then
    if [[ ! -f "$GEN_ROOT/$cfg/${MODEL}_harmbench.csv" ]]; then
      ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$RES" ) >/dev/null \
        || echo "$L v1 judge FAILED for $cfg" >&2
    else echo "$L v1 already scored"; fi
  fi

  if [[ -f "$GEN_ROOT/$cfg/${MODEL}_harmbench_v3.csv" ]]; then
    touch "$MARK/$cfg"
    echo "$L done $cfg in $((SECONDS-T0))s  ($(done_n)/$TOTAL)"
  else
    echo "$L $cfg incomplete - not marking" >&2
  fi
done

# One extra: give the REUSED HarmBench authors arm a v1 pass so E3's v1 column is
# complete. Re-judges an existing generation; no new generation.
HB_AUTH="results/ext_authors_harmbench_norm_eval/$MODEL.csv"
if [[ -f "$HB_AUTH" && ! -f "results/ext_authors_harmbench_norm_eval/${MODEL}_harmbench.csv" ]]; then
  echo ""; echo "$L v1 re-score of reused arm ext_authors_harmbench_norm_eval"
  ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$HB_AUTH" ) >/dev/null \
    || echo "$L v1 re-score failed" >&2
fi

echo ""; echo "$L ===== analysis ====="
python aaai_ugc/code/analyse_shift.py || echo "$L analysis failed (results are on disk)" >&2

if (( $(done_n) >= TOTAL )); then
  echo "$L ALL $TOTAL CONFIGS COMPLETE - handing the GPU to the next stage"
  chain_to jobs/t1_noattack_mmsafety.sh ub_t1nam
else
  resub end
fi
echo "$L finished $(date)  $(done_n)/$TOTAL configs"
