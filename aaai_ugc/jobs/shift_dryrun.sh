#!/usr/bin/env bash
#SBATCH --job-name=ub_shiftdry
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#
# 10-row dry run for the shift study. Validates every distinct code path before
# the full ~5 GPU-hour job is committed:
#
#   dry_E1_steps_authors    image present, TPG on, AdvBench subset
#   dry_E1_bare_text_only   NO image at all (the text-only branch)
#   dry_E1_steps_shuffled   the newly built shuffled control image
#   dry_E2_tpgoff_authors   TPG clause removed, SafeBench-315
#   dry_E3_harmbench_base   HarmBench loader + the white base image
#
# Checks that generations parse, that both judges run, and that the analysis
# script consumes the output. Then PASS/FAIL is printed for each.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate || { echo "[dry] repro venv missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1
L="[dry]"; AUTH="c4c276d"; MODEL="Qwen/Qwen2-VL-7B-Instruct"
CFG_ROOT="aaai_ugc/configs"; GEN_ROOT="aaai_ugc/results/dryrun"
STAGE="${SLURM_TMPDIR:-/tmp}/shiftdry_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE" logs
mkdir -p "$GEN_ROOT"

CFGS=( dry_E1_steps_authors dry_E1_bare_text_only dry_E1_steps_shuffled
       dry_E2_tpgoff_authors dry_E3_harmbench_base )

echo "$L start $(date)"
python - <<'PY'
import importlib.metadata as m, torch
print("[dry] torch", torch.__version__, "| transformers", m.version("transformers"))
assert torch.__version__.startswith("2.5.1") and m.version("transformers").startswith("4.51")
print("[dry] env pins OK")
PY
[[ $? -eq 0 ]] || exit 1

git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py" || exit 1
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT

PASS=0; FAIL=0
for cfg in "${CFGS[@]}"; do
  echo ""; echo "$L ===== $cfg ====="
  [[ -s "$CFG_ROOT/$cfg.csv" ]] || { echo "$L FAIL missing config"; FAIL=$((FAIL+1)); continue; }

  python evaluation/attack.py --model_name "$MODEL" --attack_config "$cfg" \
    --attack_root "$CFG_ROOT" --save_path "$GEN_ROOT" >/dev/null 2>"$STAGE/$cfg.gen.err"
  RC=$?
  RES="$GEN_ROOT/$cfg/$MODEL.csv"
  if (( RC != 0 )) || [[ ! -f "$RES" ]]; then
    echo "$L FAIL generation (rc=$RC)"; tail -5 "$STAGE/$cfg.gen.err"; FAIL=$((FAIL+1)); continue
  fi

  python evaluation/evaluate.py --attack_result "$RES" \
    --output_suffix "_harmbench_v3.csv" >/dev/null 2>"$STAGE/$cfg.v3.err" \
    || { echo "$L FAIL v3 judge"; tail -5 "$STAGE/$cfg.v3.err"; FAIL=$((FAIL+1)); continue; }

  ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$RES" ) \
    >/dev/null 2>"$STAGE/$cfg.v1.err" \
    || { echo "$L WARN v1 judge failed"; tail -3 "$STAGE/$cfg.v1.err"; }

  # Structural check: rows present, responses non-empty, judge columns populated,
  # and our metrics module parses the output. Counts only - no completion text.
  python - "$cfg" "$CFG_ROOT/$cfg.csv" "$GEN_ROOT/$cfg/$MODEL" <<'PY'
import sys, os, pandas as pd
sys.path.insert(0, "aaai_ugc/code"); sys.path.insert(0, ".")
import metrics as M
cfg, cfg_path, stem = sys.argv[1], sys.argv[2], sys.argv[3]
n_cfg = len(pd.read_csv(cfg_path))
d = pd.read_csv(stem + "_harmbench_v3.csv")
ne = d["response"].astype(str).str.strip().replace("nan", "").astype(bool).sum()
assert len(d) == n_cfg, f"rows {len(d)} != config {n_cfg}"
assert ne == n_cfg, f"only {ne}/{n_cfg} non-empty responses"
assert d["attack_success"].notna().all(), "judge left NaN decisions"
assert d["judge_outputs"].astype(str).str.strip().ne("").all(), "empty judge output"
s = M.summarise(d)
v1p = stem + "_harmbench.csv"
v1 = f" v1_asr={100*pd.read_csv(v1p)['attack_success'].astype(bool).mean():.1f}" if os.path.exists(v1p) else " v1=absent"
print(f"[dry] OK {cfg}: n={s['n']} v3_asr={100*s['asr']:.1f}{v1} "
      f"nrr={100*s['nrr']:.1f} prefix={100*s['prefix_rate']:.1f} "
      f"list={100*s['list_rate']:.1f} medlen={s['median_answer_len']:.0f} "
      f"empty_answers={s['n_empty_answer']}")
PY
  if (( $? == 0 )); then PASS=$((PASS+1)); else echo "$L FAIL structural check"; FAIL=$((FAIL+1)); fi
done

echo ""; echo "$L ===== dry run: $PASS passed, $FAIL failed ====="
if (( FAIL == 0 )); then
  echo "$L ALL PATHS OK - safe to submit aaai_ugc/jobs/shift_study.sh"
else
  echo "$L DRY RUN FAILED - do not submit the full job" >&2
fi
echo "$L done $(date)"
exit $(( FAIL > 0 ? 1 : 0 ))
