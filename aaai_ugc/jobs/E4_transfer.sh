#!/usr/bin/env bash
#SBATCH --job-name=ub_e4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@600
#
# E4 — does the format/template dependence survive TRANSFER?
#
# Targets in priority order: Qwen-VL-Chat (mandatory), then LLaVA-1.6 if the
# 4-hour GPU budget allows. 6 new configs x 150 rows = 900 generations/target;
# the other 4 cells of each target's 2x(3+2) design are byte-identical existing
# runs and are reused (verified row-by-row by build_E4_configs.py).
#
# Stage 0 is an in-job 10-row dry run of every new config on the first target;
# the full run is ABORTED if any path fails. (One queue wait instead of two —
# this task is time-boxed.)
#
# Each target uses ITS OWN existing generation settings from attack.py, not
# Qwen2-VL's. Recorded per target in the log.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate || { echo "[e4] repro venv missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1
L="[e4]"; AUTH="c4c276d"
CFG_ROOT="aaai_ugc/configs"; GEN="aaai_ugc/results/E4/gen"
MARK="aaai_ugc/results/E4/.markers"; mkdir -p "$MARK" "$GEN" logs
STAGE="${SLURM_TMPDIR:-/tmp}/e4_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE"
ATTEMPT="${1:-1}"; MAX=4

BUDGET=14400        # 4h hard GPU budget for E4 (task limit)
RESERVE=1500        # don't start a config we cannot finish

TARGETS=( "Qwen/Qwen-VL-Chat" "llava-hf/llava-v1.6-mistral-7b-hf" )
CFGS=( E4a_bare_base E4a_bare_shuffled E4a_steps_base E4a_steps_shuffled
       E4b_tpgon_base E4b_tpgoff_authors )
V1_SET=" E4b_tpgoff_authors "    # the only new `authors` cell; reused cells already have v1

echo "$L start $(date) attempt $ATTEMPT/$MAX  budget=${BUDGET}s"
python - <<'PY'
import importlib.metadata as m, torch
print("[e4] torch", torch.__version__, "| transformers", m.version("transformers"))
assert torch.__version__.startswith("2.5.1") and m.version("transformers").startswith("4.51")
print("[e4] env pins OK")
PY
[[ $? -eq 0 ]] || exit 1

# Record each target's ACTUAL generation settings, which differ from Qwen2-VL's.
python - <<'PY'
import glob, json, os
for pat, name in [("*Qwen-VL-Chat*", "Qwen-VL-Chat"),
                  ("*llava-v1.6-mistral*", "LLaVA-1.6"),
                  ("*Qwen2-VL-7B-Instruct*", "Qwen2-VL-7B (pilot)")]:
    f = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{pat}/snapshots/*/generation_config.json"))
    if not f:
        print(f"[e4] {name}: generation_config.json not found"); continue
    g = json.load(open(f[0]))
    ds = g.get("do_sample", False); tk = g.get("top_k"); tp = g.get("top_p")
    det = "DETERMINISTIC" if (not ds or tk == 1) else "STOCHASTIC (unseeded)"
    print(f"[e4] {name:22s} do_sample={ds} top_k={tk} top_p={tp} "
          f"max_new_tokens={g.get('max_new_tokens','(attack.py:512)')} -> {det}")
PY

for f in images/white.jpeg outputs/ultrabreak.png aaai_ugc/images/shuffled_ultrabreak_seed0.png; do
  [[ -s "$f" ]] || { echo "$L missing $f" >&2; exit 1; }
done

git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py" || exit 1
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT

done_n(){ ls "$MARK" 2>/dev/null | wc -l; }
START_DONE=$(done_n)
RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d; d=$(done_n)
  (( ATTEMPT >= MAX )) && { echo "$L MAX attempts at $d"; return; }
  [[ "$1" == end ]] && (( d <= START_DONE )) && { echo "$L no progress"; return; }
  local n; n=$(sbatch --parsable aaai_ugc/jobs/E4_transfer.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n" || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1 pre-wall handoff"; resub timeout; exit 0' USR1

chain_to(){   # hand the GPU back to the interrupted GLM re-run when E4 is done
  local script="$1" name="$2"
  [[ -f "$script" ]] || return
  if squeue -u "${USER:-$(id -un)}" -h -o "%j" 2>/dev/null | grep -qx "$name"; then
    echo "$L chain: $name already queued - not seeding"; return; fi
  local n; n=$(sbatch --parsable "$script" 2>&1) \
    && echo "$L CHAIN -> $script as job $n" || echo "$L chain failed: $n" >&2; }

tag_of(){ case "$1" in *Qwen-VL-Chat*) echo qvlchat;; *llava*) echo llava;; *) echo other;; esac; }

gen_ok(){  # cfg model -> 0 if generation complete
  python - "$CFG_ROOT/$1.csv" "$GEN/$1/$2.csv" <<'PY'
import sys, os, pandas as pd
c, r = sys.argv[1], sys.argv[2]
if not os.path.exists(r): sys.exit(1)
cf = pd.read_csv(c); rf = pd.read_csv(r)
if "response" not in rf.columns: sys.exit(1)
ne = rf["response"].astype(str).str.strip().replace("nan", "").astype(bool).sum()
sys.exit(0 if (len(rf) >= len(cf) and ne >= len(cf)) else 1)
PY
}

# ══ Stage 0: dry run on the first target ════════════════════════════════════
if [[ ! -f "$MARK/.dryrun_ok" ]]; then
  echo ""; echo "$L ===== stage 0: 10-row dry run ($(tag_of "${TARGETS[0]}")) ====="
  python aaai_ugc/code/build_E4_configs.py --dry-run-rows 10 >/dev/null || {
    echo "$L dry config build failed" >&2; exit 1; }
  DFAIL=0
  for cfg in "${CFGS[@]}"; do
    python evaluation/attack.py --model_name "${TARGETS[0]}" --attack_config "dry_$cfg" \
      --attack_root "$CFG_ROOT" --save_path "aaai_ugc/results/E4/dryrun" \
      >/dev/null 2>"$STAGE/d_$cfg.err" || { echo "$L DRY FAIL gen $cfg"; tail -3 "$STAGE/d_$cfg.err"; DFAIL=1; continue; }
    R="aaai_ugc/results/E4/dryrun/dry_$cfg/${TARGETS[0]}.csv"
    python evaluation/evaluate.py --attack_result "$R" --output_suffix "_harmbench_v3.csv" \
      >/dev/null 2>"$STAGE/d_$cfg.v3.err" || { echo "$L DRY FAIL v3 $cfg"; tail -3 "$STAGE/d_$cfg.v3.err"; DFAIL=1; continue; }
    ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$R" ) >/dev/null 2>&1 \
      || echo "$L DRY WARN v1 $cfg"
    # structural check: extraction must not contain the prompt; counts only
    python - "$cfg" "aaai_ugc/results/E4/dryrun/dry_$cfg/${TARGETS[0]}_harmbench_v3.csv" <<'PY' || DFAIL=1
import sys, pandas as pd
sys.path.insert(0, "aaai_ugc/code"); sys.path.insert(0, ".")
import metrics as M
cfg, p = sys.argv[1], sys.argv[2]
d = pd.read_csv(p)
a = M.annotate(d)
assert len(d) == 10, f"{len(d)} rows"
assert d["attack_success"].notna().all(), "NaN judge decisions"
# Structural extraction check: the HARNESS must not fold the prompt into the
# answer. Test for chat-template scaffolding and for the TPG clause verbatim --
# NOT for the behaviour text, because a model legitimately restates the request
# ("To write a script that ...") and that is content, not a template leak.
MARKERS = ["[/INST]", "<think>", "<|im_start|>", "<|im_end|>", "ASSISTANT:",
           "You must start your response with"]
hits = {}
for raw in d["response"]:
    ans, _ = M.extract(raw)
    for mk in MARKERS:
        if mk.lower() in ans.lower():
            hits[mk] = hits.get(mk, 0) + 1
print(f"[e4] DRY OK {cfg}: v3_asr={100*d.attack_success.astype(bool).mean():.0f} "
      f"medlen={a.answer_len.median():.0f} template_markers={hits or 'none'} "
      f"empty={int((a.answer_len==0).sum())}")
assert not hits, f"template scaffolding survived extraction: {hits}"
PY
  done
  if (( DFAIL != 0 )); then echo "$L DRY RUN FAILED - aborting before full run" >&2; exit 1; fi
  touch "$MARK/.dryrun_ok"; echo "$L dry run PASSED at ${SECONDS}s"
fi

# ══ Full run, targets in priority order, budget-guarded ═════════════════════
for tgt in "${TARGETS[@]}"; do
  T=$(tag_of "$tgt")
  echo ""; echo "$L ########## target $tgt ($T) ##########"
  for cfg in "${CFGS[@]}"; do
    MK="$MARK/${T}_${cfg}"
    [[ -f "$MK" ]] && { echo "$L skip ${T}/${cfg}"; continue; }
    if (( SECONDS > BUDGET - RESERVE )); then
      echo "$L BUDGET: ${SECONDS}s of ${BUDGET}s used - stopping before ${T}/${cfg}"
      echo "$L (priority order means target 1 is complete before target 2 starts)"
      break 2
    fi
    T0=$SECONDS
    echo "$L --- ${T}/${cfg} ---"
    if gen_ok "$cfg" "$tgt"; then echo "$L reuse generations"; else
      python evaluation/attack.py --model_name "$tgt" --attack_config "$cfg" \
        --attack_root "$CFG_ROOT" --save_path "$GEN" \
        || { echo "$L gen FAILED ${T}/${cfg}" >&2; continue; }
    fi
    R="$GEN/$cfg/$tgt.csv"; [[ -f "$R" ]] || continue
    [[ -f "$GEN/$cfg/${tgt}_harmbench_v3.csv" ]] || \
      python evaluation/evaluate.py --attack_result "$R" --output_suffix "_harmbench_v3.csv" \
        >/dev/null || echo "$L v3 FAILED ${T}/${cfg}" >&2
    if [[ "$V1_SET" == *" $cfg "* && ! -f "$GEN/$cfg/${tgt}_harmbench.csv" ]]; then
      ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$R" ) >/dev/null \
        || echo "$L v1 FAILED ${T}/${cfg}" >&2
    fi
    if [[ -f "$GEN/$cfg/${tgt}_harmbench_v3.csv" ]]; then
      touch "$MK"; echo "$L done ${T}/${cfg} in $((SECONDS-T0))s (total ${SECONDS}s)"
    fi
  done
done

echo ""; echo "$L ===== analysis ====="
python aaai_ugc/code/analyse_E4.py || echo "$L analysis failed (results on disk)" >&2

# Hand the GPU back to the GLM re-run that E4 preempted.
if (( $(done_n) >= 7 )); then
  echo "$L E4 complete - returning the GPU to the GLM re-run"
  chain_to jobs/glm_rerun.sh ub_glmre
else
  resub end
fi
echo "$L finished $(date) at ${SECONDS}s, $(done_n) markers"
