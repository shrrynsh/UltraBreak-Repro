#!/usr/bin/env bash
#SBATCH --job-name=ub_fig2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@900
#
# C2b / Figure 2b — surrogate-model choice. Train UltraBreak's patch on the
# LLaVA-1.6 surrogate instead of Qwen2-VL, then transfer-eval it.
#
# This was blocked until 2026-10-18 by a defect in the AUTHORS' released code:
# `optimisation/llava16_adapter.py::process_target` built every tensor and then
# fell through with no `return`, so it yielded None and `compute_loss` died
# unpacking a 7-tuple. The LLaVA surrogate path crashed on its first call as
# shipped (upstream finding #2) -- which also means the released code cannot have
# produced the paper's Fig 2b. Fixed; arity verified against the caller.
#
# Settings match the paper's Table 4 exactly except the surrogate:
# semantic+attention loss, tau 0.5, TV 0.5, lr 0.01, transforms on, D11 fix on,
# seed 0, **1300 steps -- the paper's stated budget (Table 4)**.
#
# Stage 0 is a 3-step smoke run: the adapter has NEVER successfully executed, so
# the full ~8 GPU-h is not committed until one real forward+backward lands.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate || { echo "[c2b] repro venv missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1
L="[c2b]"; EXP="c2b_llava_surrogate_1300"; CORPUS="train_configs/safebench-tiny_jailbroken_mode_full50.csv"
PATCH="outputs/${EXP}/1300.png"; MARK="outputs/${EXP}/.markers"; mkdir -p "$MARK" logs
ATTEMPT="${1:-1}"; MAX=4
AUTH="c4c276d"; STAGE="${SLURM_TMPDIR:-/tmp}/c2b_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE"

echo "$L start $(date) attempt $ATTEMPT/$MAX"
python - <<'PY'
import importlib.metadata as m, torch
print("[c2b] torch", torch.__version__, "| transformers", m.version("transformers"))
assert torch.__version__.startswith("2.5.1") and m.version("transformers").startswith("4.51")
print("[c2b] env pins OK")
PY
[[ $? -eq 0 ]] || exit 1

# The fix this job depends on must be present, and the D11 fix must be too.
python - <<'PY'
import ast
src = open("optimisation/llava16_adapter.py").read()
t = ast.parse(src)
fn = next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == "process_target")
rets = [x for x in ast.walk(fn) if isinstance(x, ast.Return)]
assert rets, "llava16_adapter.process_target still has no return - C2b would crash"
assert len(rets[0].value.elts) == 7, f"return arity {len(rets[0].value.elts)}, caller expects 7"
print("[c2b] llava16_adapter.process_target returns a 7-tuple - OK")
import inspect, sys
sys.path.insert(0, "optimisation")
from qwen2_adapter import Qwen2Adapter
assert "normalised_patch = (patch - mean_tensor) / std_tensor" in inspect.getsource(Qwen2Adapter.compute_loss)
print("[c2b] D11 normalisation fix present")
PY
[[ $? -eq 0 ]] || { echo "$L precondition failed" >&2; exit 1; }

RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1
  (( ATTEMPT >= MAX )) && { echo "$L MAX attempts"; return; }
  local n; n=$(sbatch --parsable jobs/fig2_paperfaithful.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n" || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1 pre-wall handoff"; resub; exit 0' USR1

# ── Stage 0: 3-step smoke, because this adapter has never run ───────────────
if [[ ! -f "$MARK/smoke" ]]; then
  echo ""; echo "$L ===== stage 0: 3-step smoke on the LLaVA surrogate ====="
  rm -rf "outputs/_c2b_smoke"
  python optimisation/optimise_proj.py --train_config "$CORPUS" --exp_name _c2b_smoke \
    --surrogates llava --no_projection --seed 0 --num_epochs 3 --lr 0.01 \
    || { echo "$L SMOKE FAILED - the adapter still does not run" >&2; resub; exit 1; }
  echo "$L smoke OK - adapter executes end to end"
  touch "$MARK/smoke"
fi

# ── Stage 0b: Fig 2a — the 2B surrogate, also at the paper's 1300 budget ───
# The existing c2a_qwen2b_s3000 patch is 3001 epochs, so Fig 2a carried the same
# off-budget problem as Tables 3/6. The 7B/Qwen arm at 1300 is cmp_pf_* (exists).
P2B="outputs/c2a_qwen2b_1300/1300.png"
if [[ ! -f "$P2B" ]]; then
  echo ""; echo "$L ===== Fig 2a: train the 2B surrogate at 1300 steps ====="
  python optimisation/optimise_proj.py --train_config "$CORPUS" --exp_name c2a_qwen2b_1300 \
    --surrogates qwen2b --no_projection --seed 0 --num_epochs 1301 --lr 0.01 \
    || echo "$L 2B training failed - Fig 2a will stay incomplete" >&2
fi
if [[ -f "$P2B" ]]; then
  C2A="c2a_surr2B_1300_safebench"
  [[ -s "attack_configs/${C2A}.csv" ]] || python create_attack_configs.py --dataset safebench \
    --config-type attack --exclude-train datasets/SafeBench-Tiny.csv --image "$P2B" \
    --output "attack_configs/${C2A}.csv" || true
  for v in "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen2-VL-7B-Instruct"; do
    MK="$MARK/fig2a_$(echo "$v" | tr '/' '_')"; [[ -f "$MK" ]] && continue
    R="results/${C2A}/${v}.csv"
    [[ -f "$R" ]] || python evaluation/attack.py --model_name "$v" --attack_config "$C2A" || continue
    [[ -f "results/${C2A}/${v}_harmbench_v3.csv" ]] || \
      python evaluation/evaluate.py --attack_result "$R" --output_suffix "_harmbench_v3.csv" >/dev/null || true
    [[ -f "results/${C2A}/${v}_harmbench_v3.csv" ]] && touch "$MK"
  done
fi

# ── Stage 1: train 1300 steps (paper budget) on the LLaVA surrogate ────────────────────────
if [[ ! -f "$PATCH" ]]; then
  echo ""; echo "$L ===== stage 1: train $EXP (LLaVA surrogate, 1300 steps = paper budget) ====="
  python optimisation/optimise_proj.py --train_config "$CORPUS" --exp_name "$EXP" \
    --surrogates llava --no_projection --seed 0 --num_epochs 1301 --lr 0.01 \
    || { echo "$L training failed" >&2; resub; exit 1; }
fi
[[ -f "$PATCH" ]] || { echo "$L no patch at $PATCH" >&2; resub; exit 1; }
echo "$L patch ready: $PATCH"

# ── Stage 2: transfer-eval, Fig 2b is surrogate x victim ────────────────────
echo ""; echo "$L ===== stage 2: transfer eval (SafeBench-315) ====="
git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py" || exit 1
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT
GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

CFG="c2b_llava_safebench"
[[ -s "attack_configs/${CFG}.csv" ]] || python create_attack_configs.py --dataset safebench \
  --config-type attack --exclude-train datasets/SafeBench-Tiny.csv --image "$PATCH" \
  --output "attack_configs/${CFG}.csv" || { echo "$L config build failed" >&2; exit 1; }

# LLaVA first: it is the white-box cell for this surrogate, the one Fig 2b hinges on.
VICTIMS=( "llava-hf/llava-v1.6-mistral-7b-hf" "Qwen/Qwen2-VL-7B-Instruct" \
          "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen-VL-Chat" \
          "moonshotai/Kimi-VL-A3B-Instruct" "THUDM/GLM-4.1V-9B-Thinking" )
for v in "${VICTIMS[@]}"; do
  MK="$MARK/eval_$(echo "$v" | tr '/' '_')"; [[ -f "$MK" ]] && { echo "$L skip $v"; continue; }
  RES="results/${CFG}/${v}.csv"
  if [[ ! -f "$RES" ]]; then
    gpy="$(py_for_gen "$v")"; echo "$L gen on $v"
    "$gpy" evaluation/attack.py --model_name "$v" --attack_config "$CFG" \
      || { echo "$L gen failed $v" >&2; continue; }
  fi
  [[ -f "$RES" ]] || continue
  [[ -f "results/${CFG}/${v}_harmbench_v3.csv" ]] || \
    python evaluation/evaluate.py --attack_result "$RES" --output_suffix "_harmbench_v3.csv" >/dev/null \
      || echo "$L v3 failed $v" >&2
  [[ -f "results/${CFG}/${v}_harmbench.csv" ]] || \
    ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$RES" ) >/dev/null \
      || echo "$L v1 failed $v" >&2
  [[ -f "results/${CFG}/${v}_harmbench_v3.csv" ]] && touch "$MK"
done

echo ""; echo "$L ===== Fig 2b: surrogate x victim (SafeBench-315, v3) ====="
python - <<'PY'
import os, pandas as pd
victims=["llava-hf/llava-v1.6-mistral-7b-hf","Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen2.5-VL-7B-Instruct",
         "Qwen/Qwen-VL-Chat","moonshotai/Kimi-VL-A3B-Instruct","THUDM/GLM-4.1V-9B-Thinking"]
short={victims[0]:"LLaVA-1.6",victims[1]:"Qwen2-VL",victims[2]:"Qwen2.5-VL",
       victims[3]:"Qwen-VL-Chat",victims[4]:"Kimi-VL",victims[5]:"GLM-4.1V"}
def a(cfg,v,suf="_harmbench_v3.csv"):
    p=f"results/{cfg}/{v}{suf}"
    return pd.read_csv(p).attack_success.astype(bool).mean()*100 if os.path.exists(p) else None
print(f"  {'victim':14s} {'LLaVA-surrogate':>16s} {'Qwen-surrogate @1300':>22s}")
for v in victims:
    n=a("c2b_llava_safebench",v)
    q=a("cmp_pf_sb315",v)
    print(f"  {short[v]:14s} {(f'{n:.2f}' if n is not None else '-'):>16s} "
          f"{(f'{q:.2f}' if q is not None else '-'):>22s}")
print("\n  Fig 2b's claim: the surrogate choice changes which victims transfer.")
PY

DONE=$(ls "$MARK" | grep -c '^eval_' || true)
echo ""; echo "$L eval cells: $DONE/6"
(( DONE < 6 )) && resub
echo "$L finished $(date)"
