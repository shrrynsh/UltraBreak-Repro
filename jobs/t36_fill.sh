#!/usr/bin/env bash
#SBATCH --job-name=ub_t36
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@900
#
# Fill the missing cells of paper Tables 3 and 6.
#
# Both tables report across the paper's five-model column set; ours existed only on
# the white-box Qwen2-VL column, because each ablation produced one patch that was
# only ever evaluated on the surrogate. Nothing needs retraining -- every patch is
# already on disk, and the eval configs already point at them. This job is pure
# transfer evaluation.
#
#   Table 3  3 ablation patches x 4 models x {SafeBench, AdvBench}   24 cells
#   Table 6  tv0.2 + tv1.0 x 2 victims x SafeBench                    4 cells
#            (TV=0 is the no-constraints patch, already covered above;
#             TV=0.5 is job 809, already covered by the transfer study)
#                                                            total = 28 cells
#
# MODEL-major with GLM last: GLM at the corrected 4096-token budget runs ~29 s/row
# and is ~25 of the ~29 GPU-h here, so this ordering lands every cheap cell first.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate || { echo "[t36] repro venv missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1
L="[t36]"; AUTH="c4c276d"; MARK="results/.t36_fill"; mkdir -p "$MARK" logs
STAGE="${SLURM_TMPDIR:-/tmp}/t36_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE"
ATTEMPT="${1:-1}"; MAX=6

# cheap models first, GLM last
MODELS=( "Qwen/Qwen-VL-Chat" "Qwen/Qwen2.5-VL-7B-Instruct" "llava-hf/llava-v1.6-mistral-7b-hf" \
         "THUDM/GLM-4.1V-9B-Thinking" )
T3=( full50_abl_noconstraints_safebench_eval full50_abl_noconstraints_advbench_eval
     full50_abl_ce_safebench_eval            full50_abl_ce_advbench_eval
     full50_abl_token_safebench_eval         full50_abl_token_advbench_eval )
T6=( full50_abl_tv0p2_safebench_eval full50_abl_tv1p0_safebench_eval )   # Qwen-VL-Chat + GLM only
T6_MODELS=( "Qwen/Qwen-VL-Chat" "THUDM/GLM-4.1V-9B-Thinking" )

echo "$L start $(date) attempt $ATTEMPT/$MAX"
python - <<'PY'
import importlib.metadata as m, torch
assert torch.__version__.startswith("2.5.1") and m.version("transformers").startswith("4.51")
print("[t36] repro env pins OK")
PY
[[ $? -eq 0 ]] || exit 1

for c in "${T3[@]}" "${T6[@]}"; do
  [[ -s "attack_configs/$c.csv" ]] || { echo "$L missing config $c" >&2; exit 1; }
done
echo "$L all 8 eval configs present (patches already on disk - no training here)"

done_n(){ ls "$MARK" 2>/dev/null | wc -l; }
START=$(done_n); RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d; d=$(done_n)
  (( d >= 28 )) && { echo "$L COMPLETE"; return; }
  (( ATTEMPT >= MAX )) && { echo "$L MAX attempts at $d/28"; return; }
  [[ "${1:-}" == end ]] && (( d <= START )) && { echo "$L no progress"; return; }
  local n; n=$(sbatch --parsable jobs/t36_fill.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n ($d/28)" || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1 pre-wall handoff"; resub; exit 0' USR1

git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py" || exit 1
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT
GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

score_cell(){   # cfg model
  local cfg="$1" m="$2"
  local MK="$MARK/$(echo "${cfg}_${m}" | tr '/' '_')"
  [[ -f "$MK" ]] && return 0
  local RES="results/${cfg}/${m}.csv"
  if [[ ! -f "$RES" ]]; then
    local gpy; gpy="$(py_for_gen "$m")"; echo "$L gen $cfg on $m"
    "$gpy" evaluation/attack.py --model_name "$m" --attack_config "$cfg" \
      || { echo "$L gen failed $cfg/$m" >&2; return 1; }
  fi
  [[ -f "$RES" ]] || return 1
  [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] || \
    python evaluation/evaluate.py --attack_result "$RES" --output_suffix "_harmbench_v3.csv" >/dev/null \
      || echo "$L v3 failed $cfg/$m" >&2
  [[ -f "results/${cfg}/${m}_harmbench.csv" ]] || \
    ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$RES" ) >/dev/null \
      || echo "$L v1 failed $cfg/$m" >&2
  [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] && { touch "$MK"; echo "$L done $cfg/$m ($(done_n)/28)"; }
}

for m in "${MODELS[@]}"; do
  echo ""; echo "$L ########## $m ##########"
  for cfg in "${T3[@]}"; do score_cell "$cfg" "$m"; done
  for cfg in "${T6[@]}"; do
    for tm in "${T6_MODELS[@]}"; do [[ "$tm" == "$m" ]] && score_cell "$cfg" "$m"; done
  done
done

echo ""; echo "$L ===== Table 3 (ours, v3) ====="
python - <<'PY'
import os, pandas as pd
M={"Qwen2-VL":"Qwen/Qwen2-VL-7B-Instruct","Qwen-VL-Chat":"Qwen/Qwen-VL-Chat",
   "Qwen2.5-VL":"Qwen/Qwen2.5-VL-7B-Instruct","LLaVA-1.6":"llava-hf/llava-v1.6-mistral-7b-hf",
   "GLM-4.1V":"THUDM/GLM-4.1V-9B-Thinking"}
rows=[("w/o Jailbreak Image","noattack_safebench_matrix","noattack_advbench_matrix"),
      ("w/o Constraints","full50_abl_noconstraints_safebench_eval","full50_abl_noconstraints_advbench_eval"),
      ("w/o Semantic Loss","full50_abl_ce_safebench_eval","full50_abl_ce_advbench_eval"),
      ("w/o Attention Wgt","full50_abl_token_safebench_eval","full50_abl_token_advbench_eval"),
      ("Proposed (809)","xfer_projoff_safebench","xfer_projoff_advbench")]
def a(cfg,m):
    p=f"results/{cfg}/{m}_harmbench_v3.csv"
    return f"{pd.read_csv(p).attack_success.astype(bool).mean()*100:.2f}" if os.path.exists(p) else "·"
for bench,idx in (("SafeBench",1),("AdvBench",2)):
    print(f"\n  -- {bench} --")
    print(f"  {'variant':20s}" + "".join(f"{k:>14s}" for k in M))
    for r in rows:
        cfg = r[idx]
        # the Proposed/no-attack rows live under the white-box eval name for Qwen2-VL
        line=f"  {r[0]:20s}"
        for k,m in M.items():
            v=a(cfg,m)
            if v=="·" and r[0]=="Proposed (809)" and k=="Qwen2-VL":
                v=a("full50_normfix_lr0p01_s3000_"+("safebench" if idx==1 else "advbench")+"_eval",m)
            line+=f"{v:>14s}"
        print(line)
PY

echo ""; echo "$L ===== Table 6 (ours, SafeBench v3, paper's victims) ====="
python - <<'PY'
import os, pandas as pd
V={"Qwen-VL-Chat":"Qwen/Qwen-VL-Chat","GLM-4.1V":"THUDM/GLM-4.1V-9B-Thinking"}
TV=[("0","full50_abl_noconstraints_safebench_eval"),("0.2","full50_abl_tv0p2_safebench_eval"),
    ("0.5","xfer_projoff_safebench"),("1.0","full50_abl_tv1p0_safebench_eval")]
paper={"0":(61.27,70.48),"0.2":(77.78,66.98),"0.5":(72.70,66.03),"1.0":(65.40,60.95)}
def a(cfg,m):
    p=f"results/{cfg}/{m}_harmbench_v3.csv"
    return pd.read_csv(p).attack_success.astype(bool).mean()*100 if os.path.exists(p) else None
print(f"  {'TV':5s} {'QwenChat ours':>14s} {'paper':>7s} {'GLM ours':>10s} {'paper':>7s}")
for tv,cfg in TV:
    q=a(cfg,V["Qwen-VL-Chat"]); g=a(cfg,V["GLM-4.1V"]); pq,pg=paper[tv]
    print(f"  {tv:5s} {(f'{q:.2f}' if q is not None else '·'):>14s} {pq:>7.2f} "
          f"{(f'{g:.2f}' if g is not None else '·'):>10s} {pg:>7.2f}")
print("\n  The paper's peak is TV=0.2 on Qwen-VL-Chat. This is the comparison that")
print("  settles Table 6 -- our earlier white-box-only verdict was withdrawn.")
PY

echo ""; echo "$L cells: $(done_n)/28"
(( $(done_n) < 28 )) && resub end
echo "$L finished $(date)"
