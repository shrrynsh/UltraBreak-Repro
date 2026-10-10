#!/usr/bin/env bash
#SBATCH --job-name=ub_a5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@1200
#
# A5 — Table 1 baselines VAJM + UMK, implemented per the UltraBreak paper's
# stated protocol (NOT the vendored repos' defaults, which contradict it):
#
#   VAJM : optimise an image on the derogatory corpus for 5,000 iterations
#   UMK  : seed from THAT image, jointly optimise image + GCG text suffix,
#          +2,000 iterations
#   both : SafeBench-Tiny as the few-shot corpus "consistent with UltraBreak"
#          (applied to UMK; VAJM's attacker takes no query set at all -- see
#           baselines/patches/DEVIATIONS.md)
#
# Stages, each with a resume marker so a requeue never repeats finished work:
#   0  smoke  3-iteration run of BOTH attacks, to prove the stack before
#             committing ~10 GPU-hours
#   1  VAJM   5,000 iters                      -> bad_prompt.bmp
#   2  UMK    2,000 iters seeded from stage 1  -> bad_vlm_prompt.bmp + suffix
#   3  eval   both artifacts x 6 models x {SafeBench-315, AdvBench-520,
#             MM-SafetyBench-520}, judged v3 + v1
#
# Stages 0-2 run in the minigpt4 conda env (torch 2.0.1+cu118, transformers
# 4.28.0) -- VAJM/UMK ONLY. Stage 3 runs in the pinned `repro` venv, which
# produced every other number in the study. The two are never mixed.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
L="[a5]"; MP="$HOME/.conda/envs/minigpt4/bin/python"
OUT="$REPO/baselines/out"; MARK="$OUT/.markers"; mkdir -p "$MARK" logs
ATTEMPT="${1:-1}"; MAX=6
VAJM_DIR="$OUT/vajm_5000"; UMK_DIR="$OUT/umk_2000"
VAJM_IMG="$VAJM_DIR/bad_prompt.bmp"; UMK_IMG="$UMK_DIR/bad_vlm_prompt.bmp"
MODELS=( "Qwen/Qwen2-VL-7B-Instruct" "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen-VL-Chat" \
  "llava-hf/llava-v1.6-mistral-7b-hf" "moonshotai/Kimi-VL-A3B-Instruct" "THUDM/GLM-4.1V-9B-Thinking" )
AUTH="c4c276d"
STAGE="${SLURM_TMPDIR:-/tmp}/a5_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE"

echo "$L start $(date) attempt $ATTEMPT/$MAX"
[[ -x "$MP" ]] || { echo "$L minigpt4 env missing at $MP" >&2; exit 1; }
"$MP" - <<'PY'
import torch, transformers
print(f"[a5] minigpt4 env: torch {torch.__version__} | transformers {transformers.__version__}")
assert torch.__version__.startswith("2.0.1"), "minigpt4 torch pin changed"
assert transformers.__version__.startswith("4.28"), "minigpt4 transformers pin changed"
print(f"[a5] CUDA {torch.version.cuda} | device {torch.cuda.get_device_name(0)}")
PY
[[ $? -eq 0 ]] || { echo "$L minigpt4 env assertion failed" >&2; exit 1; }

RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1
  (( ATTEMPT >= MAX )) && { echo "$L MAX attempts"; return; }
  local n; n=$(sbatch --parsable jobs/a5_vajm_umk.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n" || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1 pre-wall handoff"; resub; exit 0' USR1

# ══ Stage 0: smoke test, 3 iterations of each attack ════════════════════════
if [[ ! -f "$MARK/smoke" ]]; then
  echo ""; echo "$L ===== stage 0: smoke (3 iters each) ====="
  rm -rf "$STAGE/smoke_v" "$STAGE/smoke_u"; mkdir -p "$STAGE/smoke_v" "$STAGE/smoke_u"
  ( cd baselines/vajm && VAJM_N_ITERS=3 "$MP" minigpt_visual_attack.py \
      --cfg_path eval_configs/minigpt4_eval.yaml --gpu_id 0 --n_iters 3 \
      --alpha 1 --save_dir "$STAGE/smoke_v" ) || { echo "$L SMOKE FAIL vajm" >&2; resub; exit 1; }
  [[ -s "$STAGE/smoke_v/bad_prompt.bmp" ]] || { echo "$L SMOKE FAIL vajm produced no image" >&2; resub; exit 1; }
  echo "$L smoke: VAJM OK"
  ( cd baselines/umk && UMK_N_ITERS=3 UMK_INIT_IMG="$STAGE/smoke_v/bad_prompt.bmp" \
      UMK_CORPUS_CSV="harmful_corpus/safebench_tiny_fewshot.csv" UMK_N_FEWSHOT=50 \
      "$MP" minigpt_vlm_attack.py --cfg_path eval_configs/minigpt4_eval.yaml \
      --gpu_id 0 --n_iters 3 --alpha 1 --save_dir "$STAGE/smoke_u" ) \
    || { echo "$L SMOKE FAIL umk" >&2; resub; exit 1; }
  [[ -s "$STAGE/smoke_u/bad_vlm_prompt.bmp" ]] || { echo "$L SMOKE FAIL umk produced no image" >&2; resub; exit 1; }
  echo "$L smoke: UMK OK (seeded from VAJM, SafeBench-Tiny corpus)"
  touch "$MARK/smoke"; echo "$L stage 0 PASSED at ${SECONDS}s"
fi

# ══ Stage 1: VAJM, 5,000 iterations on the derogatory corpus ════════════════
if [[ ! -f "$MARK/vajm" ]]; then
  echo ""; echo "$L ===== stage 1: VAJM 5000 iters (derogatory corpus) ====="
  mkdir -p "$VAJM_DIR"
  ( cd baselines/vajm && VAJM_N_ITERS=5000 "$MP" minigpt_visual_attack.py \
      --cfg_path eval_configs/minigpt4_eval.yaml --gpu_id 0 --n_iters 5000 \
      --alpha 1 --save_dir "$VAJM_DIR" ) || { echo "$L VAJM FAILED" >&2; resub; exit 1; }
  [[ -s "$VAJM_IMG" ]] || { echo "$L VAJM produced no final image" >&2; resub; exit 1; }
  touch "$MARK/vajm"; echo "$L stage 1 done at ${SECONDS}s -> $VAJM_IMG"
fi

# ══ Stage 2: UMK, +2,000 iterations seeded from VAJM ════════════════════════
if [[ ! -f "$MARK/umk" ]]; then
  echo ""; echo "$L ===== stage 2: UMK 2000 iters (seed=VAJM, corpus=SafeBench-Tiny) ====="
  [[ -s "$VAJM_IMG" ]] || { echo "$L cannot start UMK: VAJM image missing" >&2; exit 1; }
  mkdir -p "$UMK_DIR"; : > "$UMK_DIR/adv_suffix.txt"
  ( cd "$UMK_DIR" && UMK_N_ITERS=2000 UMK_INIT_IMG="$VAJM_IMG" \
      UMK_CORPUS_CSV="$REPO/baselines/umk/harmful_corpus/safebench_tiny_fewshot.csv" \
      UMK_N_FEWSHOT=50 PYTHONPATH="$REPO/baselines/umk" \
      "$MP" "$REPO/baselines/umk/minigpt_vlm_attack.py" \
      --cfg_path "$REPO/baselines/umk/eval_configs/minigpt4_eval.yaml" \
      --gpu_id 0 --n_iters 2000 --alpha 1 --save_dir "$UMK_DIR" ) \
    || { echo "$L UMK FAILED" >&2; resub; exit 1; }
  [[ -s "$UMK_IMG" ]] || { echo "$L UMK produced no final image" >&2; resub; exit 1; }
  touch "$MARK/umk"; echo "$L stage 2 done at ${SECONDS}s -> $UMK_IMG"
fi

# ══ Stage 3: evaluate both baselines, 6 models x 3 benchmarks ══════════════
echo ""; echo "$L ===== stage 3: evaluation (repro venv) ====="
source repro/bin/activate || { echo "$L repro venv missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1
python - <<'PY'
import importlib.metadata as m, torch
assert torch.__version__.startswith("2.5.1") and m.version("transformers").startswith("4.51")
print("[a5] repro env pins OK for evaluation")
PY
[[ $? -eq 0 ]] || exit 1

# The attacks save .bmp; attack.py base64-encodes bytes under a data:image/png
# header, so convert losslessly to PNG rather than mislabel the mime type.
python - "$VAJM_IMG" "$UMK_IMG" <<'PY'
import sys, os
from PIL import Image
for p in sys.argv[1:]:
    if not os.path.exists(p): continue
    out = os.path.splitext(p)[0] + ".png"
    if not os.path.exists(out):
        Image.open(p).convert("RGB").save(out)
    im = Image.open(out)
    print(f"[a5] {out}  {im.mode} {im.size}")
PY
VAJM_PNG="${VAJM_IMG%.bmp}.png"; UMK_PNG="${UMK_IMG%.bmp}.png"

# UMK's attack appends its best suffix per logging step; the last line is final.
UMK_SUFFIX=""
if [[ -s "$UMK_DIR/adv_suffix.txt" ]]; then
  UMK_SUFFIX=$(grep -v '^[[:space:]]*$' "$UMK_DIR/adv_suffix.txt" | tail -1)
  echo "$L UMK final suffix captured (${#UMK_SUFFIX} chars)"
else
  echo "$L WARNING: no adv_suffix.txt - UMK will be evaluated image-only" >&2
fi

git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py" || exit 1
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT
GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

# Build configs. Each baseline uses ITS OWN method's prompt, not UltraBreak's TPG
# phrase: VAJM is image-only, UMK is image + its GCG suffix.
python - "$VAJM_PNG" "$UMK_PNG" "$UMK_SUFFIX" <<'PY'
import os, sys, pandas as pd
sys.path.insert(0, ".")
from create_attack_configs import load_advbench, load_mm_safetybench, load_safebench, stratified_subsample
vajm_img, umk_img, umk_suffix = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
rel = lambda p: os.path.relpath(p, ".")
sets = {
 "safebench": load_safebench("datasets/safebench.csv", None, "datasets/SafeBench-Tiny.csv", True),
 "advbench":  load_advbench("datasets/adv_bench.csv"),
 "mmsafety":  stratified_subsample(load_mm_safetybench("datasets/mm_safetybench.csv"), 520, 0),
}
for name, df in sets.items():
    for tag, img, suf in (("vajm", vajm_img, ""), ("umk", umk_img, umk_suffix)):
        rows = []
        for _, r in df.iterrows():
            q = str(r["clean_target"]).strip()
            text = f"{q} {suf}".strip() if suf else q
            rows.append({"text": text, "image": rel(img), "target": q.rstrip(".")})
        out = f"attack_configs/a5_{tag}_{name}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[a5] {out}: {len(rows)} rows, suffix={'yes' if suf else 'no'}")
PY

for tag in vajm umk; do
  for bench in safebench advbench mmsafety; do
    CFG="a5_${tag}_${bench}"
    [[ -s "attack_configs/${CFG}.csv" ]] || { echo "$L missing config $CFG" >&2; continue; }
    for m in "${MODELS[@]}"; do
      MK="$MARK/eval_${CFG}_$(echo "$m" | tr '/' '_')"
      [[ -f "$MK" ]] && continue
      RES="results/${CFG}/${m}.csv"
      if [[ ! -f "$RES" ]]; then
        gpy="$(py_for_gen "$m")"; echo "$L gen $CFG on $m"
        "$gpy" evaluation/attack.py --model_name "$m" --attack_config "$CFG" \
          || { echo "$L gen failed $CFG/$m" >&2; continue; }
      fi
      [[ -f "$RES" ]] || continue
      [[ -f "results/${CFG}/${m}_harmbench_v3.csv" ]] || \
        python evaluation/evaluate.py --attack_result "$RES" --output_suffix "_harmbench_v3.csv" \
          >/dev/null || echo "$L v3 failed $CFG/$m" >&2
      [[ -f "results/${CFG}/${m}_harmbench.csv" ]] || \
        ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$RES" ) >/dev/null \
          || echo "$L v1 failed $CFG/$m" >&2
      [[ -f "results/${CFG}/${m}_harmbench_v3.csv" ]] && touch "$MK"
    done
  done
done

echo ""; echo "$L ===== A5 baselines vs paper Table 1 ====="
python - <<'PY'
import os, pandas as pd
order = ["Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen2.5-VL-7B-Instruct","Qwen/Qwen-VL-Chat",
         "llava-hf/llava-v1.6-mistral-7b-hf","moonshotai/Kimi-VL-A3B-Instruct",
         "THUDM/GLM-4.1V-9B-Thinking"]
short = {order[0]:"Qwen2-VL",order[1]:"Qwen2.5-VL",order[2]:"Qwen-VL-Chat",
         order[3]:"LLaVA-1.6",order[4]:"Kimi-VL",order[5]:"GLM-4.1V"}
def asr(cfg, m, suf):
    p = f"results/{cfg}/{m}{suf}"
    if not os.path.exists(p): return None
    d = pd.read_csv(p)
    return d["attack_success"].astype(bool).mean()*100 if "attack_success" in d else None
for bench in ("safebench","advbench","mmsafety"):
    print(f"\n-- {bench} (ASR%, v1 / v3) --")
    print(f"  {'model':13s} {'VAJM v1':>8s} {'VAJM v3':>8s} {'UMK v1':>8s} {'UMK v3':>8s}")
    for m in order:
        c = []
        for tag in ("vajm","umk"):
            for suf in ("_harmbench.csv","_harmbench_v3.csv"):
                v = asr(f"a5_{tag}_{bench}", m, suf)
                c.append(f"{v:.1f}" if v is not None else "-")
        print(f"  {short[m]:13s} {c[0]:>8s} {c[1]:>8s} {c[2]:>8s} {c[3]:>8s}")
PY

DONE=$(ls "$MARK" | grep -c '^eval_' || true)
echo ""; echo "$L eval cells complete: $DONE/36"
if (( DONE < 36 )); then resub; fi
echo "$L finished $(date) at ${SECONDS}s"
