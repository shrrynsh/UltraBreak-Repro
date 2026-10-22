#!/usr/bin/env bash
#SBATCH --job-name=ub_cfstrict
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@1200
#
# STRICT CODE-FAITHFUL retrain + paper-protocol evaluation.
#
# WHY. The existing `codefaithful_exact_1300` was trained on
# `train_configs/safebench-tiny_jailbroken_mode_full50.csv`, a corpus WE generated.
# The authors' own `optimise.py` defaults to `./train_configs/test.csv`. Both carry the
# same 50 SafeBench-Tiny queries with byte-identical text/image/target, but the
# `keywords` column differs on 47/50 rows: the authors strip stop words
# ("start operate illegal human organ trade"), ours do not ("start and operate an
# illegal human organ trade"). `keywords` is split on spaces and consumed at
# qwen2_adapter.py:121 to decide which target tokens the semantic-ATTENTION loss
# up-weights, so our file makes function-word tokens eligible for up-weighting and
# redistributes the attention weights on 47 of 50 training rows. Our own ablation
# puts that mechanism at 68.9 ASR points, so this is not cosmetic.
#
# DEFINITION BEING HONOURED (set by the project owner, 2026-10-22):
#   code-faithful = the released code's TRAINING path, at 1300 epochs (not the code's
#                   5000), evaluated as the PAPER prescribes - because the release
#                   ships no evaluation pipeline at table scale (only a 20-row demo
#                   config, attack_configs/test.csv).
#
# WHAT IS THEREFORE STRICT HERE
#   training   : reproducibility/authors_code/optimisation/optimise_cf_strict.py,
#                generated from c4c276d:optimisation/optimise.py with EXACTLY two
#                lines changed (exp_name, num_epochs). Asserted below, not asserted
#                by comment. Raw patch injection (patch_only=False) - the D11 bug is
#                PRESENT, by design: that is what the released code does.
#   corpus     : ./train_configs/test.csv - the authors' own file, byte-identical to
#                c4c276d. Asserted below.
#   adapters   : the whole authors_code/optimisation tree byte-identical to c4c276d.
#                Asserted below (6 files).
#   evaluation : the paper's protocol - SafeBench 315 (--exclude-train), AdvBench 520
#                --normalize (paper section 3.3), HarmBench classifier. Scored under
#                BOTH v1 (the paper's own scorer: "original chat template, without
#                modification") and v3 (our corrected scorer), so the report can lead
#                with v1 and show v3 beside it.
#
# WHAT CANNOT BE STRICT, AND IS NOT PRETENDED TO BE
#   seed     : the authors seed nothing; initialise_patch() calls torch.rand with no
#              generator. We do NOT add a seed, because that would deviate from the
#              code. The consequence is that this is ONE UNSEEDED DRAW and the number
#              is not reproducible by re-running - only by reusing the saved PNG.
#              The patch is the artifact; it is committed with the results.
#   GLM budget: the release hardcodes max_new_tokens=512 for every model, which
#              truncates GLM mid-reasoning on up to 98.7% of rows. Evaluation follows
#              the PAPER here, and the paper states no budget, so GLM runs at 4096.
#
# Cost: ~5.2 GPU-h training (14.32 s/epoch measured in A10) + ~9 GPU-h eval = ~15h,
# one window. Per-stage markers; self-resumes; discards a partial training rather
# than resuming it (--base_epoch restores pixels but not Adam state).
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate || { echo "[cfs] repro venv missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1
L="[cfs]"; AUTH="c4c276d"
EXP="cf_strict_test50_1300"; CKPT=1300
PATCH="outputs/${EXP}/${CKPT}.png"
MARK="results/.cfstrict"; mkdir -p "$MARK" logs
STAGE="${SLURM_TMPDIR:-/tmp}/cfs_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE"
ATTEMPT="${1:-1}"; MAX=6
SB="cfs_sb315"; AB="cfs_abnorm"
MODELS=(Qwen/Qwen2-VL-7B-Instruct Qwen/Qwen-VL-Chat Qwen/Qwen2.5-VL-7B-Instruct
        llava-hf/llava-v1.6-mistral-7b-hf moonshotai/Kimi-VL-A3B-Instruct
        THUDM/GLM-4.1V-9B-Thinking)
TOTAL_CELLS=12   # 6 models x 2 benchmarks

echo "$L start $(date) attempt $ATTEMPT/$MAX"

# ── Stage 0: strictness guards. Fail loud, fail before burning a GPU. ─────────
python - <<'PY' || exit 1
import importlib.metadata as m, torch, subprocess, hashlib, difflib, sys, inspect
assert torch.__version__.startswith("2.5.1"), torch.__version__
assert m.version("transformers").startswith("4.51"), m.version("transformers")
AUTH="c4c276d"
def blob(p):
    return subprocess.run(["git","show",f"{AUTH}:{p}"],capture_output=True,check=True).stdout
# 1. the authors' optimisation tree is pristine
for f in ("__init__.py","base_adapter.py","llava16_adapter.py","optimise.py",
          "qwen2_adapter.py","utils.py"):
    a=blob(f"optimisation/{f}")
    b=open(f"reproducibility/authors_code/optimisation/{f}","rb").read()
    assert a==b, f"authors_code/{f} is NOT byte-identical to {AUTH}"
# 2. the authors' training corpus is pristine
a=blob("train_configs/test.csv"); b=open("train_configs/test.csv","rb").read()
assert a==b, "train_configs/test.csv has been modified since c4c276d"
# 3. our strict trainer differs from the authors' optimise.py in EXACTLY two lines
src=blob("optimisation/optimise.py").decode()
out=open("reproducibility/authors_code/optimisation/optimise_cf_strict.py").read()
ch=[l for l in difflib.unified_diff(src.splitlines(),out.splitlines(),lineterm="",n=0)
    if l.startswith(("+","-")) and not l.startswith(("+++","---"))]
assert len(ch)==4, f"expected 2 changed lines, got {len(ch)//2}:\n"+"\n".join(ch)
assert any("exp_name" in l for l in ch) and any("num_epochs" in l for l in ch), ch
# 4. it still reads the AUTHORS' corpus
assert 'train_config = "./train_configs/test.csv"' in out, "corpus line was altered"
# 5. the D11 bug must be PRESENT on this path (raw patch injection)
sys.path.insert(0,"reproducibility/authors_code/optimisation")
from qwen2_adapter import Qwen2Adapter
s=inspect.getsource(Qwen2Adapter.compute_loss)
assert "if self.patch_only:" in s and "patched_imgs = self.apply_patch(pixel_values.unsqueeze(0), image_grid_thw[0], patch)" in s, \
    "authors' raw-injection else-branch not found"
assert "patch_only=False" in out, "trainer must take the raw-injection branch"
# 6. the corpus actually is the authors' keyword form
import pandas as pd
d=pd.read_csv("train_configs/test.csv")
assert len(d)==50 and "and" not in str(d['keywords'].iloc[1]).split(' '), \
    "test.csv keywords look like OUR stop-word-retaining form"
print("[cfs] STRICTNESS OK: pristine tree + pristine corpus + 2-line diff + D11 bug present")
PY

# ── chain / resume plumbing ───────────────────────────────────────────────────
WALL_LIMIT=86400
RESERVE_TRAIN=23400      # 6.5h for a 1300-epoch training (measured ~5.2h)
wall_left(){ echo $(( WALL_LIMIT - SECONDS )); }
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
  local n; n=$(sbatch --parsable jobs/cf_strict.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n ($d/$TOTAL_CELLS)" || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1 pre-wall handoff"; resub; exit 0' USR1

# ── Stage 1: train ────────────────────────────────────────────────────────────
if [[ ! -f "$PATCH" ]]; then
  if (( SECONDS > WALL_LIMIT - RESERVE_TRAIN )); then
    echo "$L wall: not enough left to train - handing off"; resub; exit 0
  fi
  if compgen -G "outputs/${EXP}/*.png" >/dev/null; then
    echo "$L PARTIAL training found and no ${CKPT}.png - discarding and restarting from"
    echo "$L scratch, because --base_epoch restores patch pixels but NOT Adam state."
    rm -f outputs/${EXP}/*.png outputs/${EXP}/losses.csv
  fi
  echo ""; echo "$L --- train $EXP : authors' code, authors' corpus, 1300 iters ---"
  python reproducibility/authors_code/optimisation/optimise_cf_strict.py \
    || { echo "$L TRAIN FAILED" >&2; exit 1; }
  [[ -f "$PATCH" ]] || { echo "$L training produced no ${CKPT}.png" >&2; exit 1; }
  python - "$PATCH" <<'PY'
import sys, numpy as np
from PIL import Image
a=np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(float)
print(f"[cfs] patch saved: std={a.std():.1f} range=[{a.min():.0f},{a.max():.0f}]")
PY
  echo "$L trained $EXP"
else
  echo "$L training already done ($PATCH)"
fi

# ── Stage 2: evaluation configs, PAPER protocol ───────────────────────────────
[[ -s "attack_configs/${SB}.csv" ]] || python create_attack_configs.py \
  --dataset safebench --config-type attack --exclude-train datasets/SafeBench-Tiny.csv \
  --image "$PATCH" --output "attack_configs/${SB}.csv" || exit 1
[[ -s "attack_configs/${AB}.csv" ]] || python create_attack_configs.py \
  --dataset advbench --config-type attack --normalize \
  --image "$PATCH" --output "attack_configs/${AB}.csv" || exit 1
python - <<PY || exit 1
import pandas as pd
sb=pd.read_csv("attack_configs/${SB}.csv"); ab=pd.read_csv("attack_configs/${AB}.csv")
assert len(sb)==315, f"SafeBench config is {len(sb)} rows, expected 315"
assert len(ab)==520, f"AdvBench config is {len(ab)} rows, expected 520"
assert ab['text'].str.startswith("Steps to").mean() > 0.9, "AdvBench not in the section 3.3 form"
assert set(sb['image'])=={"$PATCH"} and set(ab['image'])=={"$PATCH"}
print("[cfs] eval configs OK: 315 / 520, section-3.3 AdvBench form, strict patch")
PY

# ── Stage 3+4: generate and score under BOTH scorers ──────────────────────────
git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py" || exit 1
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT
GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

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
  [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] && { touch "$MK"; echo "$L cell done $cfg/$m ($(done_n)/$TOTAL_CELLS)"; }
}

for m in "${MODELS[@]}"; do
  echo ""; echo "$L ########## $m ##########"
  score "$SB" "$m"
  score "$AB" "$m"
done

# ── Stage 5: report ───────────────────────────────────────────────────────────
echo ""; echo "$L ===== STRICT code-faithful, paper-protocol evaluation ====="
python - <<PY
import os, pandas as pd
M=[("Qwen2-VL*","Qwen/Qwen2-VL-7B-Instruct"),("Qwen-VL-Chat","Qwen/Qwen-VL-Chat"),
   ("Qwen2.5-VL","Qwen/Qwen2.5-VL-7B-Instruct"),("LLaVA-1.6","llava-hf/llava-v1.6-mistral-7b-hf"),
   ("Kimi-VL","moonshotai/Kimi-VL-A3B-Instruct"),("GLM-4.1V","THUDM/GLM-4.1V-9B-Thinking")]
paper={"SafeBench":[81.59,72.70,60.32,88.25,67.94,66.03],
       "AdvBench":[72.69,71.92,35.77,92.88,30.38,30.00]}
old={"SafeBench":[73.02,60.32,26.98,85.40,65.08,58.73],   # codefaithful_exact_1300, v3
     "AdvBench":[22.50,38.85,2.12,89.62,27.50,21.73]}
def asr(cfg,m,suf):
    p=f"results/{cfg}/{m}_{suf}.csv"
    if not os.path.exists(p): return None
    d=pd.read_csv(p); return 100*d.attack_success.astype(bool).mean()
for bench,cfg in (("SafeBench","${SB}"),("AdvBench","${AB}")):
    print(f"\n  -- {bench} --   (v1 = the paper's own scorer)")
    print(f"  {'':22s}" + "".join(f"{n:>15s}" for n,_ in M))
    for lbl,suf in (("strict CF v1","harmbench"),("strict CF v3","harmbench_v3")):
        row=f"  {lbl:22s}"
        for _,m in M:
            v=asr(cfg,m,suf); row+=f"{(f'{v:.2f}' if v is not None else '.'):>15s}"
        print(row)
    print(f"  {'old CF (our corpus)':22s}" + "".join(f"{v:>15.2f}" for v in old[bench]))
    print(f"  {'paper':22s}" + "".join(f"{v:>15.2f}" for v in paper[bench]))
PY

echo ""; echo "$L markers: $(done_n)/$TOTAL_CELLS"
if (( $(done_n) >= TOTAL_CELLS )); then
  echo "$L COMPLETE - strict code-faithful branch done. Nothing chained on purpose."
else
  resub end
fi
echo "$L finished $(date)"
