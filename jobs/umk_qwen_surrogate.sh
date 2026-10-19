#!/usr/bin/env bash
#SBATCH --job-name=ub_umkq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@1200
#
# UMK with Qwen2-VL-7B as the surrogate -- the paper's actual configuration.
#
# Our first UMK followed the released repo (MiniGPT-4/Vicuna-13B) and scored 1.27
# on Qwen2-VL against the paper's 97.78, because a GCG suffix is tuned to one
# tokenizer and means nothing on another. Relabelling that as "cross-architecture
# transfer" measures a different experiment; reproduction needs their surrogate.
#
# Seed = our VAJM image (paper: "that optimised image then serves as the ... seed"),
# 2000 iterations, SafeBench-Tiny all 50, suffix init = 20x '!' as UMK's code uses.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate || exit 1
export PYTHONUNBUFFERED=1
L="[umkq]"; AUTH="c4c276d"; OUT="baselines/out/umk_qwen2vl_2000"
MARK="$OUT/.markers"; mkdir -p "$MARK" logs
STAGE="${SLURM_TMPDIR:-/tmp}/umkq_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE"
ATTEMPT="${1:-1}"; MAX=6
echo "$L start $(date) attempt $ATTEMPT/$MAX"


# ── wall guard ────────────────────────────────────────────────────────────────
# A bash trap does NOT interrupt a running foreground command - it is deferred
# until that command returns. Our inner python runs for hours, so the USR1
# pre-wall trap never executes and the job is SIGKILLed at the wall with no
# handoff (exactly what killed job 9138 at 5/6 trainings). So check elapsed time
# explicitly BETWEEN stages and hand off before starting one we cannot finish.
WALL_LIMIT=86400          # 24h
RESERVE_TRAIN=19800       # 5.5h - one 1300-iteration training measured ~4.4h
RESERVE_EVAL=5400         # 1.5h - a single eval cell
wall_left(){ echo $(( WALL_LIMIT - SECONDS )); }
guard(){   # reserve label -> returns 1 if we should stop now
  local need="$1" what="$2"
  if (( SECONDS > WALL_LIMIT - need )); then
    echo "$L wall: ${SECONDS}s used, $(wall_left)s left - not starting $what"
    return 1
  fi
  return 0
}

RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1
  (( ATTEMPT >= MAX )) && { echo "$L MAX"; return; }
  local n; n=$(sbatch --parsable jobs/umk_qwen_surrogate.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME as job $n" || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1"; resub; exit 0' USR1

[[ -s baselines/out/vajm_5000/bad_prompt.png ]] || { echo "$L VAJM seed image missing" >&2; exit 1; }

# ── Stage 0: smoke — proves the suffix gradient path before 2000 GCG steps ──
if [[ ! -f "$MARK/smoke" ]]; then
  echo "$L ===== stage 0: smoke (3 iters, validates suffix gradients) ====="
  python baselines/umk_qwen/umk_qwen2vl.py --smoke \
    || { echo "$L SMOKE FAILED - gradient path not working" >&2; resub; exit 1; }
  touch "$MARK/smoke"; echo "$L smoke PASSED at ${SECONDS}s"
fi

# ── Stage 1: 2000 joint image+suffix iterations on Qwen2-VL ────────────────
if [[ ! -f "$OUT/bad_vlm_prompt.png" ]]; then
  guard $RESERVE_TRAIN "UMK 2000-iteration training" || { resub; exit 0; }
  echo "$L ===== stage 1: UMK 2000 iters, surrogate = Qwen2-VL ====="
  python baselines/umk_qwen/umk_qwen2vl.py --n_iters 2000 --out "$OUT" --resume \
    || { echo "$L training failed" >&2; resub; exit 1; }
fi
[[ -s "$OUT/bad_vlm_prompt.png" ]] || { echo "$L no artifact" >&2; resub; exit 1; }

# ── Stage 2: evaluate. Qwen2-VL/SafeBench is the decisive cell (paper 97.78) ──
SUF=$(python -c "import json;print(json.load(open('$OUT/run_config.json'))['final_suffix'])")
echo "$L final suffix: ${#SUF} chars"
git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/e1.py"; cp "$STAGE/e1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT

python - "$OUT/bad_vlm_prompt.png" "$SUF" <<'PY'
import sys, os, pandas as pd
sys.path.insert(0,".")
from create_attack_configs import load_safebench, load_advbench
img, suf = sys.argv[1], (sys.argv[2] if len(sys.argv)>2 else "")
for name, df in (("safebench", load_safebench("datasets/safebench.csv", None, "datasets/SafeBench-Tiny.csv", True)),
                 ("advbench",  load_advbench("datasets/adv_bench.csv"))):
    rows=[{"text": f"{str(r['clean_target']).strip()} {suf}".strip(), "image": img,
           "target": str(r['clean_target']).strip().rstrip(".")} for _,r in df.iterrows()]
    p=f"attack_configs/umkq_{name}.csv"; pd.DataFrame(rows).to_csv(p,index=False)
    print(f"[umkq] {p}: {len(rows)} rows")
PY

MODELS=( "Qwen/Qwen2-VL-7B-Instruct" "Qwen/Qwen-VL-Chat" "Qwen/Qwen2.5-VL-7B-Instruct" \
         "llava-hf/llava-v1.6-mistral-7b-hf" "moonshotai/Kimi-VL-A3B-Instruct" )
for m in "${MODELS[@]}"; do
  for b in safebench advbench; do
    cfg="umkq_${b}"; MK="$MARK/eval_$(echo "${cfg}_${m}" | tr '/' '_')"
    [[ -f "$MK" ]] && continue
    R="results/${cfg}/${m}.csv"
    [[ -f "$R" ]] || python evaluation/attack.py --model_name "$m" --attack_config "$cfg" || continue
    [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] || \
      python evaluation/evaluate.py --attack_result "$R" --output_suffix "_harmbench_v3.csv" >/dev/null || true
    [[ -f "results/${cfg}/${m}_harmbench.csv" ]] || \
      ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$R" ) >/dev/null || true
    [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] && touch "$MK"
  done
done

echo ""; echo "$L ===== UMK, Qwen2-VL surrogate vs paper ====="
python - <<'PY'
import os, pandas as pd
M=[("Qwen2-VL*","Qwen/Qwen2-VL-7B-Instruct"),("Qwen-VL-Chat","Qwen/Qwen-VL-Chat"),
   ("Qwen2.5-VL","Qwen/Qwen2.5-VL-7B-Instruct"),("LLaVA-1.6","llava-hf/llava-v1.6-mistral-7b-hf"),
   ("Kimi-VL","moonshotai/Kimi-VL-A3B-Instruct")]
paper={"safebench":[97.78,0.63,15.24,20.63,12.70],"advbench":[70.00,0.38,2.69,16.35,2.12]}
old={"safebench":"a5_umk_safebench","advbench":"a5_umk_advbench"}
def a(c,m):
    p=f"results/{c}/{m}_harmbench_v3.csv"
    return pd.read_csv(p).attack_success.astype(bool).mean()*100 if os.path.exists(p) else None
for b in ("safebench","advbench"):
    print(f"\n  -- {b} --")
    print(f"  {'model':13s} {'paper':>7s} {'ours Qwen2VL-surr':>18s} {'ours MiniGPT4-surr':>19s}")
    for i,(s,m) in enumerate(M):
        n=a(f"umkq_{b}",m); o=a(old[b],m)
        print(f"  {s:13s} {paper[b][i]:>7.2f} {(f'{n:.2f}' if n is not None else '·'):>18s} "
              f"{(f'{o:.2f}' if o is not None else '·'):>19s}")
print("\n  The decisive cell is Qwen2-VL/SafeBench: the paper reports 97.78.")
PY
echo "$L ALL PROTOCOL-FIX JOBS COMPLETE (Tables 3/6 @1300, MM n=1680, UMK/Qwen2-VL)"
echo "$L finished $(date)"
