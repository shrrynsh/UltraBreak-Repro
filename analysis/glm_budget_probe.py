#!/usr/bin/env python
"""Measure GLM-4.1V-Thinking's NATURAL generation length per task type, so the
token budget can be set to cover every task instead of guessed.

Generates with a high ceiling and records how many new tokens GLM actually uses
before it stops on its own, plus whether it closed </think> and emitted <answer>.
"""
import sys, os, json, argparse
import pandas as pd, torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import base64

ap = argparse.ArgumentParser()
ap.add_argument("--ceiling", type=int, default=6144, help="generation ceiling for the probe")
ap.add_argument("--per_cfg", type=int, default=4, help="rows sampled per task type")
ap.add_argument("--out", default="analysis/out/glm_budget_probe.json")
args = ap.parse_args()

# one config per task TYPE, hardest first (figstep asks for 3x100 words AFTER reasoning)
TASKS = {
    "figstep":  "attack_configs/figstep_safebench.csv",
    "mmsafety": "attack_configs/t1mm_cf_mmsafety.csv",
    "safebench":"attack_configs/xfer_projoff_safebench.csv",
    "advbench": "attack_configs/xfer_projoff_advbench.csv",
}
MODEL = "THUDM/GLM-4.1V-9B-Thinking"
proc = AutoProcessor.from_pretrained(MODEL)
model = Glm4vForConditionalGeneration.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")

rows = []
for task, cfg in TASKS.items():
    if not os.path.exists(cfg):
        print(f"[probe] skip {task}: no {cfg}", flush=True); continue
    df = pd.read_csv(cfg).head(args.per_cfg)
    for _, r in df.iterrows():
        img_path, prompt = r["image"], r["text"]
        content = []
        if isinstance(img_path, str) and img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "image", "image": f"data:image/png;base64,{b64}"})
        content.append({"type": "text", "text": prompt})
        inputs = proc.apply_chat_template([{"role": "user", "content": content}],
                    tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt").to(model.device)
        inputs.pop("token_type_ids", None)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.ceiling)
        new = out[0][n_in:]
        text = proc.decode(new, skip_special_tokens=True)
        used = int((new != proc.tokenizer.pad_token_id).sum()) if proc.tokenizer.pad_token_id is not None else len(new)
        rec = dict(task=task, new_tokens=int(len(new)), used_tokens=used,
                   hit_ceiling=bool(len(new) >= args.ceiling),
                   closed_think="</think>" in text, has_answer="<answer>" in text,
                   chars=len(text))
        rows.append(rec); print("[probe]", json.dumps(rec), flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(rows, open(args.out, "w"), indent=2)
d = pd.DataFrame(rows)
print("\n===== GLM natural generation length by task =====")
if not d.empty:
    print(d.groupby("task").agg(n=("new_tokens","size"), max_tokens=("new_tokens","max"),
          mean_tokens=("new_tokens","mean"), hit_ceiling=("hit_ceiling","sum"),
          closed_think=("closed_think","sum"), has_answer=("has_answer","sum")).to_string())
    print(f"\nOVERALL MAX new_tokens = {d.new_tokens.max()}  | any hit ceiling({args.ceiling}): {int(d.hit_ceiling.sum())}")
