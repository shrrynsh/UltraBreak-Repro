#!/usr/bin/env python
"""C6 / Fig 6 — first-token probability distribution under an adversarial image.

Paper: token-probability distribution "at the first generation position when using
adversarial images with varying temperature settings" (tau=0, 0.5, ->inf). Pass the
image trained under each tau; this extracts and plots the softmax over the vocab at
the first target position (top-k tokens).

    python analysis/first_token_dist.py --image outputs/<run>/<ckpt>.png \
        --label "tau=0.5" --topk 15 --out analysis/out/ftok_<tag>
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "optimisation"))
import numpy as np, torch, pandas as pd
from PIL import Image
import torchvision.transforms as T
from qwen2_adapter import Qwen2Adapter
from utils import AblationConfig, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_patch(path, device):
    img = Image.open(path).convert("RGB")
    return T.Compose([T.Resize((224, 224)), T.ToTensor()])(img).to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--corpus", default="train_configs/safebench-tiny_jailbroken_mode_full50.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    adapter = Qwen2Adapter("Qwen/Qwen2-VL-7B-Instruct", patch_only=False, cfg=AblationConfig())
    patch = load_patch(args.image, adapter.device)
    row = pd.read_csv(args.corpus).fillna("").iloc[0]

    input_ids, labels, pixel_values, image_grid_thw, attention_mask, _ = adapter.process_target(
        row["image"], row["text"], row["target"], row["keywords"].split(" "))
    mean = torch.tensor(OPENAI_CLIP_MEAN).view(-1, 1, 1).to(adapter.device)
    std = torch.tensor(OPENAI_CLIP_STD).view(-1, 1, 1).to(adapter.device)
    norm_patch = (patch - mean) / std if adapter.cfg.normalise_patch else patch
    patched = adapter.apply_patch(pixel_values.unsqueeze(0), image_grid_thw[0], norm_patch)

    with torch.no_grad():
        out = adapter.model(input_ids=input_ids, pixel_values=patched,
                            image_grid_thw=image_grid_thw, attention_mask=attention_mask)
    logits = out.logits[0]                              # [T, V]
    tgt_pos = (labels[0] != -100).nonzero(as_tuple=True)[0]
    p = int(tgt_pos[0].item())                          # first target token index
    dist = torch.softmax(logits[p - 1].float(), dim=-1) # logits at p-1 predict token p
    probs, ids = dist.topk(args.topk)
    tok = adapter.processor.tokenizer
    toks = [tok.decode([int(i)]).strip() or "·" for i in ids]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"\n===== C6 first-token distribution [{args.label}] =====")
    for t, pr in zip(toks, probs.tolist()):
        print(f"  {t:18s} {pr:.4f}")
    print(f"  entropy = {float(-(dist*torch.log(dist+1e-12)).sum()):.3f} nats")
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(toks)), probs.tolist(), color="#4c78a8")
    plt.xticks(range(len(toks)), toks, rotation=60, ha="right", fontsize=8)
    plt.ylabel("probability"); plt.title(f"First-token distribution  {args.label}")
    plt.tight_layout(); plt.savefig(args.out + ".png", dpi=130)
    np.save(args.out + ".npy", np.array(probs.tolist()))
    print(f"[first-token] wrote {args.out}.png / .npy")

if __name__ == "__main__":
    main()
