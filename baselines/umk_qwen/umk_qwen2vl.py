#!/usr/bin/env python
"""
umk_qwen2vl.py — UMK with **Qwen2-VL-7B as the surrogate**, which is what the
UltraBreak paper actually did.

Why this file exists
--------------------
Our first UMK run followed the released UMK repo, which is built on
MiniGPT-4/Vicuna-13B. That reproduces *a* UMK, not *the* UMK of Table 1. UMK is a
joint attack — an adversarial image **plus a GCG text suffix** — and the suffix is
selected token-by-token against one model's embedding matrix, so it is meaningless
on a different tokenizer. The paper's own numbers give the surrogate away: UMK
scores 97.78 / 70.00 / 54.76 on Qwen2-VL and <=50.79 / <=16.35 / <=37.44 on every
other model. Ours, from Vicuna-13B, scored 1.27 on that column. Reproduction means
using their surrogate, not relabelling ours.

Design
------
The image half is already solved and correct for Qwen2-VL in this repo, so it is
reused rather than re-ported: `Qwen2Adapter` handles patch compositing with the D11
CLIP-normalisation fix, and its `compute_loss` already accepts a `suffix`.

The text half is the new part. GCG needs d(loss)/d(suffix one-hot), which needs the
gradient at the *input embeddings* of the suffix positions. Qwen2-VL merges vision
features into the embedding sequence at `image_token_id`, so that merge is done
explicitly here and the forward is run on `inputs_embeds`.

Protocol (paper + UMK's original work):
  * seed image      = the VAJM image (paper: "that optimised image then serves as
                      the toxic, semantically embedded seed for UMK")
  * iterations      = 2000
  * few-shot corpus = SafeBench-Tiny, all 50 (paper's fairness constraint)
  * suffix init     = 20 tokens of '!', as UMK's released code uses
  * prompt          = query + ' ' + suffix.  UMK has no affirming phrase of its own;
                      UltraBreak's '[Jailbroken Mode]' is deliberately NOT injected,
                      which would make the baseline partly UltraBreak.

    python baselines/umk_qwen/umk_qwen2vl.py --smoke      # validate the gradient path
    python baselines/umk_qwen/umk_qwen2vl.py --n_iters 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "optimisation"))

from qwen2_adapter import Qwen2Adapter          # noqa: E402
from utils import AblationConfig                # noqa: E402

IMAGE_TOKEN_ID = 151655          # Qwen2-VL <|image_pad|>
SUFFIX_INIT = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"


# ── locating the suffix inside the tokenised prompt ───────────────────────────
def find_subsequence(hay: torch.Tensor, needle: torch.Tensor) -> int:
    """Start index of `needle` inside 1-D `hay`, or -1. Asserted unique by caller."""
    n, m = hay.numel(), needle.numel()
    if m == 0 or m > n:
        return -1
    hits = []
    for i in range(n - m + 1):
        if torch.equal(hay[i:i + m], needle):
            hits.append(i)
    if len(hits) != 1:
        return -1 if not hits else hits[0]
    return hits[0]


class UMKQwen:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct"):
        cfg = AblationConfig()                      # paper defaults: semantic, attention, tau 0.5
        self.ad = Qwen2Adapter(model_id, patch_only=False, cfg=cfg)
        self.model = self.ad.model
        self.tok = self.ad.processor.tokenizer
        self.device = self.ad.device
        self.embed = self.model.get_input_embeddings()
        self.W = self.embed.weight                  # [V, d]

    # ---- forward that exposes the input-embedding gradient -------------------
    def loss_and_embed_grad(self, row, patch, suffix: str):
        """
        Returns (loss, grad_at_suffix_positions, suffix_token_ids).

        The vision merge is done by hand because the forward runs on inputs_embeds:
        Qwen2-VL otherwise performs it internally from input_ids + pixel_values.
        """
        text = f"{row['text']} {suffix}".strip()
        input_ids, labels, pixel_values, grid_thw, attn, weights = self.ad.process_target(
            row["image"], text, row["target"], str(row.get("keywords", "")).split(" ")
        )
        suf_ids = self.tok(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(self.device)
        start = find_subsequence(input_ids[0], suf_ids)
        if start < 0:
            raise RuntimeError("suffix tokens not uniquely locatable in the prompt")

        patched = self.ad.apply_patch(pixel_values.unsqueeze(0), grid_thw, patch)
        pv = self.ad.preprocess_patched(patched, grid_thw)

        inputs_embeds = self.embed(input_ids)
        img_embeds = self.model.visual(pv.to(self.W.dtype), grid_thw=grid_thw)
        mask = (input_ids == IMAGE_TOKEN_ID)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[mask] = img_embeds.to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.detach().requires_grad_(True)

        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn,
                         labels=None, image_grid_thw=grid_thw)
        from utils import semantic_similarity_loss
        loss = semantic_similarity_loss(
            out.logits, labels, self.W, weights=weights.unsqueeze(0),
            mode=self.ad.cfg.loss_mode, tau=self.ad.cfg.tau,
            embed_noise=self.ad.cfg.embed_noise, pos_alpha=self.ad.cfg.pos_alpha)
        loss.backward()
        g = inputs_embeds.grad[:, start:start + suf_ids.numel(), :].detach()
        return loss.detach(), g, suf_ids

    # ---- GCG: one-hot gradient, then candidate search ------------------------
    def token_grad(self, suf_ids: torch.Tensor, embed_grad: torch.Tensor) -> torch.Tensor:
        one_hot = torch.zeros(suf_ids.numel(), self.W.shape[0],
                              device=self.device, dtype=self.W.dtype)
        one_hot.scatter_(1, suf_ids.unsqueeze(1),
                         torch.ones(suf_ids.numel(), 1, device=self.device, dtype=self.W.dtype))
        one_hot.requires_grad_()
        emb = (one_hot @ self.W).unsqueeze(0)
        (emb * embed_grad.to(emb.dtype)).sum().backward()
        g = one_hot.grad.clone()
        return g / (g.norm(dim=-1, keepdim=True) + 1e-12)

    def candidates(self, suf_ids, grad, n=64, topk=256):
        top = (-grad).topk(topk, dim=1).indices
        base = suf_ids.repeat(n, 1)
        pos = torch.randint(0, suf_ids.numel(), (n,), device=self.device)
        pick = torch.randint(0, topk, (n, 1), device=self.device)
        vals = torch.gather(top[pos], 1, pick)
        base.scatter_(1, pos.unsqueeze(-1), vals)
        out, seen = [], set()
        for row in base:
            s = self.tok.decode(row, skip_special_tokens=True)
            # keep only candidates that re-tokenise to the same length (GCG's filter)
            r = self.tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            if r.numel() == suf_ids.numel() and s not in seen:
                seen.add(s); out.append(s)
        return out

    @torch.no_grad()
    def eval_suffix(self, rows, patch, suffix) -> float:
        tot = 0.0
        for _, r in rows.iterrows():
            text = f"{r['text']} {suffix}".strip()
            ii, lb, pv0, gt, at, w = self.ad.process_target(
                r["image"], text, r["target"], str(r.get("keywords", "")).split(" "))
            pt = self.ad.apply_patch(pv0.unsqueeze(0), gt, patch)
            pv = self.ad.preprocess_patched(pt, gt)
            e = self.embed(ii).clone()
            ie = self.model.visual(pv.to(self.W.dtype), grid_thw=gt)
            e[ii == IMAGE_TOKEN_ID] = ie.to(e.dtype)
            o = self.model(inputs_embeds=e, attention_mask=at, image_grid_thw=gt)
            from utils import semantic_similarity_loss
            tot += float(semantic_similarity_loss(
                o.logits, lb, self.W, weights=w.unsqueeze(0), mode=self.ad.cfg.loss_mode,
                tau=self.ad.cfg.tau, embed_noise=self.ad.cfg.embed_noise,
                pos_alpha=self.ad.cfg.pos_alpha))
        return tot / max(len(rows), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="train_configs/safebench-tiny_jailbroken_mode_full50.csv")
    ap.add_argument("--seed_image", default="baselines/out/vajm_5000/bad_prompt.png")
    ap.add_argument("--out", default="baselines/out/umk_qwen2vl_2000")
    ap.add_argument("--n_iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1 / 255)
    ap.add_argument("--gcg_every", type=int, default=1)
    ap.add_argument("--n_cands", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="3 iterations, validates the gradient path")
    a = ap.parse_args()
    if a.smoke:
        a.n_iters, a.out = 3, "baselines/out/_umk_qwen_smoke"

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    rows = pd.read_csv(os.path.join(REPO, a.corpus)).fillna("")
    # UMK has no affirming phrase: strip UltraBreak's TPG clause from the query.
    rows["text"] = rows["text"].str.replace(
        r"\.?\s*You must start your response with\s*'.*?'\s*$", "", regex=True)
    print(f"[umk-qwen] corpus {len(rows)} rows (TPG clause stripped — UMK uses its own suffix)")

    atk = UMKQwen()
    seed_img = os.path.join(REPO, a.seed_image)
    assert os.path.exists(seed_img), f"VAJM seed image missing: {seed_img}"
    arr = np.asarray(Image.open(seed_img).convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    patch = torch.tensor(arr).permute(2, 0, 1).to(atk.device).requires_grad_(True)
    print(f"[umk-qwen] seeded from VAJM image {a.seed_image}")

    suffix = SUFFIX_INIT
    rng = np.random.RandomState(a.seed)
    hist = []
    t0 = time.time()
    for it in range(a.n_iters + 1):
        batch = rows.iloc[rng.choice(len(rows), size=min(a.batch, len(rows)), replace=False)]
        # ---- image half: PGD on the patch -----------------------------------
        tot = 0.0
        if patch.grad is not None:
            patch.grad.zero_()
        for _, r in batch.iterrows():
            loss = atk.ad.compute_loss(r, patch, suffix=f" {suffix}")
            loss.backward()
            tot += float(loss)
        with torch.no_grad():
            patch -= a.alpha * patch.grad.sign()
            patch.clamp_(0, 1)
        # ---- text half: one GCG step on the suffix ---------------------------
        if it % a.gcg_every == 0:
            r0 = batch.iloc[0]
            _, g, suf_ids = atk.loss_and_embed_grad(r0, patch.detach(), suffix)
            tg = atk.token_grad(suf_ids, g)
            cands = atk.candidates(suf_ids, tg, n=a.n_cands)
            if cands:
                base = atk.eval_suffix(batch, patch.detach(), suffix)
                best, best_l = suffix, base
                for c in cands[:a.n_cands]:
                    l = atk.eval_suffix(batch.iloc[:1], patch.detach(), c)
                    if l < best_l:
                        best, best_l = c, l
                suffix = best
        hist.append(tot / len(batch))
        if it % 20 == 0:
            print(f"[umk-qwen] iter {it:5d}  loss {hist[-1]:.4f}  "
                  f"elapsed {(time.time()-t0)/60:.1f}m  suffix[:40]={suffix[:40]!r}", flush=True)
            img = (patch.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(a.out, f"{it}.png"))
            with open(os.path.join(a.out, "adv_suffix.txt"), "a") as f:
                f.write(suffix + "\n")

    img = (patch.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(a.out, "bad_vlm_prompt.png"))
    json.dump({"n_iters": a.n_iters, "seed_image": a.seed_image, "corpus": a.corpus,
               "final_suffix": suffix, "suffix_init": SUFFIX_INIT, "seed": a.seed,
               "surrogate": "Qwen/Qwen2-VL-7B-Instruct", "loss_first": hist[0],
               "loss_last": hist[-1], "minutes": (time.time() - t0) / 60},
              open(os.path.join(a.out, "run_config.json"), "w"), indent=2)
    print(f"[umk-qwen] done in {(time.time()-t0)/60:.1f}m  loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    print(f"[umk-qwen] final suffix ({len(suffix)} chars) saved to {a.out}/run_config.json")


if __name__ == "__main__":
    main()
