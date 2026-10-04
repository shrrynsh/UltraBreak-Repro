"""
embed.py — sentence embeddings for E0b, without adding a package to `repro`.

`sentence-transformers` is not installed and must not be: the project rule is
that `repro`'s package versions never change, because that env produced every
existing number in the study. For `all-MiniLM-L6-v2` the library's forward pass
is exactly (a) transformer encode, (b) mean-pool over the attention mask,
(c) L2-normalise — its `modules.json` is Transformer -> Pooling(mean) ->
Normalize, and `sentence_bert_config.json` sets max_seq_length=256. Reproducing
those three steps on `transformers.AutoModel` gives the same vectors, so E0b's
similarities are unaffected by the substitution.

CPU by default: 885 short strings is a few seconds, and keeping it off the GPU
means E0 can run while the cluster's single job slot is busy.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LEN = 256  # from the model's sentence_bert_config.json


def _mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """Mask-aware mean over tokens — the library's Pooling(mean) module."""
    mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(texts, batch_size: int = 64, device: str = "cpu",
                model_id: str = MODEL_ID, verbose: bool = True) -> np.ndarray:
    """
    Return an (n, 384) float32 array of L2-normalised embeddings.

    Because rows are unit-norm, cosine similarity is a plain dot product, which
    is how `max_similarity_to_train` computes it.
    """
    texts = [("" if t is None else str(t)) for t in texts]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            vec = _mean_pool(hidden, enc["attention_mask"])
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            out.append(vec.cpu().numpy().astype(np.float32))
            if verbose:
                print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(out)


def max_similarity_to_train(eval_emb: np.ndarray, train_emb: np.ndarray):
    """
    For each evaluation query, the maximum cosine similarity to ANY training
    query, plus the index of that nearest training row.

    Max (not mean) is the right statistic for the question being asked: a query
    is "near the training distribution" if it resembles *some* trained-on query,
    not if it resembles the corpus average.
    """
    sims = eval_emb @ train_emb.T          # unit-norm rows -> dot == cosine
    return sims.max(axis=1), sims.argmax(axis=1)


def _selftest() -> None:
    # pooling correctness on a hand-built case, mask must exclude padding
    h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    m = torch.tensor([[1, 1, 0]])
    got = _mean_pool(h, m).numpy()
    assert np.allclose(got, [[2.0, 2.0]]), got

    # max-similarity: identical text must score ~1.0 against itself
    e = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = np.array([[1.0, 0.0]], dtype=np.float32)
    s, idx = max_similarity_to_train(e, t)
    assert np.allclose(s, [1.0, 0.0]), s
    assert list(idx) == [0, 0]
    print("embed.py self-test OK (pooling + similarity)")


if __name__ == "__main__":
    _selftest()
