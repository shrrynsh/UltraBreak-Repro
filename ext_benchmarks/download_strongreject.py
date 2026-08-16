"""
download_strongreject.py
------------------------
Fetch the StrongREJECT dataset (Souly et al. 2024, NeurIPS) into this repo.

StrongREJECT is a text-instruction jailbreak benchmark: 313 specific,
fact-verifiable forbidden prompts across 6 categories. It fits UltraBreak's
paradigm directly -- each row is a self-contained harmful *text* query, which we
pair with UltraBreak's universal adversarial image, exactly as with AdvBench.

We save the full 313-prompt set. Columns kept: category, source, forbidden_prompt.
`create_attack_configs.load_strongreject()` maps forbidden_prompt -> clean_target.

Pulled as a raw CSV over urllib (no pyarrow / no new deps -- the venv is pinned).

    python ext_benchmarks/download_strongreject.py
"""

import csv
import io
import os
import urllib.request

URL = ("https://raw.githubusercontent.com/alexandrasouly/strongreject/"
       "main/strongreject_dataset/strongreject_dataset.csv")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strongreject.csv")
EXPECTED_TOTAL = 313


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as response:
        text = response.read().decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(text)))

    for r in rows:
        assert r.get("forbidden_prompt", "").strip(), "empty forbidden_prompt"

    print(f"  StrongREJECT rows fetched: {len(rows)}")
    if len(rows) != EXPECTED_TOTAL:
        raise SystemExit(
            f"Expected {EXPECTED_TOTAL} prompts but got {len(rows)}. "
            f"Refusing to write a set that will not compare.")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "source", "forbidden_prompt"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in ("category", "source", "forbidden_prompt")})
    print(f"Saved {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
