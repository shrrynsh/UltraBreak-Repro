"""
download_harmbench.py
---------------------
Fetch the HarmBench behaviours (Mazeika et al. 2024, ICML) into this repo.

HarmBench ships 400 text behaviours across three functional categories:
standard (200), copyright (100), contextual (100). We keep only the **200
standard** behaviours -- they are self-contained harmful instructions, the clean
drop-in for UltraBreak's single-text-query + universal-image paradigm.

  - contextual behaviours need a per-item ContextString prepended (they are not
    self-contained), so they do not fit a one-image-fits-all attack;
  - copyright behaviours are a different verbatim-reproduction task.

Note the judge alignment: this repo already scores with the HarmBench classifier
(v3), so evaluating on HarmBench's own behaviours pairs the benchmark with the
scorer it was designed for.

Columns kept: Behavior, SemanticCategory, BehaviorID. `load_harmbench()` maps
Behavior -> clean_target, SemanticCategory -> category_name.

    python ext_benchmarks/download_harmbench.py
"""

import csv
import io
import os
import urllib.request

URL = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
       "main/data/behavior_datasets/harmbench_behaviors_text_all.csv")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harmbench_standard.csv")
EXPECTED_STANDARD = 200


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as response:
        text = response.read().decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(text)))

    standard = [r for r in rows if r.get("FunctionalCategory") == "standard"]
    for r in standard:
        assert r.get("Behavior", "").strip(), "empty Behavior"

    print(f"  HarmBench total fetched: {len(rows)}  |  standard kept: {len(standard)}")
    if len(standard) != EXPECTED_STANDARD:
        raise SystemExit(
            f"Expected {EXPECTED_STANDARD} standard behaviours but got {len(standard)}. "
            f"Refusing to write a set that will not compare.")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Behavior", "SemanticCategory", "BehaviorID"])
        writer.writeheader()
        for r in standard:
            writer.writerow({k: r.get(k, "") for k in ("Behavior", "SemanticCategory", "BehaviorID")})
    print(f"Saved {len(standard)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
