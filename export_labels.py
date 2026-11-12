"""Export the publishable per-item labels (no model outputs) to results/release/labels.csv."""
import csv
import os

from analyze import load
from common import RESULTS

out_dir = os.path.join(RESULTS, "release")
os.makedirs(out_dir, exist_ok=True)
fields = ["model", "id", "set", "attack", "index", "category", "input_pep_flag", "output_pep_flag",
          "output_pep_checked", "judge_unsafe_raw", "pure_refusal", "jailbreak", "refusal_phrase",
          "gen_ms", "input_pep_ms", "output_pep_ms"]
n = 0
with open(os.path.join(out_dir, "labels.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for model, rows in load().items():
        for r in rows:
            w.writerow({"model": model, "id": r["id"], "set": r["set"], "attack": r["attack"],
                        "index": r["index"], "category": r["category"],
                        "input_pep_flag": int(r["in_flag"]), "output_pep_flag": int(r["out_flag"]),
                        "output_pep_checked": int(r["out_checked"]),
                        "judge_unsafe_raw": "" if r["judge_raw"] is None else int(r["judge_raw"]),
                        "pure_refusal": int(r["pure_refusal"]),
                        "jailbreak": "" if r["unsafe"] is None else int(r["unsafe"]),
                        "refusal_phrase": int(r["refusal"]), "gen_ms": r["gen_ms"],
                        "input_pep_ms": r["in_ms"], "output_pep_ms": r["out_ms"] or ""})
            n += 1
print(f"{n} rows -> {os.path.join(out_dir, 'labels.csv')}")
