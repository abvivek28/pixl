#!/usr/bin/env python3
"""Summarize systematic alignment/pipeline offsets from alignment_history.jsonl."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def median_mad(values):
    if not values:
        return None, None
    med = statistics.median(values)
    return med, statistics.median(abs(v - med) for v in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history", nargs="?", default="data/jobs/alignment_history.jsonl")
    args = parser.parse_args()
    path = Path(args.history)
    if not path.is_file():
        raise SystemExit(f"History file not found: {path}")
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Skipping malformed line {line_no}")
    alignments = [r.get("alignment", {}) for r in records]
    # Fit-best captures the raw estimator's preferred transform even when the
    # independent validation gate later rejects it.
    tx = [float(a.get("fit_best_translation_x", a.get("proposed_translation_x", 0))) for a in alignments]
    ty = [float(a.get("fit_best_translation_y", a.get("proposed_translation_y", 0))) for a in alignments]
    scale_pct = [(float(a.get("fit_best_scale", a.get("proposed_scale", 1))) - 1) * 100 for a in alignments]
    norm_x = [float((r.get("normalization") or {}).get("scale_x_provider_to_crop", 1)) for r in records]
    norm_y = [float((r.get("normalization") or {}).get("scale_y_provider_to_crop", 1)) for r in records]
    applied = sum(bool(a.get("applied")) for a in alignments)
    print(f"Records: {len(records)}")
    print(f"Transforms applied: {applied}/{len(records)}")
    for label, values, unit in [
        ("Fit-best X shift", tx, "px"),
        ("Fit-best Y shift", ty, "px"),
        ("Fit-best zoom correction", scale_pct, "%"),
        ("Provider-to-crop X resize", norm_x, "x"),
        ("Provider-to-crop Y resize", norm_y, "x"),
    ]:
        med, mad = median_mad(values)
        if med is not None:
            same_sign = max(sum(v > 0 for v in values), sum(v < 0 for v in values)) / len(values)
            print(f"{label}: median={med:.4f}{unit}, MAD={mad:.4f}{unit}, dominant-sign={same_sign:.1%}")
    if len(records) >= 20:
        print("Interpretation:")
        if abs(statistics.median(tx)) >= 0.75 or abs(statistics.median(ty)) >= 0.75:
            print("- Persistent translation bias is present; inspect crop/pad/normalization geometry.")
        if abs(statistics.median(scale_pct)) >= 0.5:
            print("- Persistent zoom correction is present; inspect provider output sizing and resampling.")
        if abs(statistics.median(tx)) < 0.75 and abs(statistics.median(ty)) < 0.75 and abs(statistics.median(scale_pct)) < 0.5:
            print("- No strong global transform bias detected; remaining errors are more likely edit-specific.")


if __name__ == "__main__":
    main()
