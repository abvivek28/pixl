#!/usr/bin/env python3
"""Synthetic alignment benchmark with known zoom/translation, lighting and noise."""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw
from app.engine.alignment import align_generated_image, _transform


def textured_scene(seed: int, size=(300, 260)) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = size
    noise = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(noise, (0, 0), 2.0)
    image = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(image)
    for i in range(12):
        x1 = int(rng.integers(0, w - 45)); y1 = int(rng.integers(0, h - 45))
        x2 = min(w - 1, x1 + int(rng.integers(15, 55)))
        y2 = min(h - 1, y1 + int(rng.integers(15, 55)))
        colour = tuple(int(v) for v in rng.integers(30, 240, 3))
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=2)
    draw.line((5, h - 10, w - 10, 8), fill=(245, 245, 245), width=3)
    draw.ellipse((w // 3, h // 4, 2 * w // 3, 3 * h // 4), outline=(255, 220, 30), width=4)
    return image


def load_real_crops(input_dir: str | None, cases: int, seed: int):
    if not input_dir:
        return None
    folder = Path(input_dir)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"})
    if not paths:
        raise SystemExit(f"No supported image crops found in {folder}")
    rng = random.Random(seed)
    rng.shuffle(paths)
    images = []
    for path in paths[:cases]:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if min(image.size) < 120:
                continue
            image.thumbnail((600, 600), Image.Resampling.LANCZOS)
            images.append(image.copy())
    if not images:
        raise SystemExit("No usable real crops (minimum dimension is 120px).")
    return images


def run(cases: int, seed: int, input_dir: str | None = None):
    rng = random.Random(seed)
    rows = []
    real_crops = load_real_crops(input_dir, cases, seed)
    for i in range(cases):
        original = real_crops[i % len(real_crops)].copy() if real_crops else textured_scene(seed + i)
        h, w = np.asarray(original).shape[:2]
        # Keep the selected region central; the rest is registration context.
        selection = (w // 3, h // 3, 2 * w // 3, 2 * h // 3)
        generated_scale = rng.uniform(0.94, 1.06)
        generated_tx = rng.uniform(-10, 10)
        generated_ty = rng.uniform(-10, 10)
        generated = _transform(original, generated_scale, generated_tx, generated_ty)
        arr = np.asarray(generated).astype(np.float32)
        # Mild global illumination change and sensor-like noise.
        arr = arr * rng.uniform(0.94, 1.06) + rng.uniform(-5, 5)
        arr += np.random.default_rng(seed + 1000 + i).normal(0, 2.0, arr.shape)
        generated = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        result = align_generated_image(
            original, generated, selection_box=selection,
            context_margin_px=40, safety_margin_px=10, max_translation_px=30,
            min_scale=0.80, max_scale=1.20,
        )
        expected_scale = 1.0 / generated_scale
        expected_tx = -generated_tx / generated_scale
        expected_ty = -generated_ty / generated_scale
        rows.append({
            "case": i,
            "expected": {"scale": expected_scale, "tx": expected_tx, "ty": expected_ty},
            "fit_best": {"scale": result.fit_best_scale, "tx": result.fit_best_translation_x,
                          "ty": result.fit_best_translation_y},
            "validation_selected": {"scale": result.proposed_scale, "tx": result.proposed_translation_x,
                                    "ty": result.proposed_translation_y},
            "applied": result.applied,
            "fit_best_translation_error_px": math.hypot(result.fit_best_translation_x - expected_tx,
                                                          result.fit_best_translation_y - expected_ty),
            "fit_best_scale_error_percent": abs(result.fit_best_scale - expected_scale) * 100,
            "validation_selected_translation_error_px": math.hypot(result.proposed_translation_x - expected_tx,
                                                                    result.proposed_translation_y - expected_ty),
            "validation_selected_scale_error_percent": abs(result.proposed_scale - expected_scale) * 100,
            "effective_output_translation_error_px": math.hypot((result.translation_x if result.applied else 0.0) - expected_tx,
                                                                 (result.translation_y if result.applied else 0.0) - expected_ty),
            "effective_output_scale_error_percent": abs((result.scale if result.applied else 1.0) - expected_scale) * 100,
            "validation_improvement_percent": result.validation_improvement_percent,
        })
    translation_errors = [r["fit_best_translation_error_px"] for r in rows]
    scale_errors = [r["fit_best_scale_error_percent"] for r in rows]
    accepted = [r for r in rows if r["applied"]]
    accepted_translation = [r["validation_selected_translation_error_px"] for r in accepted]
    accepted_scale = [r["validation_selected_scale_error_percent"] for r in accepted]
    effective_translation = [r["effective_output_translation_error_px"] for r in rows]
    effective_scale = [r["effective_output_scale_error_percent"] for r in rows]
    summary = {
        "cases": cases,
        "source": "real_crops" if input_dir else "synthetic_textured_scenes",
        "applied_count": sum(r["applied"] for r in rows),
        "translation_error_mean_px": float(np.mean(translation_errors)),
        "translation_error_p95_px": float(np.percentile(translation_errors, 95)),
        "scale_error_mean_percentage_points": float(np.mean(scale_errors)),
        "scale_error_p95_percentage_points": float(np.percentile(scale_errors, 95)),
        "accepted_translation_error_mean_px": float(np.mean(accepted_translation)) if accepted else None,
        "accepted_scale_error_mean_percentage_points": float(np.mean(accepted_scale)) if accepted else None,
        "effective_output_translation_error_mean_px": float(np.mean(effective_translation)),
        "effective_output_scale_error_mean_percentage_points": float(np.mean(effective_scale)),
        "results": rows,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="alignment_benchmark.json")
    parser.add_argument("--input-dir", default=None,
                        help="Optional folder of real crop images. The central third is treated as the synthetic selection.")
    args = parser.parse_args()
    summary = run(args.cases, args.seed, args.input_dir)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Detailed results: {args.output}")


if __name__ == "__main__":
    main()
