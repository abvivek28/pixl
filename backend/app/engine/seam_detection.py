from __future__ import annotations

"""Polygon-aware seam diagnostics and conservative mask feathering."""

import cv2
import numpy as np
from PIL import Image, ImageFilter


def polygon_mask(size: tuple[int, int], points: list[list[int]] | None,
                 box: tuple[int, int, int, int]) -> np.ndarray:
    w, h = size
    mask = np.zeros((h, w), np.uint8)
    if points and len(points) >= 3:
        cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
    else:
        x1, y1, x2, y2 = box
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
    return mask.astype(bool)


def _boundary_pairs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    inner = (m > 0) & (cv2.erode(m, kernel, iterations=1) == 0)
    outer = (cv2.dilate(m, kernel, iterations=1) > 0) & (m == 0)
    iy, ix = np.where(inner)
    oy, ox = np.where(outer)
    if len(ix) == 0 or len(ox) == 0:
        return np.empty((0, 2), int), np.empty((0, 2), int)
    pairs_in, pairs_out = [], []
    # Every inner boundary pixel has at least one 8-connected outer neighbour.
    for y, x in zip(iy.tolist(), ix.tolist()):
        y0, y1 = max(0, y - 1), min(mask.shape[0], y + 2)
        x0, x1 = max(0, x - 1), min(mask.shape[1], x + 2)
        candidates = np.argwhere(~mask[y0:y1, x0:x1])
        if not len(candidates):
            continue
        candidates[:, 0] += y0
        candidates[:, 1] += x0
        j = np.argmin((candidates[:, 0] - y) ** 2 + (candidates[:, 1] - x) ** 2)
        pairs_in.append((x, y))
        pairs_out.append((int(candidates[j, 1]), int(candidates[j, 0])))
    return np.asarray(pairs_in, int), np.asarray(pairs_out, int)


def evaluate_seam(original: np.ndarray, composite: np.ndarray,
                  mask: np.ndarray) -> dict:
    """Measure actual freehand boundary continuity against original baseline."""
    inside, outside = _boundary_pairs(mask)
    if len(inside) < 4:
        return {"boundary_pixels": int(len(inside)), "mean_boundary_step": 0.0,
                "p95_boundary_step": 0.0, "baseline_step": 0.0,
                "excess_boundary_step": 0.0, "halo_score": 0.0,
                "seam_confidence": "low"}
    ix, iy = inside[:, 0], inside[:, 1]
    ox, oy = outside[:, 0], outside[:, 1]
    comp_step = np.linalg.norm(composite[iy, ix].astype(np.float32) -
                               composite[oy, ox].astype(np.float32), axis=1)
    base_step = np.linalg.norm(original[iy, ix].astype(np.float32) -
                               original[oy, ox].astype(np.float32), axis=1)
    # Positive excess captures a new discontinuity, while avoiding a penalty for
    # pre-existing strong edges that were already present in the source image.
    excess = np.maximum(0.0, comp_step - base_step)
    # Halo proxy: unusually bright/dark boundary band relative to adjacent source.
    comp_luma = (0.2126 * composite[..., 0] + 0.7152 * composite[..., 1] + 0.0722 * composite[..., 2])
    base_luma = (0.2126 * original[..., 0] + 0.7152 * original[..., 1] + 0.0722 * original[..., 2])
    luma_delta = np.abs(comp_luma[iy, ix] - base_luma[iy, ix])
    halo = float(np.percentile(luma_delta, 95)) if len(luma_delta) else 0.0
    mean_excess = float(np.mean(excess))
    p95_excess = float(np.percentile(excess, 95))
    return {"boundary_pixels": int(len(inside)),
            "mean_boundary_step": float(np.mean(comp_step)),
            "p95_boundary_step": float(np.percentile(comp_step, 95)),
            "baseline_step": float(np.mean(base_step)),
            "excess_boundary_step": mean_excess,
            "p95_excess_boundary_step": p95_excess,
            "halo_score": halo,
            "seam_confidence": "high" if len(inside) >= 100 else "medium"}


def adaptive_feather_alpha(mask: np.ndarray, base_radius: float = 1.5,
                            seam_metrics: dict | None = None) -> np.ndarray:
    """Create an inward-only feather alpha; pixels outside the hard mask stay 0.

    The transition is contained within the user's selected polygon. This avoids
    Gaussian blur spilling generated pixels into unselected background.
    """
    radius = float(base_radius)
    if seam_metrics:
        excess = float(seam_metrics.get("excess_boundary_step", 0.0))
        p95 = float(seam_metrics.get("p95_excess_boundary_step", 0.0))
        if max(excess, p95 * 0.65) > 24:
            radius = min(3.0, max(radius, 2.5))
        elif max(excess, p95 * 0.65) > 12:
            radius = min(2.5, max(radius, 2.0))
    hard = np.asarray(mask, dtype=bool)
    if not hard.any() or radius <= 0:
        return hard.astype(np.float32)
    distance = cv2.distanceTransform(hard.astype(np.uint8), cv2.DIST_L2, 5)
    # Boundary pixels receive a partial blend; the alpha is exactly zero
    # outside the selected polygon and reaches 1 deeper than `radius` pixels.
    alpha = np.clip((distance - 0.5) / max(radius, 1e-6), 0.0, 1.0)
    alpha[~hard] = 0.0
    return alpha.astype(np.float32)
