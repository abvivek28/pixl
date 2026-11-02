from __future__ import annotations

"""Bounded image registration for localized edits.

The generated crop is registered to the original crop using uniform scale and
translation only. The user's selection remains fixed in original coordinates.
OpenCV is used for phase-correlation initialization, masked ECC seeding, fast
warping and gradient registration. Robust fitting uses fit-only context tiles;
a separate held-out metric gates the proposed transform.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class AlignmentResult:
    image: Image.Image
    applied: bool
    scale: float
    translation_x: float
    translation_y: float
    initial_score: float
    final_score: float
    improvement_percent: float
    candidates_evaluated: int
    confidence: str
    reason: str
    phase_correlation_x: float = 0.0
    phase_correlation_y: float = 0.0
    context_ssim: float = 0.0
    context_gradient_similarity: float = 0.0
    context_colour_similarity: float = 0.0
    context_coverage: float = 1.0
    search_config: dict | None = None
    validation_initial_score: float = 0.0
    validation_final_score: float = 0.0
    validation_improvement_percent: float = 0.0
    registration_context_pixels: int = 0
    validation_context_pixels: int = 0
    context_sufficient: bool = False
    proposed_scale: float = 1.0
    proposed_translation_x: float = 0.0
    proposed_translation_y: float = 0.0
    proposed_improvement_percent: float = 0.0
    fit_best_scale: float = 1.0
    fit_best_translation_x: float = 0.0
    fit_best_translation_y: float = 0.0
    fit_best_improvement_percent: float = 0.0


def _selection_mask(size, points, selection_box):
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if points and len(points) >= 3:
        draw.polygon([(float(x), float(y)) for x, y in points], fill=255)
    else:
        x1, y1, x2, y2 = selection_box
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
    return np.asarray(mask, dtype=np.uint8) > 0


def _valid_mask(size, valid_rect):
    w, h = size
    x1, y1, x2, y2 = valid_rect
    result = np.zeros((h, w), np.uint8)
    result[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
    return result.astype(bool)


def _warp(rgb: np.ndarray, scale: float, tx: float, ty: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    matrix = np.array([[scale, 0.0, (1.0 - scale) * (w / 2.0) + tx],
                       [0.0, scale, (1.0 - scale) * (h / 2.0) + ty]], dtype=np.float32)
    warped = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    support = cv2.warpAffine(np.ones((h, w), np.uint8), matrix, (w, h),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0).astype(bool)
    return warped, support


def _transform(image: Image.Image, scale: float, tx: float, ty: float) -> Image.Image:
    """Compatibility helper used by alignment tests and diagnostics."""
    warped, _ = _warp(np.asarray(image.convert("RGB")), scale, tx, ty)
    return Image.fromarray(warped, "RGB")


def _ssim_score(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Masked global SSIM approximation; returns a similarity in [-1,1]."""
    af = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)[mask]
    bf = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)[mask]
    if af.size < 2:
        return -1.0
    ma, mb = float(af.mean()), float(bf.mean())
    va, vb = float(af.var()), float(bf.var())
    cov = float(np.mean((af - ma) * (bf - mb)))
    c1, c2 = 6.5025, 58.5225
    return float(np.clip(((2 * ma * mb + c1) * (2 * cov + c2)) /
                         ((ma * ma + mb * mb + c1) * (va + vb + c2) + 1e-6), -1.0, 1.0))


def _score(original: np.ndarray, candidate: np.ndarray, mask: np.ndarray,
           boundary_weight: np.ndarray) -> tuple[float, dict]:
    """Robust fitting loss; large context disagreements are down-weighted."""
    if int(mask.sum()) < 64:
        return float("inf"), {}
    og = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cg = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ogx = cv2.Sobel(og, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    ogy = cv2.Sobel(og, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    cgx = cv2.Sobel(cg, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    cgy = cv2.Sobel(cg, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad_diff = np.sqrt((ogx - cgx) ** 2 + (ogy - cgy) ** 2)
    colour_diff = np.mean(np.abs(original.astype(np.float32) -
                                 candidate.astype(np.float32)), axis=2)
    residual = np.minimum(grad_diff / 180.0, 1.0) * 0.65 + np.minimum(colour_diff / 100.0, 1.0) * 0.35
    selected_residual = residual[mask]
    median = float(np.median(selected_residual))
    mad = float(np.median(np.abs(selected_residual - median)))
    cutoff = max(0.12, median + 3.0 * 1.4826 * mad)
    reliability = np.ones_like(residual, dtype=np.float32)
    reliability[mask] = np.minimum(1.0, cutoff / np.maximum(residual[mask], 1e-6))
    weights = boundary_weight[mask] * reliability[mask]
    weights = weights / max(float(weights.sum()), 1e-6)
    grad = float(np.sum(np.minimum(grad_diff[mask], 180.0) * weights) / 180.0)
    colour = float(np.sum(np.minimum(colour_diff[mask], 100.0) * weights) / 100.0)
    ssim = _ssim_score(original, candidate, mask)
    # Do not let the legacy global-SSIM approximation dominate registration:
    # global image statistics can prefer a visibly wrong zoom when illumination
    # or texture changes. Keep it diagnostic-only; fit on robust local gradients
    # and colour residuals instead.
    loss = 0.65 * grad + 0.35 * colour
    return loss, {"ssim": ssim, "gradient_similarity": 1.0 - grad,
                  "colour_similarity": 1.0 - colour, "robust_cutoff": cutoff,
                  "downweighted_context_fraction": float(np.mean(reliability[mask] < 0.999))}


def _validation_score(original: np.ndarray, candidate: np.ndarray,
                      mask: np.ndarray) -> float:
    """Independent held-out metric: texture residual + gradient orientation."""
    if int(mask.sum()) < 32:
        return float("inf")
    og = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cg = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ogx = cv2.Sobel(og, cv2.CV_32F, 1, 0, ksize=3)
    ogy = cv2.Sobel(og, cv2.CV_32F, 0, 1, ksize=3)
    cgx = cv2.Sobel(cg, cv2.CV_32F, 1, 0, ksize=3)
    cgy = cv2.Sobel(cg, cv2.CV_32F, 0, 1, ksize=3)
    omag = np.sqrt(ogx * ogx + ogy * ogy)
    cmag = np.sqrt(cgx * cgx + cgy * cgy)
    denom = np.maximum(omag * cmag, 1e-5)
    cosine = np.clip((ogx * cgx + ogy * cgy) / denom, -1.0, 1.0)
    edge_weights = np.clip(np.minimum(omag, cmag) / 24.0, 0.0, 1.0)
    orientation_error = 1.0 - np.abs(cosine)
    orientation = float(np.sum(orientation_error[mask] * edge_weights[mask]) /
                        max(float(edge_weights[mask].sum()), 1e-6))
    oa = og - cv2.GaussianBlur(og, (0, 0), 3.0)
    ca = cg - cv2.GaussianBlur(cg, (0, 0), 3.0)
    texture_error = float(np.median(np.abs(oa[mask] - ca[mask])))
    return 0.65 * orientation + 0.35 * min(texture_error / 32.0, 1.0)


def _split_fit_validation(mask: np.ndarray, tile_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Spatial checkerboard split so validation tiles are held out from fitting."""
    h, w = mask.shape
    yy, xx = np.indices((h, w))
    validation_tiles = ((xx // tile_size + yy // tile_size) % 2) == 0
    fit = mask & ~validation_tiles
    validation = mask & validation_tiles
    if int(fit.sum()) < 64 or int(validation.sum()) < 32:
        validation_tiles = ((xx + yy) % 2) == 0
        fit = mask & ~validation_tiles
        validation = mask & validation_tiles
    return fit, validation

def _phase_shift(original: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    a = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Suppress the selected subject so registration is driven by context.
    m = mask.astype(np.float32)
    a *= m
    b *= m
    try:
        shift, response = cv2.phaseCorrelate(a, b)
        if not np.isfinite(shift).all() or response < 0.01:
            return 0.0, 0.0
        # phaseCorrelate returns the shift from first to second; invert to map
        # generated pixels back onto the original coordinate frame.
        return float(-shift[0]), float(-shift[1])
    except cv2.error:
        return 0.0, 0.0


def _ecc_seed(original: np.ndarray, generated: np.ndarray, context_mask: np.ndarray,
              full_width: int, full_height: int) -> tuple[float, float, float] | None:
    """Estimate a near-uniform affine transform with ECC, then constrain it.

    ECC's affine matrix is converted to a uniform scale + translation seed.
    Significant rotation, shear or anisotropic scaling is rejected rather than
    introducing a distortion that the product's alignment model does not allow.
    """
    template = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    moving = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(
            template, moving, warp, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 1e-5),
            inputMask=context_mask.astype(np.uint8), gaussFiltSize=5)
        # ECC's matrix is used with WARP_INVERSE_MAP, so invert to get the
        # forward generated->original transform represented by our search.
        forward = cv2.invertAffineTransform(warp)
        col1 = forward[:2, 0]
        col2 = forward[:2, 1]
        s1, s2 = float(np.linalg.norm(col1)), float(np.linalg.norm(col2))
        scale = (s1 + s2) / 2.0
        orthogonality = abs(float(np.dot(col1, col2))) / max(s1 * s2, 1e-6)
        anisotropy = abs(s1 - s2) / max(scale, 1e-6)
        rotation_radians = math.atan2(
            float(forward[1, 0] - forward[0, 1]),
            float(forward[0, 0] + forward[1, 1]),
        )
        # Do not quietly turn this into a rotation/shear/warp system.
        if (not np.isfinite([scale, rotation_radians, *forward.ravel()]).all()
                or anisotropy > 0.035 or orthogonality > 0.035
                or abs(rotation_radians) > 0.035):
            return None
        sx = original.shape[1] / full_width
        sy = original.shape[0] / full_height
        tx_small = float(forward[0, 2]) - (1.0 - scale) * (original.shape[1] / 2.0)
        ty_small = float(forward[1, 2]) - (1.0 - scale) * (original.shape[0] / 2.0)
        return scale, tx_small / max(sx, 1e-8), ty_small / max(sy, 1e-8)
    except cv2.error:
        return None


def align_generated_image(
    original_crop: Image.Image,
    generated_image: Image.Image,
    *, selection_box: tuple[int, int, int, int],
    points_local: list[list[int]] | None = None,
    valid_rect: tuple[int, int, int, int] | None = None,
    context_margin_px: int = 40,
    safety_margin_px: int = 10,
    max_translation_px: float = 30.0,
    min_scale: float = 0.80,
    max_scale: float = 1.20,
    minimum_improvement_percent: float = 0.75,
    scoring_max_side: int = 160,
    registration_band_px: float = 8.0,
    validation_min_improvement_percent: float = 0.25,
    coarse_best_count: int = 4,
) -> AlignmentResult:
    original_crop = original_crop.convert("RGB")
    generated_image = generated_image.convert("RGB").resize(original_crop.size, Image.Resampling.LANCZOS)
    w, h = original_crop.size
    if valid_rect is None:
        valid_rect = (0, 0, w, h)
    selection = _selection_mask(original_crop.size, points_local, selection_box)
    valid = _valid_mask(original_crop.size, valid_rect)
    context = (~selection) & valid
    if int(context.sum()) < 64 or min(w, h) < 16:
        return AlignmentResult(
            generated_image, False, 1.0, 0, 0, 0, 0, 0, 0,
            "low", "Insufficient valid context; pipeline-normalized identity retained.",
            context_coverage=1.0,
            search_config={"method": "pipeline_identity_fallback",
                           "valid_context_pixels": int(context.sum())},
            context_sufficient=False)

    factor = min(1.0, scoring_max_side / max(w, h))
    sw, sh = max(16, round(w * factor)), max(16, round(h * factor))
    orig = cv2.resize(np.asarray(original_crop), (sw, sh), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
    gen = cv2.resize(np.asarray(generated_image), (sw, sh), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
    ctx = cv2.resize(context.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
    valid_small = cv2.resize(valid.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
    selection_distance = cv2.distanceTransform((~selection).astype(np.uint8), cv2.DIST_L2, 3)
    registration_context_full = context & (selection_distance >= float(registration_band_px))
    registration_small = cv2.resize(registration_context_full.astype(np.uint8), (sw, sh),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
    fit_mask, validation_mask = _split_fit_validation(registration_small)
    validation_available = int(fit_mask.sum()) >= 64 and int(validation_mask.sum()) >= 32
    if not validation_available:
        return AlignmentResult(
            generated_image, False, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
            "low", "Insufficient stable context outside the selection exclusion band; "
                  "pipeline-normalized identity retained.",
            context_coverage=1.0,
            search_config={"method": "pipeline_identity_fallback",
                           "registration_exclusion_band_px": registration_band_px,
                           "registration_context_pixels": int(registration_small.sum()),
                           "validation_context_pixels": int(validation_mask.sum())},
            registration_context_pixels=int(registration_small.sum()),
            validation_context_pixels=int(validation_mask.sum()),
            context_sufficient=False,
        )
    # Registration cannot infer geometry from a featureless context. Avoid
    # spending thousands of comparisons (or asking ECC to optimise a flat field).
    original_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
    texture_mask = registration_small if int(registration_small.sum()) >= 64 else ctx
    if float(np.std(original_gray[texture_mask])) < 1.5:
        return AlignmentResult(generated_image, False, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1, "low",
            "Surrounding context has insufficient visual structure for reliable registration; identity retained.",
            context_coverage=1.0,
            search_config={"method": "skipped_low_texture_context", "translation_limit_px": max_translation_px,
                           "scale_range": [min_scale, max_scale],
                           "registration_context_pixels": int(registration_small.sum()),
                           "validation_context_pixels": int(validation_mask.sum())},
            registration_context_pixels=int(registration_small.sum()),
            validation_context_pixels=int(validation_mask.sum()),
            context_sufficient=False)
    # Weight the first 12px around the polygon more heavily, tapering into context.
    selection_small = cv2.resize(selection.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
    dist = cv2.distanceTransform((~selection_small).astype(np.uint8), cv2.DIST_L2, 3)
    boundary_weight = 1.0 + 2.0 * np.exp(-dist / max(1.0, 12.0 * sw / w))
    phase_x_small, phase_y_small = _phase_shift(orig, gen, fit_mask)
    sx, sy = sw / w, sh / h
    phase_x, phase_y = phase_x_small / max(sx, 1e-8), phase_y_small / max(sy, 1e-8)
    ecc_seed = _ecc_seed(orig, gen, fit_mask, w, h)

    max_t = min(float(max_translation_px), max(0.0, float(context_margin_px - safety_margin_px)))
    evaluated = 0
    candidates: list[tuple[float, float, float, float, dict, float]] = []
    initial_score = None

    def evaluate(scale: float, tx: float, ty: float):
        nonlocal evaluated, initial_score
        if not (min_scale - 1e-8 <= scale <= max_scale and -max_t <= tx <= max_t and -max_t <= ty <= max_t):
            return
        scaled_tx, scaled_ty = tx * sx, ty * sy
        warped, support = _warp(gen, scale, scaled_tx, scaled_ty)
        matrix = np.array([[scale, 0.0, (1.0 - scale) * (sw / 2.0) + scaled_tx],
                           [0.0, scale, (1.0 - scale) * (sh / 2.0) + scaled_ty]], dtype=np.float32)
        # A padded edge pixel is not genuine context. Warp the valid-source mask
        # alongside the generated image and require the selected area to remain
        # backed by real source pixels as well.
        valid_support = cv2.warpAffine(valid_small.astype(np.uint8), matrix, (sw, sh),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        real_support = support & valid_support
        selection_coverage = float(np.count_nonzero(selection_small & real_support)) / max(1, int(selection_small.sum()))
        overlap = ctx & real_support
        coverage = float(overlap.sum()) / max(1, int(ctx.sum()))
        evaluated += 1
        if coverage < 0.88 or selection_coverage < 0.98:
            return
        fit_overlap = fit_mask & real_support
        if int(fit_overlap.sum()) < 48:
            return
        score, detail = _score(orig, warped, fit_overlap, boundary_weight)
        score += (1.0 - coverage) * 0.75
        if initial_score is None and scale == 1.0 and tx == 0 and ty == 0:
            initial_score = score
        if np.isfinite(score):
            candidates.append((score, scale, tx, ty, detail, coverage))

    evaluate(1.0, 0.0, 0.0)
    # Cover the complete permitted space efficiently, then refine the best
    # coarse basins. A 5% scale grid avoids leaving the correct zoom outside
    # every local refinement basin.
    translations = np.unique(np.append(np.arange(-max_t, max_t + 0.001, 10.0), 0.0))
    scales = np.unique(np.append(np.arange(min_scale, max_scale + 0.001, 0.05), 1.0))
    for scale in scales:
        for tx in translations:
            for ty in translations:
                if abs(scale - 1.0) < 1e-8 and tx == 0 and ty == 0:
                    continue
                evaluate(float(scale), float(tx), float(ty))
    # Phase-correlation seed is evaluated explicitly even if it falls between grid points.
    evaluate(1.0, float(np.clip(phase_x, -max_t, max_t)), float(np.clip(phase_y, -max_t, max_t)))
    if ecc_seed is not None:
        evaluate(float(ecc_seed[0]), float(np.clip(ecc_seed[1], -max_t, max_t)),
                 float(np.clip(ecc_seed[2], -max_t, max_t)))
    if not candidates:
        return AlignmentResult(generated_image, False, 1.0, 0, 0, 0, 0, 0, evaluated,
                               "low", "No candidate retained sufficient real-context overlap.",
                               phase_x, phase_y, context_coverage=0.0)
    if initial_score is None:
        identity = next((c for c in candidates if c[1] == 1 and c[2] == 0 and c[3] == 0), None)
        initial_score = identity[0] if identity else min(c[0] for c in candidates)

    # Fine search around several best coarse candidates to avoid local minima.
    coarse_best = sorted(candidates, key=lambda c: c[0])[:max(1, int(coarse_best_count))]
    refinement_seeds = [(c[1], c[2], c[3]) for c in coarse_best]
    # Always refine identity and the independent registration estimates too;
    # a coarse ranking must not be allowed to exclude the true local basin.
    refinement_seeds.append((1.0, 0.0, 0.0))
    refinement_seeds.append((1.0, float(np.clip(phase_x, -max_t, max_t)),
                             float(np.clip(phase_y, -max_t, max_t))))
    if ecc_seed is not None:
        refinement_seeds.append((float(ecc_seed[0]), float(np.clip(ecc_seed[1], -max_t, max_t)),
                                 float(np.clip(ecc_seed[2], -max_t, max_t))))
    unique_seeds = []
    seen_seeds = set()
    for seed_scale, seed_tx, seed_ty in refinement_seeds:
        key = (round(seed_scale, 3), round(seed_tx, 1), round(seed_ty, 1))
        if key not in seen_seeds:
            seen_seeds.add(key)
            unique_seeds.append((seed_scale, seed_tx, seed_ty))
    for scale0, tx0, ty0 in unique_seeds:
        for scale in np.arange(max(min_scale, scale0 - 0.05), min(max_scale, scale0 + 0.0501), 0.025):
            for tx in np.arange(max(-max_t, tx0 - 5), min(max_t, tx0 + 5.01), 2.0):
                for ty in np.arange(max(-max_t, ty0 - 5), min(max_t, ty0 + 5.01), 2.0):
                    evaluate(float(scale), float(tx), float(ty))
    best = min(candidates, key=lambda c: c[0])
    # Deterministic pattern refinement to sub-quarter-pixel and 0.1% scale steps.
    for t_step, s_step in ((0.5, 0.005), (0.25, 0.0025), (0.125, 0.001)):
        _, bs, bx, by, _, _ = min(candidates, key=lambda c: c[0])
        for scale in (max(min_scale, bs - s_step), bs, min(max_scale, bs + s_step)):
            for tx in (max(-max_t, bx - t_step), bx, min(max_t, bx + t_step)):
                for ty in (max(-max_t, by - t_step), by, min(max_t, by + t_step)):
                    evaluate(float(scale), float(tx), float(ty))
    # Validate several strong fitting candidates, not only the single training
    # winner. A candidate that fits altered context can overfit the fitting set;
    # held-out context selects among the top fitting basins.
    fit_best_candidate = min(candidates, key=lambda c: c[0])
    fit_best_score, fit_best_scale, fit_best_tx, fit_best_ty, _, _ = fit_best_candidate
    fit_best_improvement = max(0.0, (initial_score - fit_best_score) /
                               max(initial_score, 1e-8) * 100.0)
    identity_warp, identity_support = _warp(gen, 1.0, 0.0, 0.0)
    validation_candidates = sorted(candidates, key=lambda c: c[0])[:12]
    identity_candidate = next((c for c in candidates
                               if abs(c[1] - 1.0) < 1e-8 and abs(c[2]) < 1e-8 and abs(c[3]) < 1e-8), None)
    if identity_candidate is not None and identity_candidate not in validation_candidates:
        validation_candidates.append(identity_candidate)

    validated = []
    for candidate in validation_candidates:
        candidate_score, candidate_scale, candidate_tx, candidate_ty, candidate_details, candidate_coverage = candidate
        candidate_warp, candidate_support = _warp(gen, candidate_scale,
                                                  candidate_tx * sx, candidate_ty * sy)
        common = validation_mask & identity_support & candidate_support
        common_coverage = float(common.sum()) / max(1, int(validation_mask.sum()))
        if common_coverage < 0.88 or int(common.sum()) < 32:
            continue
        baseline_validation = _validation_score(orig, identity_warp, common)
        candidate_validation = _validation_score(orig, candidate_warp, common)
        validated.append((candidate_validation, candidate, baseline_validation, common_coverage))

    if validated:
        validation_final, selected_candidate, validation_initial, validation_coverage = min(
            validated, key=lambda row: row[0]
        )
        best_score, best_scale, best_tx, best_ty, details, coverage = selected_candidate
        validation_improvement = max(
            0.0, (validation_initial - validation_final) / max(validation_initial, 1e-8) * 100.0
        )
        validation_available = True
    else:
        best_score, best_scale, best_tx, best_ty, details, coverage = min(candidates, key=lambda c: c[0])
        validation_initial = validation_final = float("inf")
        validation_improvement = 0.0
        validation_coverage = 0.0
        validation_available = False

    improvement = max(0.0, (initial_score - best_score) / max(initial_score, 1e-8) * 100.0)
    proposed_scale, proposed_tx, proposed_ty = best_scale, best_tx, best_ty
    proposed_improvement = improvement
    applied = bool(validation_available and best_score < initial_score
                   and improvement >= minimum_improvement_percent
                   and validation_final < validation_initial
                   and validation_improvement >= validation_min_improvement_percent
                   and (abs(best_scale - 1) >= .002 or abs(best_tx) >= .75 or abs(best_ty) >= .75))
    if not applied:
        best_scale, best_tx, best_ty = 1.0, 0.0, 0.0
        best_score = initial_score
        details = _score(orig, gen, fit_mask, boundary_weight)[1]
        reason = ("Independent held-out context did not confirm the proposed transform; identity retained."
                  if validation_available else
                  "Insufficient independent validation context; identity retained.")
    else:
        reason = "Transform improved fitting context and independent held-out validation."
    final, _ = _warp(np.asarray(generated_image), best_scale, best_tx, best_ty)
    confidence = ("high" if validation_available and validation_improvement >= 20 and validation_coverage >= .97
                  else "medium" if validation_available and validation_improvement >= 5 and validation_coverage >= .93
                  else "low")
    return AlignmentResult(Image.fromarray(final, "RGB"), applied, float(best_scale),
        float(best_tx), float(best_ty), float(initial_score), float(best_score),
        float(improvement), evaluated, confidence, reason, phase_x, phase_y,
        float(details.get("ssim", 0.0)), float(details.get("gradient_similarity", 0.0)),
        float(details.get("colour_similarity", 0.0)), float(coverage),
        {"translation_limit_px": max_t, "scale_range": [min_scale, max_scale],
         "coarse_translation_step_px": 10, "coarse_scale_step": .05,
         "fine_translation_step_px": 2, "fine_scale_step": .025,
         "final_translation_step_px": .125, "final_scale_step": .001,
         "registration_exclusion_band_px": registration_band_px,
         "coarse_best_candidates_refined": int(coarse_best_count),
         "coarse_candidates": int(len(translations) ** 2 * len(scales)),
         "fine_basins_refined": int(len(unique_seeds)),
         "method": "pipeline-aware phase correlation + masked ECC + robust fit + held-out validation",
         "ecc_seed": list(ecc_seed) if ecc_seed is not None else None,
         "phase_correlation_small_px": [phase_x_small, phase_y_small],
         "phase_correlation_crop_px": [phase_x, phase_y],
         "validation_metric": "held-out gradient orientation + high-pass texture residual",
         "validation_candidates_evaluated": len(validated),
         "validation_coverage": round(validation_coverage, 5)},
        float(validation_initial if np.isfinite(validation_initial) else 0.0),
        float(validation_final if np.isfinite(validation_final) else 0.0),
        float(validation_improvement), int(fit_mask.sum()), int(validation_mask.sum()),
        bool(validation_available),
        float(proposed_scale), float(proposed_tx), float(proposed_ty),
        float(proposed_improvement),
        float(fit_best_scale), float(fit_best_tx), float(fit_best_ty),
        float(fit_best_improvement))
