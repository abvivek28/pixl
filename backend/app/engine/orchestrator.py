from __future__ import annotations
from pathlib import Path
from typing import Any
import json, time, os
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from .deterministic import (Box, expand_box, prepare_model_input, to_oklab, from_oklab,
                            delta_e, _ring_mask, _median_oklab_offset, _feather_mask,
                            _laplacian_variance, file_sha256, normalize_provider_output)
from app.providers.cloudflare_klein4b import EditProvider, CloudflareKlein4B, ProviderError
from app.planning.meta_planner import MetaPlanner, PlanningResult, convert_box_2d_to_pixels
from .alignment import align_generated_image
from .seam_detection import polygon_mask, evaluate_seam, adaptive_feather_alpha

class FakeHueShiftProvider(EditProvider):
    """Offline test provider used when --provider fake is selected."""
    name = "fake"
    model = "hue-shift-v1"
    def edit(self, image, instruction, out_size, seed, *, preserve=None):
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
        shift = 17  # deterministic 24-degree-ish shift in PIL's 0..255 hue space
        hsv[..., 0] = ((hsv[..., 0].astype(np.uint16) + shift) % 256).astype(np.uint8)
        return Image.fromarray(hsv, "HSV").convert("RGB")

def _texture_ratio(patch: np.ndarray, ring_crop: np.ndarray, ring_mask: np.ndarray) -> float:
    patch_var = _laplacian_variance(patch)
    ring_image = ring_crop.copy()
    if ring_mask.any():
        ring_image[~ring_mask] = np.median(ring_crop[ring_mask], axis=0).astype(np.uint8)
    ring_var = _laplacian_variance(ring_image)
    return float(patch_var / max(ring_var, 1e-6))

def run_edit(image_path: str | Path, instruction: str, output_dir: str | Path, *,
             selected_box: Box | None = None, points: list[list[int]] | None = None,
             box_source: str = "manual", provider: EditProvider | None = None,
             planner: MetaPlanner | None = None, seed: int | None = 0,
             min_margin: int = 8, feather_px: int = 4,
             coordinate_format_verified: bool = False, plan_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Orchestrate plan -> box/points -> crop -> provider -> align/composite -> diff -> receipt."""
    started = time.perf_counter()
    image_path, output_dir = Path(image_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened: original = opened.convert("RGB").copy()
    original_arr = np.asarray(original, dtype=np.uint8)
    h, w = original_arr.shape[:2]

    planning: PlanningResult | None = None
    plan_data = dict(plan_data or {})
    if planner is not None:
        planning = planner.plan(original, instruction)
        if planning.plan is not None:
            plan_data.update(planning.plan.model_dump(mode="json"))
            if not instruction.strip(): instruction = planning.plan.instruction
        if planning.status == "global_warning":
            return {"status": planning.status, "message": planning.message,
                    "planning": planning.model_dump(mode="json"), "warnings": planning.warnings}
        if selected_box is None and planning.plan and planning.plan.box_2d and coordinate_format_verified:
            coords = convert_box_2d_to_pixels(planning.plan.box_2d, w, h,
                verified_format="ymin_xmin_ymax_xmax_0_1000")
            selected_box = Box(*coords)
            box_source = "meta"
        elif selected_box is None:
            return {"status": planning.status, "message": planning.message,
                    "planning": planning.model_dump(mode="json"),
                    "warnings": planning.warnings + ["No pixel box supplied; Meta coordinate conversion is disabled until five-image verification is complete."]}

    if selected_box is None and points and len(points) >= 3:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        selected_box = Box(max(0, min(xs)), max(0, min(ys)), min(w, max(xs) + 1), min(h, max(ys) + 1))
        box_source = "freehand"

    if selected_box is None:
        raise ValueError("A manual selected_box or points list is required until Meta coordinate coordinates are verified on five images.")
    B = Box(max(0, selected_box.x1), max(0, selected_box.y1), min(w, selected_box.x2), min(h, selected_box.y2))
    if B.width <= 0 or B.height <= 0: raise ValueError("Selected box must have positive area inside the image.")

    # Give registration a 30px movement budget plus a 10px genuine-context
    # safety band. Edge padding is tracked separately and never scored as context.
    context_margin = 40
    virtual_x1, virtual_y1 = B.x1 - context_margin, B.y1 - context_margin
    virtual_x2, virtual_y2 = B.x2 + context_margin, B.y2 + context_margin
    E = Box(virtual_x1, virtual_y1, virtual_x2, virtual_y2)
    clipped_x1, clipped_y1 = max(0, virtual_x1), max(0, virtual_y1)
    clipped_x2, clipped_y2 = min(w, virtual_x2), min(h, virtual_y2)
    clipped = np.asarray(original.crop((clipped_x1, clipped_y1, clipped_x2, clipped_y2)), dtype=np.uint8)
    pad_left, pad_top = clipped_x1 - virtual_x1, clipped_y1 - virtual_y1
    pad_right, pad_bottom = virtual_x2 - clipped_x2, virtual_y2 - clipped_y2
    context_arr = np.pad(clipped, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
    crop = Image.fromarray(context_arr.astype(np.uint8), "RGB")
    model_input, out_size = prepare_model_input(crop, max_side=768, preferred_side=768)
    provider = provider or FakeHueShiftProvider()
    preserve = plan_data.get("preserve", []) if plan_data else []
    effective_instruction = plan_data.get("instruction") or instruction or "Apply a localized edit"
    generated_provider_raw = provider.edit(model_input, effective_instruction, out_size, seed, preserve=preserve).convert("RGB")
    provider_output_size = generated_provider_raw.size
    provider_raw_path = output_dir / "generated_provider_raw.png"
    generated_provider_raw.save(provider_raw_path, format="PNG", optimize=False)
    generated, normalization = normalize_provider_output(generated_provider_raw, crop.size)
    normalization["model_input_resize_scale_x"] = round(model_input.width / crop.width, 10)
    normalization["model_input_resize_scale_y"] = round(model_input.height / crop.height, 10)
    original_crop = np.asarray(crop, dtype=np.uint8)

    # Keep the user's selection fixed in original-image coordinates. Alignment
    # transforms only the generated crop, never the polygon or compositing mask.
    local_selection_box = (
        context_margin, context_margin,
        context_margin + B.width, context_margin + B.height,
    )
    local_points = None
    if points and len(points) >= 3:
        local_points = [[int(round(p[0] - virtual_x1)),
                         int(round(p[1] - virtual_y1))] for p in points]

    valid_context_rect = (
        pad_left, pad_top,
        pad_left + clipped.shape[1], pad_top + clipped.shape[0],
    )
    raw_generated_path = output_dir / "generated_unaligned.png"
    generated.save(raw_generated_path, format="PNG", optimize=False)

    alignment = align_generated_image(
        crop, generated,
        selection_box=local_selection_box,
        points_local=local_points,
        valid_rect=valid_context_rect,
        context_margin_px=context_margin,
        safety_margin_px=10,
        max_translation_px=30.0,
        min_scale=0.80,
        max_scale=1.20,
        registration_band_px=8.0,
        coarse_best_count=4,
    )
    generated = alignment.image
    aligned_generated_path = output_dir / "generated_aligned.png"
    generated.save(aligned_generated_path, format="PNG", optimize=False)
    generated_arr = np.asarray(generated, dtype=np.uint8)

    # Colour correction uses valid context outside the actual selection polygon,
    # including unselected pixels inside its bounding rectangle.
    selection_mask = np.zeros((crop.height, crop.width), dtype=bool)
    if local_points and len(local_points) >= 3:
        polygon = Image.new("L", crop.size, 0)
        ImageDraw.Draw(polygon).polygon(
            [(x, y) for x, y in local_points], fill=255
        )
        selection_mask = np.asarray(polygon, dtype=np.uint8) > 0
    else:
        selection_mask[local_selection_box[1]:local_selection_box[3],
                       local_selection_box[0]:local_selection_box[2]] = True
    valid_mask = np.zeros((crop.height, crop.width), dtype=bool)
    valid_mask[pad_top:pad_top + clipped.shape[0],
               pad_left:pad_left + clipped.shape[1]] = True
    ring = (~selection_mask) & valid_mask
    if ring.any():
        offset = _median_oklab_offset(original_crop[ring], generated_arr[ring])
    else:
        offset = np.zeros(3, dtype=np.float32)
    corrected = from_oklab(to_oklab(generated_arr) + offset)
    bx1, by1 = context_margin, context_margin
    bx2, by2 = bx1 + B.width, by1 + B.height
    patch = corrected[by1:by2, bx1:bx2]
    old_patch = original_arr[B.y1:B.y2, B.x1:B.x2]
    poly_coords = None
    if points and len(points) >= 3:
        poly_coords = [(p[0] - B.x1, p[1] - B.y1) for p in points]
        mask_patch = polygon_mask((B.width, B.height),
                                  [[int(x), int(y)] for x, y in poly_coords],
                                  (0, 0, B.width, B.height))
        alpha_2d = adaptive_feather_alpha(mask_patch, base_radius=2.0)
        alpha = alpha_2d[..., None]
    else:
        mask_patch = np.ones((B.height, B.width), dtype=bool)
        alpha = _feather_mask(B.width, B.height, feather_px, B.x1 == 0, B.y1 == 0, B.x2 == w, B.y2 == h)[..., None]
    blended = np.clip(np.rint(old_patch * (1-alpha) + patch * alpha), 0, 255).astype(np.uint8)
    # Evaluate the real polygon contour, then adapt feathering only if the
    # measured discontinuity is substantial. The original mask remains fixed.
    seam_composite = original_crop.copy()
    seam_composite[by1:by2, bx1:bx2] = blended
    seam_before = evaluate_seam(original_crop, seam_composite, selection_mask)
    if poly_coords is not None and max(
            seam_before.get("excess_boundary_step", 0.0),
            seam_before.get("p95_excess_boundary_step", 0.0) * 0.65) > 12.0:
        alpha_2d = adaptive_feather_alpha(mask_patch, base_radius=2.0, seam_metrics=seam_before)
        alpha = alpha_2d[..., None]
        blended = np.clip(np.rint(old_patch * (1-alpha) + patch * alpha), 0, 255).astype(np.uint8)
    result_arr = original_arr.copy()
    result_arr[B.y1:B.y2, B.x1:B.x2] = blended

    changed_pixels = np.any(result_arr != original_arr, axis=2)
    outside_box = np.ones((h, w), dtype=bool)
    outside_box[B.y1:B.y2, B.x1:B.x2] = False
    outside_changed = int(np.count_nonzero(changed_pixels & outside_box))
    hard_selection_full = np.zeros((h, w), dtype=bool)
    hard_selection_full[B.y1:B.y2, B.x1:B.x2] = mask_patch
    outside_hard_selection_changed = int(np.count_nonzero(changed_pixels & ~hard_selection_full))
    feather_band_patch = (mask_patch & (alpha[..., 0] < 0.999))
    feather_band_full = np.zeros((h, w), dtype=bool)
    feather_band_full[B.y1:B.y2, B.x1:B.x2] = feather_band_patch
    feather_band_changed = int(np.count_nonzero(changed_pixels & feather_band_full))
    outside_feather_support_changed = int(np.count_nonzero(changed_pixels & ~hard_selection_full))
    full_delta = delta_e(original_arr, result_arr)
    box_delta = full_delta[B.y1:B.y2, B.x1:B.x2]
    changed_pct = float(np.mean(box_delta > 0.05) * 100)
    mean_delta = float(np.mean(box_delta)) if box_delta.size else 0.0
    corrected_ring = from_oklab(to_oklab(generated_arr) + offset)
    ring_delta = delta_e(original_crop[ring], corrected_ring[ring]) if ring.any() else np.array([], dtype=np.float32)
    ring_drift = float(np.median(ring_delta)) if ring_delta.size else 0.0

    seam_composite = original_crop.copy()
    seam_composite[by1:by2, bx1:bx2] = blended
    seam_metrics = evaluate_seam(original_crop, seam_composite, selection_mask)
    seam_ratio = float(seam_metrics.get("excess_boundary_step", 0.0) /
                       max(float(seam_metrics.get("baseline_step", 0.0)), 1.0))
    texture_ratio = _texture_ratio(blended, original_crop, ring)

    result_path, heat_path = output_dir / "result.png", output_dir / "heatmap.png"
    Image.fromarray(result_arr, "RGB").save(result_path, format="PNG")
    heat = np.zeros((h,w,3), dtype=np.uint8)
    scaled = np.clip(box_delta / max(0.25, float(box_delta.max()) if box_delta.size else 0.25) * 255, 0, 255).astype(np.uint8)
    heat[B.y1:B.y2,B.x1:B.x2,0] = scaled
    heat[B.y1:B.y2,B.x1:B.x2,1] = (scaled.astype(np.float32)*0.28).astype(np.uint8)
    Image.fromarray(heat, "RGB").save(heat_path, format="PNG")
    metrics = {"outside_changed_pixels": outside_changed,
        "outside_selected_polygon_changed_pixels": outside_hard_selection_changed,
        "outside_feather_support_changed_pixels": outside_feather_support_changed,
        "feather_band_changed_pixels": feather_band_changed,
        "percent_box_changed": changed_pct,
        "mean_delta_e_oklab": mean_delta, "ring_drift_median_delta_e_oklab": ring_drift,
        "seam_ratio": seam_ratio, "texture_ratio": texture_ratio, "changed_pixel_threshold_delta_e": 0.05,
        "seam_boundary_mean_step": seam_metrics.get("mean_boundary_step", 0.0),
        "seam_boundary_p95_step": seam_metrics.get("p95_boundary_step", 0.0),
        "seam_excess_boundary_step": seam_metrics.get("excess_boundary_step", 0.0),
        "seam_p95_excess_boundary_step": seam_metrics.get("p95_excess_boundary_step", 0.0),
        "seam_halo_score": seam_metrics.get("halo_score", 0.0),
        "alignment_improvement_percent": round(alignment.improvement_percent, 4),
        "alignment_scale": round(alignment.scale, 5),
        "alignment_translation_x": round(alignment.translation_x, 3),
        "alignment_translation_y": round(alignment.translation_y, 3),
        "alignment_proposed_scale": round(alignment.proposed_scale, 5),
        "alignment_proposed_translation_x": round(alignment.proposed_translation_x, 3),
        "alignment_proposed_translation_y": round(alignment.proposed_translation_y, 3),
        "alignment_fit_best_scale": round(alignment.fit_best_scale, 5),
        "alignment_fit_best_translation_x": round(alignment.fit_best_translation_x, 3),
        "alignment_fit_best_translation_y": round(alignment.fit_best_translation_y, 3),
        "alignment_initial_score": round(alignment.initial_score, 7),
        "alignment_final_score": round(alignment.final_score, 7),
        "alignment_proposed_improvement_percent": round(alignment.proposed_improvement_percent, 4),
        "alignment_fit_best_improvement_percent": round(alignment.fit_best_improvement_percent, 4),
        "alignment_validation_improvement_percent": round(alignment.validation_improvement_percent, 4),
        "alignment_phase_correlation_x": round(alignment.phase_correlation_x, 4),
        "alignment_phase_correlation_y": round(alignment.phase_correlation_y, 4),
        "alignment_context_ssim": round(alignment.context_ssim, 5),
        "alignment_context_gradient_similarity": round(alignment.context_gradient_similarity, 5),
        "alignment_context_colour_similarity": round(alignment.context_colour_similarity, 5),
        "alignment_context_coverage": round(alignment.context_coverage, 5),
        "alignment_registration_context_pixels": alignment.registration_context_pixels,
        "alignment_validation_context_pixels": alignment.validation_context_pixels,
        "alignment_candidates_evaluated": alignment.candidates_evaluated,
        "alignment_context_sufficient": bool(alignment.context_sufficient),
        "elapsed_seconds": round(time.perf_counter()-started, 4)}
    seam_mean = float(seam_metrics.get("excess_boundary_step", 0.0))
    seam_p95 = float(seam_metrics.get("p95_excess_boundary_step", 0.0))
    halo_score = float(seam_metrics.get("halo_score", 0.0))
    checks = [
        {"name":"zero_changes_outside_selected_box", "pass":outside_changed == 0, "actual":outside_changed, "threshold":0},
        {"name":"zero_changes_outside_hard_selection", "pass":outside_hard_selection_changed == 0,
         "actual":outside_hard_selection_changed, "threshold":0},
        {"name":"zero_changes_outside_feather_support", "pass":outside_feather_support_changed == 0,
         "actual":outside_feather_support_changed, "threshold":0},
        {"name":"ring_drift", "pass":ring_drift < .03, "actual":ring_drift, "threshold":"< 0.03"},
        {"name":"polygon_seam_mean_excess", "pass":seam_mean < 24.0,
         "actual":seam_mean, "threshold":"< 24 RGB levels"},
        {"name":"polygon_seam_p95_excess", "pass":seam_p95 < 36.0,
         "actual":seam_p95, "threshold":"< 36 RGB levels"},
        {"name":"polygon_halo", "pass":halo_score < 40.0,
         "actual":halo_score, "threshold":"< 40 luminance levels"},
        {"name":"texture_ratio", "pass":.5 <= texture_ratio <= 2.0, "actual":texture_ratio, "threshold":"0.5–2.0"}]
    warnings = []
    if planning and planning.status == "global_warning": warnings.append("Global edit requested: regional isolation is not guaranteed semantically.")
    if not ring.any(): warnings.append("Expanded crop has no ring pixels; ring color matching was skipped.")
    if B.width < 8 or B.height < 8: warnings.append("Tiny selection; quality metrics may be unreliable.")
    result_bytes = result_path.read_bytes(); master_bytes = image_path.read_bytes()
    summary = (f"Changed {changed_pct:.1f}% of the selected region. Mean OKLab ΔE was {mean_delta:.4f}. "
               f"Pixels changed outside the selection: {outside_changed}. "
               f"{'All hard checks passed.' if all(c['pass'] for c in checks) else 'One or more quality checks need review.'}")
    receipt = {"schema_version":"1.2", "image":{"master_sha256":file_sha256(master_bytes), "result_sha256":file_sha256(result_bytes),
        "master_size":{"width":w,"height":h,"bytes":len(master_bytes)}, "result_size":{"width":w,"height":h,"bytes":len(result_bytes)},
        "result_format":"PNG"}, "boxes":{"selected":B.as_dict(),"expanded":E.as_dict(),"coordinate_convention":"pixel half-open [x1,y1,x2,y2]"},
        "box_source":box_source,"selection_type":"freehand" if points else "box","points":points,"provider":provider.name,"model":provider.model,"seed":seed,"prompt":effective_instruction,
        "plan":plan_data,"metrics":metrics,"metric_legend": {
            "outside_changed_pixels": {"label":"Changed pixels outside selection box", "purpose":"Counts changed pixels outside the selection bounding rectangle; should be zero.", "unit":"pixels"},
            "outside_selected_polygon_changed_pixels": {"label":"Changed pixels outside hard polygon", "purpose":"Counts changed pixels outside the user's actual polygon, excluding allowed feather support.", "unit":"pixels"},
            "outside_feather_support_changed_pixels": {"label":"Changed pixels outside feather support", "purpose":"Detects any generated-image contribution beyond the mask's permitted support.", "unit":"pixels"},
            "feather_band_changed_pixels": {"label":"Changed pixels in feather band", "purpose":"Counts changed pixels in the inward transition band used to soften the composite edge.", "unit":"pixels"},
            "percent_box_changed": {"label":"Selected box changed", "purpose":"Percentage of pixels in the selected bounding box that changed beyond the configured OKLab threshold.", "unit":"percent"},
            "mean_delta_e_oklab": {"label":"Mean OKLab colour difference", "purpose":"Average perceptual colour difference between original and edited pixels in the selected region.", "unit":"OKLab ΔE"},
            "ring_drift_median_delta_e_oklab": {"label":"Context ring colour drift", "purpose":"Median OKLab colour difference in the surrounding context ring after colour matching.", "unit":"OKLab ΔE"},
            "seam_ratio": {"label":"Seam ratio", "purpose":"Boundary discontinuity relative to the original boundary baseline; lower is generally better.", "unit":"ratio"},
            "texture_ratio": {"label":"Texture ratio", "purpose":"Generated-to-original high-frequency texture energy ratio in the measured region; values near 1 indicate similar texture energy.", "unit":"ratio"},
            "changed_pixel_threshold_delta_e": {"label":"Changed-pixel threshold", "purpose":"OKLab ΔE threshold used to classify a pixel as changed.", "unit":"OKLab ΔE"},
            "seam_boundary_mean_step": {"label":"Mean boundary step", "purpose":"Average colour/gradient step measured across the polygon boundary.", "unit":"RGB levels"},
            "seam_boundary_p95_step": {"label":"P95 boundary step", "purpose":"95th percentile of boundary discontinuity; highlights locally bad seam segments.", "unit":"RGB levels"},
            "seam_excess_boundary_step": {"label":"Mean seam excess", "purpose":"Mean boundary discontinuity above the original-image baseline.", "unit":"RGB levels"},
            "seam_p95_excess_boundary_step": {"label":"P95 seam excess", "purpose":"95th percentile of boundary discontinuity above the original-image baseline.", "unit":"RGB levels"},
            "seam_halo_score": {"label":"Halo proxy", "purpose":"Luminance-based indicator of bright/dark halos around the compositing boundary; a heuristic, not a perceptual guarantee.", "unit":"luminance levels"},
            "alignment_improvement_percent": {"label":"Applied alignment improvement", "purpose":"Relative fitting-score improvement for the transformation actually applied; zero when identity is retained.", "unit":"percent"},
            "alignment_scale": {"label":"Applied alignment scale", "purpose":"Uniform zoom factor applied to the generated crop; 1.0 means no zoom.", "unit":"scale factor"},
            "alignment_translation_x": {"label":"Applied horizontal translation", "purpose":"Horizontal movement applied to the generated crop.", "unit":"pixels"},
            "alignment_translation_y": {"label":"Applied vertical translation", "purpose":"Vertical movement applied to the generated crop.", "unit":"pixels"},
            "alignment_proposed_scale": {"label":"Validated proposal scale", "purpose":"Scale of the best candidate selected for independent validation, whether or not ultimately accepted.", "unit":"scale factor"},
            "alignment_proposed_translation_x": {"label":"Proposed horizontal translation", "purpose":"Horizontal translation of the validated candidate before the acceptance gate.", "unit":"pixels"},
            "alignment_proposed_translation_y": {"label":"Proposed vertical translation", "purpose":"Vertical translation of the validated candidate before the acceptance gate.", "unit":"pixels"},
            "alignment_fit_best_scale": {"label":"Fit-best scale", "purpose":"Scale preferred by the fitting objective before independent validation.", "unit":"scale factor"},
            "alignment_fit_best_translation_x": {"label":"Fit-best horizontal translation", "purpose":"Horizontal translation preferred by the fitting objective before independent validation.", "unit":"pixels"},
            "alignment_fit_best_translation_y": {"label":"Fit-best vertical translation", "purpose":"Vertical translation preferred by the fitting objective before independent validation.", "unit":"pixels"},
            "alignment_initial_score": {"label":"Initial alignment score", "purpose":"Fitting loss before applying a candidate transformation; lower is better.", "unit":"score"},
            "alignment_final_score": {"label":"Final alignment score", "purpose":"Fitting loss for the selected candidate; lower is better.", "unit":"score"},
            "alignment_proposed_improvement_percent": {"label":"Proposed fitting improvement", "purpose":"Fitting-score improvement associated with the validated proposal before the acceptance gate.", "unit":"percent"},
            "alignment_fit_best_improvement_percent": {"label":"Fit-best improvement", "purpose":"Best improvement found on the registration/fitting pixels before held-out validation.", "unit":"percent"},
            "alignment_validation_improvement_percent": {"label":"Independent validation improvement", "purpose":"Relative improvement on held-out context pixels; used to decide whether a proposed transform generalizes.", "unit":"percent"},
            "alignment_phase_correlation_x": {"label":"Phase-correlation X estimate", "purpose":"Initial horizontal translation estimate from phase correlation, before refinement.", "unit":"pixels"},
            "alignment_phase_correlation_y": {"label":"Phase-correlation Y estimate", "purpose":"Initial vertical translation estimate from phase correlation, before refinement.", "unit":"pixels"},
            "alignment_context_sufficient": {"label":"Sufficient registration context", "purpose":"Whether enough stable, non-excluded context was available to attempt reliable registration.", "unit":"boolean"},
            "alignment_context_ssim": {"label":"Context SSIM diagnostic", "purpose":"Structural similarity diagnostic on registration context; not used alone to accept an alignment.", "unit":"score"},
            "alignment_context_gradient_similarity": {"label":"Context gradient similarity", "purpose":"Similarity of edge/gradient structures in stable context; higher indicates closer structural agreement.", "unit":"score"},
            "alignment_context_colour_similarity": {"label":"Context colour similarity", "purpose":"Colour agreement in stable context used as part of registration diagnostics.", "unit":"score"},
            "alignment_context_coverage": {"label":"Context coverage", "purpose":"Fraction of relevant context that remains valid/overlapping for registration.", "unit":"fraction"},
            "alignment_registration_context_pixels": {"label":"Registration context pixels", "purpose":"Number of stable context pixels used to fit the candidate transformation.", "unit":"pixels"},
            "alignment_validation_context_pixels": {"label":"Validation context pixels", "purpose":"Number of held-out context pixels used to validate the candidate transformation.", "unit":"pixels"},
            "alignment_candidates_evaluated": {"label":"Alignment candidates evaluated", "purpose":"Number of geometric transformations scored during the search.", "unit":"candidates"},
            "elapsed_seconds": {"label":"Processing time", "purpose":"Elapsed time for the edit processing pipeline.", "unit":"seconds"}
        },"checks":checks,"warnings":warnings,"summary":summary,
        "alignment": {
            "applied": alignment.applied,
            "scale": alignment.scale,
            "translation_x": alignment.translation_x,
            "translation_y": alignment.translation_y,
            "proposed_scale": alignment.proposed_scale,
            "proposed_translation_x": alignment.proposed_translation_x,
            "proposed_translation_y": alignment.proposed_translation_y,
            "proposed_improvement_percent": alignment.proposed_improvement_percent,
            "fit_best_scale": alignment.fit_best_scale,
            "fit_best_translation_x": alignment.fit_best_translation_x,
            "fit_best_translation_y": alignment.fit_best_translation_y,
            "fit_best_improvement_percent": alignment.fit_best_improvement_percent,
            "initial_score": alignment.initial_score,
            "final_score": alignment.final_score,
            "improvement_percent": alignment.improvement_percent,
            "candidates_evaluated": alignment.candidates_evaluated,
            "confidence": alignment.confidence,
            "reason": alignment.reason,
            "context_margin_px": context_margin,
            "safety_margin_px": 10,
            "translation_limit_px": 30,
            "scale_range": [0.80, 1.20],
            "phase_correlation_x": alignment.phase_correlation_x,
            "phase_correlation_y": alignment.phase_correlation_y,
            "context_ssim": alignment.context_ssim,
            "context_gradient_similarity": alignment.context_gradient_similarity,
            "context_colour_similarity": alignment.context_colour_similarity,
            "context_coverage": alignment.context_coverage,
            "validation_initial_score": alignment.validation_initial_score,
            "validation_final_score": alignment.validation_final_score,
            "validation_improvement_percent": alignment.validation_improvement_percent,
            "registration_context_pixels": alignment.registration_context_pixels,
            "validation_context_pixels": alignment.validation_context_pixels,
            "context_sufficient": alignment.context_sufficient,
            "search_config": alignment.search_config,
            "selection_mask_moved": False,
        },
        "artifacts":{"result":str(result_path),"heatmap":str(heat_path),
                     "generated_unaligned":str(raw_generated_path),
                     "generated_aligned":str(aligned_generated_path)},
        "seam": {**seam_metrics,
                 "mean_threshold": 24.0, "p95_threshold": 36.0, "halo_threshold": 40.0,
                 "feather_policy": "inward_only_within_hard_selection"},
        "compositing": {
            "hard_selection_pixels": int(mask_patch.sum()),
            "feather_band_pixels": int(feather_band_patch.sum()),
            "changed_outside_hard_selection_pixels": outside_hard_selection_changed,
            "changed_outside_feather_support_pixels": outside_feather_support_changed,
            "feather_policy": "inward_only_within_hard_selection",
        },
        "model_input":{"width":model_input.width,"height":model_input.height,"proposed_output_width":out_size[0],"proposed_output_height":out_size[1],
                       "selection_aspect_ratio":round(B.width/B.height,6),"context_margin_px":context_margin,
                       "context_padding_px":{"left":pad_left,"top":pad_top,"right":pad_right,"bottom":pad_bottom},
                       "provider_output_width":provider_output_size[0],"provider_output_height":provider_output_size[1],
                       "provider_output_aspect_ratio":round(provider_output_size[0]/provider_output_size[1],6),
                       "crop_aspect_ratio":round(crop.width/crop.height,6),
                       "aspect_ratio_corrected_without_stretching": normalization["aspect_ratio_corrected_without_stretching"],
                       "normalization":normalization,
                       "requested_output_aspect_error": round(abs((out_size[0] / out_size[1]) / (crop.width / crop.height) - 1.0), 8)},
        "pipeline_geometry":{
            "crop_size":{"width":crop.width,"height":crop.height},
            "model_input_size":{"width":model_input.width,"height":model_input.height},
            "model_input_scale":{"x":round(model_input.width/crop.width,10),"y":round(model_input.height/crop.height,10)},
            "requested_provider_output_size":{"width":out_size[0],"height":out_size[1]},
            "provider_output_size":{"width":provider_output_size[0],"height":provider_output_size[1]},
            "provider_output_normalization":normalization,
            "requested_output_aspect_error": round(abs((out_size[0] / out_size[1]) / (crop.width / crop.height) - 1.0), 8),
            "crop_virtual_origin":{"x":virtual_x1,"y":virtual_y1},
            "edge_padding":{"left":pad_left,"top":pad_top,"right":pad_right,"bottom":pad_bottom},
            "alignment_coordinates":"normalized provider output mapped to crop pixel coordinates; all translations are crop-space pixels"
        }}
    receipt["artifacts"]["generated_provider_raw"] = str(provider_raw_path)
    receipt_path = output_dir / "receipt.json"
    receipt["artifacts"]["receipt"] = str(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    # One self-contained JSONL record per edit makes 50-100 edit audits easy.
    diagnostic = {
        "timestamp_unix": time.time(),
        "provider": provider.name, "model": provider.model, "seed": seed,
        "crop_size": [crop.width, crop.height],
        "model_input_size": [model_input.width, model_input.height],
        "requested_output_size": list(out_size),
        "provider_output_size": list(provider_output_size),
        "normalization": normalization,
        "context_padding": {"left": pad_left, "top": pad_top, "right": pad_right, "bottom": pad_bottom},
        "alignment": receipt["alignment"],
    }
    diagnostic_line = (json.dumps(diagnostic, separators=(",", ":")) + "\n").encode("utf-8")
    diagnostic_path = output_dir / "alignment_diagnostics.jsonl"
    diagnostic_path.write_bytes(diagnostic_line)
    # Append-only project-level history; each record is a single append write.
    history_path = output_dir.parent / "alignment_history.jsonl"
    fd = os.open(str(history_path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        os.write(fd, diagnostic_line)
    finally:
        os.close(fd)
    receipt["artifacts"]["alignment_diagnostics"] = str(diagnostic_path)
    receipt["artifacts"]["alignment_history"] = str(history_path)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
