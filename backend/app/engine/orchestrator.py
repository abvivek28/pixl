from __future__ import annotations
from pathlib import Path
from typing import Any
import json, time
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from .deterministic import (Box, expand_box, prepare_model_input, to_oklab, from_oklab,
                            delta_e, _ring_mask, _median_oklab_offset, _feather_mask,
                            _laplacian_variance, file_sha256)
from app.providers.cloudflare_klein4b import EditProvider, CloudflareKlein4B, ProviderError
from app.planning.meta_planner import MetaPlanner, PlanningResult, convert_box_2d_to_pixels

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

    # Fixed context border: exactly 15 source pixels on each side. Near image
    # edges, missing context is edge-padded so the model receives surrounding context
    # and the selected region remains at offset (15, 15) in that crop.
    context_margin = 15
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
    model_input, out_size = prepare_model_input(crop, max_side=504, preferred_side=504)
    provider = provider or FakeHueShiftProvider()
    preserve = plan_data.get("preserve", []) if plan_data else []
    effective_instruction = plan_data.get("instruction") or instruction or "Apply a localized edit"
    generated = provider.edit(model_input, effective_instruction, out_size, seed, preserve=preserve)
    generated = generated.convert("RGB")
    provider_output_size = generated.size
    # Preserve aspect ratio if a provider unexpectedly returns an image with
    # a different aspect ratio: center-crop to target_ratio, then resize to native
    # context dimensions. Never stretch a result into the crop.
    target_ratio = crop.width / crop.height
    output_ratio = generated.width / generated.height
    if abs(output_ratio - target_ratio) > 1e-6:
        if output_ratio > target_ratio:
            fitted_width = max(1, min(generated.width, int(round(generated.height * target_ratio))))
            left = max(0, (generated.width - fitted_width) // 2)
            generated = generated.crop((left, 0, left + fitted_width, generated.height))
        else:
            fitted_height = max(1, min(generated.height, int(round(generated.width / target_ratio))))
            top = max(0, (generated.height - fitted_height) // 2)
            generated = generated.crop((0, top, generated.width, top + fitted_height))
    generated = generated.resize(crop.size, Image.Resampling.LANCZOS)
    original_crop = np.asarray(crop, dtype=np.uint8)
    generated_arr = np.asarray(generated, dtype=np.uint8)
    # B is always exactly 15 pixels inward from the virtual expanded crop.
    # Ring-mask helper uses absolute boxes, so build a local context ring here.
    ring = np.ones((crop.height, crop.width), dtype=bool)
    ring[context_margin:context_margin+B.height, context_margin:context_margin+B.width] = False
    if ring.any():
        offset = _median_oklab_offset(original_crop[ring], generated_arr[ring])
    else:
        offset = np.zeros(3, dtype=np.float32)
    corrected = from_oklab(to_oklab(generated_arr) + offset)
    bx1, by1 = context_margin, context_margin
    bx2, by2 = bx1 + B.width, by1 + B.height
    patch = corrected[by1:by2, bx1:bx2]
    old_patch = original_arr[B.y1:B.y2, B.x1:B.x2]
    if points and len(points) >= 3:
        poly_coords = [(p[0] - B.x1, p[1] - B.y1) for p in points]
        mask_img = Image.new("L", (B.width, B.height), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon(poly_coords, fill=255)
        feathered = mask_img.filter(ImageFilter.GaussianBlur(radius=1.5))
        alpha = (np.asarray(feathered, dtype=np.float32) / 255.0)[..., None]
    else:
        alpha = _feather_mask(B.width, B.height, feather_px, B.x1 == 0, B.y1 == 0, B.x2 == w, B.y2 == h)[..., None]
    blended = np.clip(np.rint(old_patch * (1-alpha) + patch * alpha), 0, 255).astype(np.uint8)
    result_arr = original_arr.copy()
    result_arr[B.y1:B.y2, B.x1:B.x2] = blended

    outside = np.ones((h, w), dtype=bool); outside[B.y1:B.y2, B.x1:B.x2] = False
    outside_changed = int(np.count_nonzero(np.any(result_arr != original_arr, axis=2) & outside))
    full_delta = delta_e(original_arr, result_arr)
    box_delta = full_delta[B.y1:B.y2, B.x1:B.x2]
    changed_pct = float(np.mean(box_delta > 0.05) * 100)
    mean_delta = float(np.mean(box_delta)) if box_delta.size else 0.0
    corrected_ring = from_oklab(to_oklab(generated_arr) + offset)
    ring_delta = delta_e(original_crop[ring], corrected_ring[ring]) if ring.any() else np.array([], dtype=np.float32)
    ring_drift = float(np.median(ring_delta)) if ring_delta.size else 0.0

    rgb = result_arr.astype(np.float32)
    neighbors = []
    if w > 1: neighbors.append(np.linalg.norm(np.diff(rgb, axis=1), axis=2).ravel())
    if h > 1: neighbors.append(np.linalg.norm(np.diff(rgb, axis=0), axis=2).ravel())
    typical = float(np.median(np.concatenate(neighbors))) if neighbors else 1.0
    edge_steps = []
    if B.x1 > 0: edge_steps.extend(np.linalg.norm(rgb[B.y1:B.y2,B.x1] - rgb[B.y1:B.y2,B.x1-1], axis=1).tolist())
    if B.x2 < w: edge_steps.extend(np.linalg.norm(rgb[B.y1:B.y2,B.x2-1] - rgb[B.y1:B.y2,B.x2], axis=1).tolist())
    if B.y1 > 0: edge_steps.extend(np.linalg.norm(rgb[B.y1, B.x1:B.x2] - rgb[B.y1-1,B.x1:B.x2], axis=1).tolist())
    if B.y2 < h: edge_steps.extend(np.linalg.norm(rgb[B.y2-1,B.x1:B.x2] - rgb[B.y2,B.x1:B.x2], axis=1).tolist())
    seam_ratio = float(np.median(edge_steps) / max(typical, 1.0)) if edge_steps else 0.0
    texture_ratio = _texture_ratio(blended, original_crop, ring)

    result_path, heat_path = output_dir / "result.png", output_dir / "heatmap.png"
    Image.fromarray(result_arr, "RGB").save(result_path, format="PNG")
    heat = np.zeros((h,w,3), dtype=np.uint8)
    scaled = np.clip(box_delta / max(0.25, float(box_delta.max()) if box_delta.size else 0.25) * 255, 0, 255).astype(np.uint8)
    heat[B.y1:B.y2,B.x1:B.x2,0] = scaled
    heat[B.y1:B.y2,B.x1:B.x2,1] = (scaled.astype(np.float32)*0.28).astype(np.uint8)
    Image.fromarray(heat, "RGB").save(heat_path, format="PNG")
    metrics = {"outside_changed_pixels": outside_changed, "percent_box_changed": changed_pct,
        "mean_delta_e_oklab": mean_delta, "ring_drift_median_delta_e_oklab": ring_drift,
        "seam_ratio": seam_ratio, "texture_ratio": texture_ratio, "changed_pixel_threshold_delta_e": 0.05,
        "elapsed_seconds": round(time.perf_counter()-started, 4)}
    checks = [
        {"name":"zero_changes_outside_selected_box", "pass":outside_changed == 0, "actual":outside_changed, "threshold":0},
        {"name":"ring_drift", "pass":ring_drift < .03, "actual":ring_drift, "threshold":"< 0.03"},
        {"name":"seam_ratio", "pass":seam_ratio < 1.5, "actual":seam_ratio, "threshold":"< 1.5"},
        {"name":"texture_ratio", "pass":.5 <= texture_ratio <= 2.0, "actual":texture_ratio, "threshold":"0.5–2.0"}]
    warnings = []
    if planning and planning.status == "global_warning": warnings.append("Global edit requested: regional isolation is not guaranteed semantically.")
    if not ring.any(): warnings.append("Expanded crop has no ring pixels; ring color matching was skipped.")
    if B.width < 8 or B.height < 8: warnings.append("Tiny selection; quality metrics may be unreliable.")
    result_bytes = result_path.read_bytes(); master_bytes = image_path.read_bytes()
    summary = (f"Changed {changed_pct:.1f}% of the selected region. Mean OKLab ΔE was {mean_delta:.4f}. "
               f"Pixels changed outside the selection: {outside_changed}. "
               f"{'All hard checks passed.' if all(c['pass'] for c in checks) else 'One or more quality checks need review.'}")
    receipt = {"schema_version":"1.1", "image":{"master_sha256":file_sha256(master_bytes), "result_sha256":file_sha256(result_bytes),
        "master_size":{"width":w,"height":h,"bytes":len(master_bytes)}, "result_size":{"width":w,"height":h,"bytes":len(result_bytes)},
        "result_format":"PNG"}, "boxes":{"selected":B.as_dict(),"expanded":E.as_dict(),"coordinate_convention":"pixel half-open [x1,y1,x2,y2]"},
        "box_source":box_source,"selection_type":"freehand" if points else "box","points":points,"provider":provider.name,"model":provider.model,"seed":seed,"prompt":effective_instruction,
        "plan":plan_data,"metrics":metrics,"checks":checks,"warnings":warnings,"summary":summary,
        "artifacts":{"result":str(result_path),"heatmap":str(heat_path)},
        "model_input":{"width":model_input.width,"height":model_input.height,"proposed_output_width":out_size[0],"proposed_output_height":out_size[1],
                       "selection_aspect_ratio":round(B.width/B.height,6),"context_margin_px":context_margin,
                       "context_padding_px":{"left":pad_left,"top":pad_top,"right":pad_right,"bottom":pad_bottom},
                       "provider_output_width":provider_output_size[0],"provider_output_height":provider_output_size[1],
                       "provider_output_aspect_ratio":round(provider_output_size[0]/provider_output_size[1],6),
                       "crop_aspect_ratio":round(crop.width/crop.height,6),
                       "aspect_ratio_corrected_without_stretching":abs(provider_output_size[0]/provider_output_size[1]-crop.width/crop.height)>1e-6}}
    receipt_path = output_dir / "receipt.json"; receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    receipt["artifacts"]["receipt"] = str(receipt_path)
    return receipt
