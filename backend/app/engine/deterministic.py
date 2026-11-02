from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import math
import time

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFilter, ImageOps

OKLAB_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)
OKLAB_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=np.float32)

@dataclass(frozen=True)
class Box:
    """Pixel-space half-open rectangle: x1/y1 inclusive, x2/y2 exclusive."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def file_sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


def _convert_to_srgb(image: Image.Image) -> Image.Image:
    """Convert embedded ICC profiles to sRGB when possible, then normalize to RGB8."""
    image = ImageOps.exif_transpose(image)
    icc = image.info.get("icc_profile")
    if icc:
        try:
            src_profile = ImageCms.ImageCmsProfile(__import__("io").BytesIO(icc))
            dst_profile = ImageCms.createProfile("sRGB")
            image = ImageCms.profileToProfile(image, src_profile, dst_profile, outputMode="RGB")
        except Exception:
            # Some images contain broken ICC profiles; conversion to RGB is a safe fallback.
            image = image.convert("RGB")
    else:
        image = image.convert("RGB")
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def ingest_image(source: bytes | str | Path | Image.Image, output_path: str | Path) -> dict[str, Any]:
    """Apply EXIF orientation once, normalize to RGB8/sRGB where possible, and save master PNG."""
    if isinstance(source, Image.Image):
        image = source.copy()
        original_bytes = b""
        image = _convert_to_srgb(image)
    elif isinstance(source, bytes):
        import io
        original_bytes = source
        with Image.open(io.BytesIO(source)) as opened:
            image = _convert_to_srgb(opened.copy())
    else:
        path = Path(source)
        original_bytes = path.read_bytes()
        with Image.open(path) as opened:
            image = _convert_to_srgb(opened.copy())

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=False)
    master_bytes = output.read_bytes()
    return {
        "path": str(output),
        "sha256": file_sha256(master_bytes),
        "original_sha256": file_sha256(original_bytes) if original_bytes else None,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": "PNG",
        "size_bytes": len(master_bytes),
    }


def expand_box(box: Box, image_width: int, image_height: int,
               expansion_fraction: float = 0.10, min_margin: int = 8) -> Box:
    """Expand by 10% of selected width/height on each side, at least min_margin, then clamp."""
    x1, y1 = max(0, box.x1), max(0, box.y1)
    x2, y2 = min(image_width, box.x2), min(image_height, box.y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Selected box must have positive area and intersect the image.")
    mx = max(min_margin, int(math.ceil((x2 - x1) * expansion_fraction)))
    my = max(min_margin, int(math.ceil((y2 - y1) * expansion_fraction)))
    return Box(max(0, x1 - mx), max(0, y1 - my),
               min(image_width, x2 + mx), min(image_height, y2 + my))


def prepare_model_input(crop: Image.Image, max_side: int = 512,
                        preferred_side: int = 504) -> tuple[Image.Image, tuple[int, int]]:
    """Downscale crop to <= max_side and propose provider output dimensions matching crop size.

    The output dimensions are rounded to the nearest multiple of 8 (required by
    most diffusion-model providers) and clamped to [256, 1920].  By matching the
    crop's native resolution instead of a fixed preferred_side, the provider
    result can be composited back without a large rescale that would distort
    spatial proportions.
    """
    source = crop.convert("RGB")
    scale = min(1.0, max_side / max(source.size))
    model_size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
    model_input = source.resize(model_size, Image.Resampling.LANCZOS) if model_size != source.size else source.copy()

    # Propose output dimensions that match the crop's native size so the
    # provider result aligns with the compositing target without a large
    # rescale.  Round to the nearest multiple of 8 for provider compatibility,
    # clamped to the commonly supported [256, 1920] range.
    def _round_to_8(v: int) -> int:
        return max(256, min(1920, int(round(v / 8)) * 8))

    out_size = (_round_to_8(source.width), _round_to_8(source.height))
    return model_input, out_size


def to_oklab(rgb_u8: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb_u8, dtype=np.float32) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.cbrt(linear @ OKLAB_M1.T) @ OKLAB_M2.T


def from_oklab(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    rgb_linear = np.stack([
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ], axis=-1)
    rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
    srgb = np.where(rgb_linear <= 0.0031308, rgb_linear * 12.92,
                    1.055 * np.power(rgb_linear, 1 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def delta_e(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    return np.linalg.norm(to_oklab(a_u8) - to_oklab(b_u8), axis=-1)


def _ring_mask(expanded: Box, selected: Box) -> np.ndarray:
    mask = np.ones((expanded.height, expanded.width), dtype=bool)
    x1, y1 = selected.x1 - expanded.x1, selected.y1 - expanded.y1
    x2, y2 = selected.x2 - expanded.x1, selected.y2 - expanded.y1
    mask[y1:y2, x1:x2] = False
    return mask


def _hue_shift_fake(image: Image.Image, degrees: float = 24.0) -> Image.Image:
    """Deterministic vectorized fake model: shift HSV hue while retaining texture."""
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    shift = int(round((degrees / 360.0) * 255.0))
    hsv[..., 0] = ((hsv[..., 0].astype(np.uint16) + shift) % 256).astype(np.uint8)
    return Image.fromarray(hsv, "HSV").convert("RGB")


def _laplacian_variance(arr: np.ndarray) -> float:
    gray = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)
    if min(gray.shape) < 3:
        return 0.0
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(np.var(lap))


def _feather_mask(width: int, height: int, feather: int,
                  touches_left: bool, touches_top: bool,
                  touches_right: bool, touches_bottom: bool) -> np.ndarray:
    """Feather only inside B; an image-border side stays fully opaque."""
    yy, xx = np.mgrid[0:height, 0:width]
    distances = []
    if not touches_left: distances.append(xx + 1)
    if not touches_right: distances.append(width - xx)
    if not touches_top: distances.append(yy + 1)
    if not touches_bottom: distances.append(height - yy)
    if feather <= 0 or not distances:
        return np.ones((height, width), dtype=np.float32)
    d = np.minimum.reduce(distances).astype(np.float32)
    return np.clip(d / float(feather), 0.0, 1.0)


def _median_oklab_offset(original_ring: np.ndarray, generated_ring: np.ndarray) -> np.ndarray:
    if original_ring.size == 0 or generated_ring.size == 0:
        return np.zeros(3, dtype=np.float32)
    diff = to_oklab(original_ring) - to_oklab(generated_ring)
    return np.median(diff.reshape(-1, 3), axis=0).astype(np.float32)


def run_fake_edit(
    master_path: str | Path,
    selected_box: Box,
    output_dir: str | Path,
    *,
    box_source: str = "manual",
    provider: str = "fake",
    model: str = "hue-shift-v1",
    seed: int | None = 0,
    prompt: str = "Deterministic hue-shift fake edit",
    feather_px: int = 4,
    min_margin: int = 8,
) -> dict[str, Any]:
    """Run a deterministic crop edit, color-match the ring, composite only B, and write a receipt."""
    start_time = time.perf_counter()
    master_path, output_dir = Path(master_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(master_path) as opened:
        original = opened.convert("RGB").copy()
    original_arr = np.asarray(original, dtype=np.uint8)
    h, w = original_arr.shape[:2]
    B = Box(max(0, selected_box.x1), max(0, selected_box.y1),
            min(w, selected_box.x2), min(h, selected_box.y2))
    if B.width <= 0 or B.height <= 0:
        raise ValueError("Selected box must have positive area inside the master image.")
    E = expand_box(B, w, h, min_margin=min_margin)
    crop = original.crop((E.x1, E.y1, E.x2, E.y2))
    model_input, proposed_out_size = prepare_model_input(crop)
    # Phase 1 uses a local deterministic fake provider. Run it on native crop for pixel-stable testing.
    generated = _hue_shift_fake(crop)
    if generated.size != crop.size:
        generated = generated.resize(crop.size, Image.Resampling.LANCZOS)
    generated_arr = np.asarray(generated, dtype=np.uint8)
    crop_arr = np.asarray(crop, dtype=np.uint8)
    ring = _ring_mask(E, B)
    offset = _median_oklab_offset(crop_arr[ring], generated_arr[ring])
    corrected_lab = to_oklab(generated_arr) + offset
    corrected = from_oklab(corrected_lab)

    bx1, by1 = B.x1 - E.x1, B.y1 - E.y1
    bx2, by2 = B.x2 - E.x1, B.y2 - E.y1
    patch = corrected[by1:by2, bx1:bx2]
    original_patch = original_arr[B.y1:B.y2, B.x1:B.x2]
    alpha = _feather_mask(
        B.width, B.height, feather_px,
        B.x1 == 0, B.y1 == 0, B.x2 == w, B.y2 == h
    )[..., None]
    # Round-to-nearest makes the composite deterministic.
    composite_patch = np.clip(np.rint(original_patch * (1.0 - alpha) + patch * alpha), 0, 255).astype(np.uint8)
    result_arr = original_arr.copy()
    result_arr[B.y1:B.y2, B.x1:B.x2] = composite_patch

    # Exact outside-B invariant.
    outside = np.ones((h, w), dtype=bool)
    outside[B.y1:B.y2, B.x1:B.x2] = False
    outside_changed = int(np.count_nonzero(np.any(result_arr != original_arr, axis=2) & outside))

    full_delta = delta_e(original_arr, result_arr)
    b_delta = full_delta[B.y1:B.y2, B.x1:B.x2]
    changed_mask = b_delta > 0.05
    percent_changed = float(changed_mask.mean() * 100.0)
    mean_delta = float(b_delta.mean()) if b_delta.size else 0.0

    # Ring drift measures the color-matched generated ring against the original ring.
    corrected_ring = from_oklab(to_oklab(generated_arr) + offset)
    ring_deltas = delta_e(crop_arr[ring], corrected_ring[ring]) if ring.any() else np.array([], dtype=np.float32)
    ring_drift = float(np.median(ring_deltas)) if ring_deltas.size else 0.0

    # Boundary seam ratio: boundary step in final result divided by typical local neighbor step.
    result_rgb = result_arr.astype(np.float32)
    rgb_delta = np.linalg.norm(np.diff(result_rgb, axis=1), axis=2)
    rgb_delta_v = np.linalg.norm(np.diff(result_rgb, axis=0), axis=2)
    boundary_steps = []
    if B.x1 > 0: boundary_steps.extend(np.linalg.norm(result_rgb[B.y1:B.y2, B.x1] - result_rgb[B.y1:B.y2, B.x1-1], axis=1).tolist())
    if B.x2 < w: boundary_steps.extend(np.linalg.norm(result_rgb[B.y1:B.y2, B.x2-1] - result_rgb[B.y1:B.y2, B.x2], axis=1).tolist())
    if B.y1 > 0: boundary_steps.extend(np.linalg.norm(result_rgb[B.y1, B.x1:B.x2] - result_rgb[B.y1-1, B.x1:B.x2], axis=1).tolist())
    if B.y2 < h: boundary_steps.extend(np.linalg.norm(result_rgb[B.y2-1, B.x1:B.x2] - result_rgb[B.y2, B.x1:B.x2], axis=1).tolist())
    typical = np.concatenate([rgb_delta.ravel(), rgb_delta_v.ravel()]) if rgb_delta.size and rgb_delta_v.size else np.array([1.0])
    typical_neighbor = float(np.median(typical))
    seam_ratio = float(np.median(boundary_steps) / max(typical_neighbor, 1.0)) if boundary_steps else 0.0

    patch_texture = _laplacian_variance(composite_patch)
    ring_arr = crop_arr[ring]
    # Texture variance expects spatial neighborhoods; use a bounding ring image with B area masked by original.
    ring_image = crop_arr.copy()
    ring_image[~ring] = original_patch.mean(axis=(0, 1)).astype(np.uint8)
    ring_texture = _laplacian_variance(ring_image)
    texture_ratio = float(patch_texture / max(ring_texture, 1e-6))

    result_path = output_dir / "result.png"
    Image.fromarray(result_arr, "RGB").save(result_path, format="PNG", optimize=False)

    heat = np.zeros((h, w), dtype=np.uint8)
    max_delta = max(float(full_delta.max()), 1e-8)
    heat[B.y1:B.y2, B.x1:B.x2] = np.clip(
        full_delta[B.y1:B.y2, B.x1:B.x2] / max(0.25, max_delta) * 255, 0, 255
    ).astype(np.uint8)
    # Black outside selection; orange/red inside changed pixels.
    heat_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    heat_rgb[..., 0] = heat
    heat_rgb[..., 1] = np.clip(heat.astype(np.float32) * 0.28, 0, 255).astype(np.uint8)
    heat_path = output_dir / "heatmap.png"
    Image.fromarray(heat_rgb, "RGB").save(heat_path, format="PNG", optimize=False)

    original_bytes = master_path.read_bytes()
    result_bytes = result_path.read_bytes()
    metrics = {
        "outside_changed_pixels": outside_changed,
        "percent_box_changed": percent_changed,
        "mean_delta_e_oklab": mean_delta,
        "ring_drift_median_delta_e_oklab": ring_drift,
        "seam_ratio": seam_ratio,
        "texture_ratio": texture_ratio,
        "changed_pixel_threshold_delta_e": 0.05,
        "elapsed_seconds": round(time.perf_counter() - start_time, 4),
    }
    checks = [
        {"name": "zero_changes_outside_selected_box", "pass": outside_changed == 0,
         "actual": outside_changed, "threshold": 0},
        {"name": "ring_drift", "pass": ring_drift < 0.03, "actual": ring_drift, "threshold": "< 0.03"},
        {"name": "seam_ratio", "pass": seam_ratio < 1.5, "actual": seam_ratio, "threshold": "< 1.5"},
        {"name": "texture_ratio", "pass": 0.5 <= texture_ratio <= 2.0,
         "actual": texture_ratio, "threshold": "0.5–2.0"},
    ]
    warnings = []
    if B.width < 8 or B.height < 8:
        warnings.append("Selected box is tiny; quality metrics are less reliable.")
    if not ring.any():
        warnings.append("Expanded crop has no ring pixels; ring-based color matching was skipped.")
    if box_source == "manual":
        warnings.append("Selection box was supplied manually.")
    summary = (
        f"Changed {percent_changed:.1f}% of the selected region. "
        f"Mean OKLab ΔE was {mean_delta:.4f}. "
        f"Pixels changed outside the selection: {outside_changed}. "
        f"{'All hard checks passed.' if all(c['pass'] for c in checks) else 'One or more quality checks need review.'}"
    )
    receipt = {
        "schema_version": "1.0",
        "image": {
            "master_sha256": file_sha256(original_bytes),
            "result_sha256": file_sha256(result_bytes),
            "master_size": {"width": w, "height": h, "bytes": len(original_bytes)},
            "result_size": {"width": w, "height": h, "bytes": len(result_bytes)},
            "master_format": "PNG",
            "result_format": "PNG",
        },
        "boxes": {"selected": B.as_dict(), "expanded": E.as_dict(), "coordinate_convention": "pixel half-open [x1,y1,x2,y2]"},
        "box_source": box_source,
        "provider": provider,
        "model": model,
        "seed": seed,
        "prompt": prompt,
        "metrics": metrics,
        "checks": checks,
        "warnings": warnings,
        "summary": summary,
        "artifacts": {"result": str(result_path), "heatmap": str(heat_path)},
        "model_input": {"width": model_input.width, "height": model_input.height,
                        "proposed_output_width": proposed_out_size[0], "proposed_output_height": proposed_out_size[1]},
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    receipt["artifacts"]["receipt"] = str(receipt_path)
    return receipt
