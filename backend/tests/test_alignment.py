import numpy as np
from PIL import Image, ImageDraw

from app.engine.alignment import align_generated_image, _transform


def _textured_image(size=(180, 160)):
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack([
        (xx * 3 + yy * 2) % 256,
        (xx * 2 + yy * 5) % 256,
        (xx * 7 + yy) % 256,
    ], axis=-1).astype(np.uint8)
    image = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 20, 65, 115), outline="white", width=4)
    draw.ellipse((90, 30, 150, 100), outline="yellow", width=5)
    draw.line((10, 140, 165, 8), fill="cyan", width=3)
    return image


def test_alignment_recovers_small_translation_and_zoom():
    original = _textured_image()
    generated = _transform(original, 1.03, 3.0, -2.0)
    result = align_generated_image(
        original, generated,
        selection_box=(65, 55, 115, 105),
        context_margin_px=15,
        safety_margin_px=5,
    )
    assert result.candidates_evaluated > 1
    assert result.applied
    assert abs(result.scale - (1 / 1.03)) < 0.025
    assert abs(result.translation_x + 3.0) <= 3.0
    assert abs(result.translation_y - 2.0) <= 3.0


def test_alignment_keeps_identity_when_context_is_uninformative():
    original = Image.new("RGB", (100, 100), (120, 120, 120))
    generated = Image.new("RGB", (100, 100), (120, 120, 120))
    result = align_generated_image(
        original, generated,
        selection_box=(30, 30, 70, 70),
    )
    assert not result.applied
    assert result.scale == 1.0
    assert result.translation_x == 0.0
    assert result.translation_y == 0.0


def test_freehand_selection_mask_is_fixed_and_supported():
    original = _textured_image()
    generated = _transform(original, 1.02, -2.0, 2.0)
    polygon = [[70, 65], [110, 60], [118, 95], [82, 105]]
    result = align_generated_image(
        original, generated,
        selection_box=(65, 55, 125, 110),
        points_local=polygon,
        context_margin_px=15,
        safety_margin_px=5,
    )
    # The function returns only the transformed generated image and transform;
    # it has no mutation path for the selection polygon or mask.
    assert result.candidates_evaluated > 1
    assert result.image.size == original.size


def test_alignment_phase_correlation_offsets_are_reported_in_crop_pixels():
    # A larger crop forces the 160px scoring downscale. The recovered movement
    # must still be expressed in original crop pixels, not downscaled pixels.
    original = _textured_image((420, 360))
    generated = _transform(original, 1.0, 6.0, -4.0)
    result = align_generated_image(
        original, generated,
        selection_box=(150, 125, 260, 235),
        context_margin_px=40,
        safety_margin_px=10,
        max_translation_px=30,
    )
    assert result.search_config is not None
    small = result.search_config["phase_correlation_small_px"]
    crop = result.search_config["phase_correlation_crop_px"]
    assert abs(crop[0]) > abs(small[0]) * 1.5 or abs(crop[1]) > abs(small[1]) * 1.5


def test_alignment_reports_independent_validation():
    original = _textured_image((220, 200))
    generated = _transform(original, 1.025, 3.0, -2.0)
    result = align_generated_image(
        original, generated,
        selection_box=(80, 65, 140, 135),
        context_margin_px=40,
        safety_margin_px=10,
    )
    assert result.registration_context_pixels > 0
    assert result.validation_context_pixels > 0
    assert result.context_sufficient
    assert result.search_config["validation_metric"].startswith("held-out")


def test_synthetic_known_transform_error_is_bounded():
    original = _textured_image((240, 210))
    # The generated image contains the inverse of the desired correction.
    generated = _transform(original, 1.02, 3.0, -2.0)
    result = align_generated_image(
        original, generated,
        selection_box=(85, 65, 155, 145),
        context_margin_px=40,
        safety_margin_px=10,
        max_translation_px=30,
        minimum_improvement_percent=0.25,
    )
    assert result.applied
    assert abs(result.scale - (1 / 1.02)) < 0.025
    assert abs(result.translation_x + 3.0) < 2.5
    assert abs(result.translation_y - 2.0) < 2.5


def test_alignment_recovers_known_transform_with_lighting_change_and_noise():
    original = _textured_image((260, 220))
    generated = _transform(original, 1.03, 3.0, -2.0)
    arr = np.asarray(generated).astype(np.float32)
    rng = np.random.default_rng(42)
    arr = arr * 1.025 + 2.0 + rng.normal(0.0, 1.5, arr.shape)
    generated = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    result = align_generated_image(
        original, generated,
        selection_box=(90, 70, 170, 150),
        context_margin_px=40,
        safety_margin_px=10,
        max_translation_px=30,
    )
    assert result.applied
    assert abs(result.scale - (1 / 1.03)) < 0.015
    assert abs(result.translation_x + (3 / 1.03)) < 1.0
    assert abs(result.translation_y - (2 / 1.03)) < 1.0


def test_insufficient_stable_context_skips_search_and_keeps_pipeline_identity():
    original = _textured_image((100, 100))
    generated = _transform(original, 1.04, 2.0, -1.0)
    result = align_generated_image(
        original, generated,
        selection_box=(8, 8, 92, 92),
        context_margin_px=40,
        registration_band_px=8,
    )
    assert not result.applied
    assert result.scale == 1.0
    assert result.translation_x == 0.0
    assert result.translation_y == 0.0
    assert result.candidates_evaluated == 0
    assert "Insufficient stable context" in result.reason
