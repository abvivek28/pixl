from PIL import Image
from app.engine.deterministic import normalize_provider_output


def test_small_aspect_rounding_keeps_full_frame_without_center_crop():
    image = Image.new("RGB", (1024, 600), (80, 100, 120))
    normalized, meta = normalize_provider_output(image, (1019, 603))
    assert normalized.size == (1019, 603)
    assert meta["policy"] == "uniform_fit_with_edge_extension"
    assert meta["crop_offsets_px"] == {"left": 0, "top": 0, "right_removed": 0, "bottom_removed": 0}
    assert meta["scale_x_provider_to_crop"] == meta["scale_y_provider_to_crop"]
    assert meta["content_cropped"] is False


def test_large_aspect_mismatch_uniformly_fits_and_extends_edges_without_crop():
    image = Image.new("RGB", (1200, 600), (80, 100, 120))
    normalized, meta = normalize_provider_output(image, (800, 600))
    assert normalized.size == (800, 600)
    assert meta["policy"] == "uniform_fit_with_edge_extension"
    assert meta["crop_offsets_px"] == {"left": 0, "top": 0, "right_removed": 0, "bottom_removed": 0}
    assert meta["edge_extension_px"]["top"] > 0
    assert meta["content_cropped"] is False


def test_provider_dimensions_preserve_extreme_small_crop_aspect_ratio():
    from app.engine.deterministic import prepare_model_input
    _, out_size = prepare_model_input(Image.new("RGB", (318, 207)))
    assert out_size == (392, 256)
    assert abs((out_size[0] / out_size[1]) / (318 / 207) - 1) < 0.005


def test_provider_dimensions_respect_maximum_without_distortion():
    from app.engine.deterministic import prepare_model_input
    _, out_size = prepare_model_input(Image.new("RGB", (4000, 2000)))
    assert max(out_size) <= 1920
    assert min(out_size) >= 256
    assert abs((out_size[0] / out_size[1]) / 2.0 - 1) < 0.005
