from pathlib import Path
import random
import time

import numpy as np
import pytest
from PIL import Image

from app.engine.deterministic import Box, ingest_image, run_fake_edit, to_oklab, from_oklab

@pytest.fixture
def patterned_image(tmp_path):
    rng = np.random.default_rng(12)
    arr = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    path = tmp_path / "source.png"
    Image.fromarray(arr, "RGB").save(path)
    master = tmp_path / "master.png"
    ingest_image(path, master)
    return master

def test_oklab_white_is_near_expected():
    lab = to_oklab(np.array([[[255, 255, 255]]], dtype=np.uint8))[0, 0]
    assert abs(float(lab[0]) - 1.0) < 0.002
    assert abs(float(lab[1])) < 0.002
    assert abs(float(lab[2])) < 0.002

def test_oklab_round_trip():
    sample = np.array([[[0, 0, 0], [255, 255, 255], [220, 90, 30], [12, 100, 230]]], dtype=np.uint8)
    recovered = from_oklab(to_oklab(sample))
    assert np.max(np.abs(recovered.astype(int) - sample.astype(int))) <= 2

@pytest.mark.parametrize("box", [
    Box(0, 0, 1, 1), Box(127, 95, 128, 96), Box(0, 20, 5, 30),
    Box(30, 0, 45, 10), Box(100, 50, 128, 96),
    *[Box(x, y, min(128, x + 12), min(96, y + 11))
      for x, y in [(4, 4), (16, 7), (37, 15), (56, 19), (80, 9),
                   (8, 43), (28, 36), (45, 50), (66, 60), (88, 70),
                   (110, 20), (120, 40), (3, 80), (30, 85), (70, 25)]]
])
def test_outside_selected_box_is_exactly_unchanged(patterned_image, tmp_path, box):
    original = np.asarray(Image.open(patterned_image).convert("RGB"))
    receipt = run_fake_edit(patterned_image, box, tmp_path / f"edit-{box.x1}-{box.y1}", feather_px=3)
    result = np.asarray(Image.open(receipt["artifacts"]["result"]).convert("RGB"))
    outside = np.ones(original.shape[:2], dtype=bool)
    x1, y1 = max(0, box.x1), max(0, box.y1)
    x2, y2 = min(original.shape[1], box.x2), min(original.shape[0], box.y2)
    outside[y1:y2, x1:x2] = False
    assert np.array_equal(original[outside], result[outside])
    assert receipt["metrics"]["outside_changed_pixels"] == 0
    assert Path(receipt["artifacts"]["heatmap"]).exists()
    assert Path(receipt["artifacts"]["receipt"]).exists()

def test_exif_rotation_applied_once_and_saved_as_rgb_png(tmp_path):
    # EXIF orientation 6 means rotate 90 degrees clockwise for display.
    image = Image.new("RGB", (40, 20), (230, 40, 20))
    exif = Image.Exif()
    exif[274] = 6
    source = tmp_path / "oriented.jpg"
    image.save(source, format="JPEG", exif=exif)
    master = tmp_path / "master.png"
    info = ingest_image(source, master)
    with Image.open(master) as result:
        assert result.size == (20, 40)
        assert result.mode == "RGB"
        assert result.format == "PNG"
    assert info["sha256"]
    assert info["width"] == 20 and info["height"] == 40

def test_jpeg_input_becomes_png_result(tmp_path):
    source = tmp_path / "input.jpg"
    Image.new("RGB", (80, 60), (20, 100, 220)).save(source, "JPEG")
    master = tmp_path / "master.png"
    ingest_image(source, master)
    receipt = run_fake_edit(master, Box(20, 15, 50, 45), tmp_path / "output")
    with Image.open(receipt["artifacts"]["result"]) as result:
        assert result.format == "PNG"
        assert result.size == (80, 60)

def test_expanded_box_clamps_at_image_edges(patterned_image):
    from app.engine.deterministic import expand_box
    assert expand_box(Box(0, 0, 10, 10), 128, 96) == Box(0, 0, 18, 18)
    assert expand_box(Box(120, 90, 128, 96), 128, 96) == Box(112, 82, 128, 96)

def test_12mp_image_runs_in_reasonable_time(tmp_path):
    # 4000x3000 = 12 MP. Use a small selected region to keep this smoke test practical.
    rng = np.random.default_rng(5)
    arr = rng.integers(0, 256, size=(3000, 4000, 3), dtype=np.uint8)
    master = tmp_path / "12mp.png"
    Image.fromarray(arr, "RGB").save(master, optimize=False)
    started = time.perf_counter()
    receipt = run_fake_edit(master, Box(1800, 1300, 2200, 1700), tmp_path / "12mp-result")
    elapsed = time.perf_counter() - started
    assert receipt["metrics"]["outside_changed_pixels"] == 0
    assert Path(receipt["artifacts"]["heatmap"]).stat().st_size > 0
    assert elapsed < 45
