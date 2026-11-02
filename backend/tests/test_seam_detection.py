import numpy as np
from PIL import Image, ImageDraw

from app.engine.seam_detection import polygon_mask, evaluate_seam, adaptive_feather_alpha


def test_polygon_mask_tracks_freehand_selection_not_its_bounding_box():
    mask = polygon_mask((40, 40), [[5, 5], [30, 5], [10, 30]], (0, 0, 40, 40))
    assert mask[8, 8]
    assert not mask[28, 28]


def test_seam_diagnostics_detect_new_boundary_discontinuity():
    original = np.full((50, 50, 3), 120, dtype=np.uint8)
    composite = original.copy()
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True
    composite[mask] = 220
    metrics = evaluate_seam(original, composite, mask)
    assert metrics["boundary_pixels"] > 0
    assert metrics["excess_boundary_step"] > 50
    assert metrics["p95_excess_boundary_step"] > 50


def test_adaptive_feather_is_bounded_and_smooth():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:25, 5:25] = True
    alpha = adaptive_feather_alpha(mask, seam_metrics={"excess_boundary_step": 80})
    assert alpha.shape == mask.shape
    assert alpha.min() >= 0 and alpha.max() <= 1
    assert alpha[15, 15] > 0.99
    assert alpha[4, 15] == 0.0
    assert alpha[5, 15] > 0.0


def test_inward_feather_never_changes_pixels_outside_polygon():
    mask = np.zeros((40, 40), dtype=bool)
    mask[8:32, 8:32] = True
    alpha = adaptive_feather_alpha(mask, base_radius=2.0)
    assert np.all(alpha[~mask] == 0.0)
    assert np.all(alpha[mask] >= 0.0)
    assert alpha[20, 20] == 1.0
