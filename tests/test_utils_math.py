"""TEST-4 — utils math: normalize_image + weighted_average_scan.

Characterizes the numeric behavior of the shared helpers, including the edge
cases that silently produce wrong numbers: constant input, NaN/inf, the center
pixel, points exactly at the radius edge, and out-of-bounds clipping.
"""

import numpy as np
import pytest

from mito_linescan.utils import normalize_image, weighted_average_scan


# ---- normalize_image ------------------------------------------------------

def test_normalize_basic_min_max():
    out = normalize_image(np.array([0.0, 5.0, 10.0]))
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_normalize_constant_returns_zeros():
    out = normalize_image(np.full((3, 3), 7.0))
    assert np.array_equal(out, np.zeros((3, 3)))


def test_normalize_nan_propagates():
    # NaN is not handled specially: min/max become NaN, output is all NaN.
    out = normalize_image(np.array([0.0, np.nan, 10.0]))
    assert np.isnan(out).all()


def test_normalize_inf_characterization():
    # With +inf present, span is inf: finite entries map to 0.0 and inf -> nan.
    out = normalize_image(np.array([0.0, 1.0, np.inf]))
    assert out[0] == 0.0 and out[1] == 0.0
    assert np.isnan(out[2])


# ---- weighted_average_scan ------------------------------------------------

def test_was_constant_image_returns_constant():
    img = np.full((5, 5), 3.0)
    assert weighted_average_scan(img, 2, 2, radius=2) == pytest.approx(3.0)


def test_was_center_pixel_dominates():
    # 3x3: bright center (w=1 at dist 0), 4 edge neighbors (w=0.5 at dist 1),
    # corners excluded (dist sqrt2 > 1). => 9*1 / (1 + 4*0.5) = 3.0
    img = np.zeros((3, 3))
    img[1, 1] = 9.0
    assert weighted_average_scan(img, 1, 1, radius=1) == pytest.approx(3.0)


def test_was_radius_edge_inclusive():
    # A neighbor at distance exactly == radius must be included (dist <= radius).
    img = np.zeros((3, 3))
    img[1, 2] = 8.0  # distance 1 from center (1,1)
    # center=0 (w=1), the edge pixel=8 (w=0.5), other edges=0 => 4 / 3
    val = weighted_average_scan(img, 1, 1, radius=1)
    assert val == pytest.approx((8.0 * 0.5) / (1.0 + 4 * 0.5))


def test_was_out_of_bounds_clipping():
    # Center at a corner with a radius larger than the image must clip the
    # sampling window instead of indexing out of bounds.
    img = np.ones((3, 3))
    val = weighted_average_scan(img, 0, 0, radius=10)
    assert val == pytest.approx(1.0)  # all-ones image, whatever the window


def test_was_empty_window_returns_zero():
    # No pixel within radius (radius 0 still samples the center; use negative
    # radius to force an empty window) -> defined 0.0 fallback.
    img = np.ones((3, 3))
    assert weighted_average_scan(img, 1, 1, radius=-1) == 0.0


def test_was_nan_propagates():
    img = np.zeros((3, 3))
    img[1, 1] = np.nan
    assert np.isnan(weighted_average_scan(img, 1, 1, radius=1))
