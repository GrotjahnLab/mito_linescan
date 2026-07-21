"""TEST-5 — peak detection + consecutive-distance math.

Unit-tests the two pure helpers in plot_peak_distance_analysis:
`find_filtered_peaks` (height + prominence filtering, edge cases) and
`consecutive_distances` (peak-to-peak gaps, including a non-monotonic Distance
axis, which the function must sort before differencing).
"""

import numpy as np

from mito_linescan.plot_peak_distance_analysis import (
    find_filtered_peaks,
    consecutive_distances,
)


# ---- find_filtered_peaks --------------------------------------------------

def test_finds_two_clear_peaks():
    y = [0, 0, 0, 0.6, 1.0, 0.6, 0, 0, 0, 0.6, 1.0, 0.6, 0, 0, 0]
    peaks = find_filtered_peaks(y, min_intensity=0.3, min_prominence=0.1, wlen=10)
    assert list(peaks) == [4, 10]


def test_peak_below_min_intensity_excluded():
    y = [0, 0, 0.2, 0, 0]  # single bump, height 0.2 < 0.3
    peaks = find_filtered_peaks(y, min_intensity=0.3, min_prominence=0.1, wlen=10)
    assert peaks.size == 0


def test_peak_below_min_prominence_excluded():
    y = [0.4, 0.45, 0.4]  # height ok (>=0.3) but prominence 0.05 < 0.1
    peaks = find_filtered_peaks(y, min_intensity=0.3, min_prominence=0.1, wlen=10)
    assert peaks.size == 0


def test_too_short_returns_empty():
    assert find_filtered_peaks([1.0, 2.0], 0.3, 0.1, 10).size == 0


def test_monotonic_has_no_peaks():
    assert find_filtered_peaks([0, 1, 2, 3, 4], 0.3, 0.1, 10).size == 0


# ---- consecutive_distances ------------------------------------------------

def test_consecutive_distances_monotonic():
    d = [0, 2, 4, 6, 8, 10]
    peaks = np.array([1, 3, 5])  # distances 2, 6, 10
    assert np.allclose(consecutive_distances(d, peaks), [4.0, 4.0])


def test_consecutive_distances_needs_two_peaks():
    assert consecutive_distances([0, 1, 2], np.array([1])).size == 0
    assert consecutive_distances([0, 1, 2], np.array([], dtype=int)).size == 0


def test_consecutive_distances_non_monotonic_path():
    # Distance axis out of order: the function must sort by distance before
    # differencing, so gaps reflect true spatial spacing (0,5,10 -> 5,5),
    # not raw index order.
    d = [0.0, 10.0, 5.0]
    peaks = np.array([0, 1, 2])
    assert np.allclose(consecutive_distances(d, peaks), [5.0, 5.0])
