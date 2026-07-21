"""Regression test for SCI-1 (highest-stakes).

`analyze_omm_scans.visualize_intensity_profiles` smoothed the distance axis
(`distances = smooth_profile(distances, ...)`) and then used it as the `xp`
sample points for `np.interp`. Moving-average smoothing shrinks the endpoints,
so `xp` was no longer monotonic and no longer equal to the true distances,
silently corrupting every interpolated distance the tool reports.

This pins the fix: the `xp` handed to `np.interp` must be the raw distances,
strictly increasing.
"""

import pickle

import numpy as np

import mito_linescan.analyze_omm_scans as aoms


def _write_pkl(path, distances):
    n = len(distances)
    intens = np.linspace(0.0, 1.0, n)
    detailed = [{
        "mito_intensities": intens,
        "scan_intensities": intens,
        "mask_intensities": intens,
        "skeleton_point": (0, 0),
        "normal_line_points": np.zeros((n, 2)),
        "normal_distances": np.asarray(distances, dtype=float),
    }]
    with open(path, "wb") as f:
        pickle.dump({"detailed_data": detailed, "mito_id": 1,
                     "image_name": "img"}, f)


def test_interp_xp_is_raw_monotonic_distances(monkeypatch, tmp_path):
    raw = np.linspace(-5.0, 5.0, 11)  # strictly increasing, step 1.0
    pkl = tmp_path / "pt_detailed.pkl"
    _write_pkl(pkl, raw)

    real_interp = np.interp
    captured = {}

    def capturing_interp(x, xp, fp, *args, **kwargs):
        captured.setdefault("xp", np.asarray(xp).copy())
        return real_interp(x, xp, fp, *args, **kwargs)

    monkeypatch.setattr(aoms.np, "interp", capturing_interp)

    try:
        aoms.visualize_intensity_profiles(str(pkl), output_dir=str(tmp_path))
    except Exception:
        # Downstream peak/plot logic is irrelevant here; we only assert on the
        # xp captured at the first np.interp call inside the loop.
        pass

    assert "xp" in captured, "np.interp was never called"
    xp = captured["xp"]
    assert np.all(np.diff(xp) > 0), f"xp into np.interp is not strictly monotonic: {xp}"
    assert np.allclose(xp, raw), "xp into np.interp must equal the raw distances"
