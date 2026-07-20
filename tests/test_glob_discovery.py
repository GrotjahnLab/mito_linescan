"""TEST-7 — glob discovery order-independence.

Filesystem glob order is not guaranteed. Workflows that pool per-file results
must process files in a deterministic (sorted) order so outputs are
reproducible regardless of the order the OS hands back. This drives
plot_peak_distance_analysis with glob returning a scrambled list and asserts
the files are read in sorted order.
"""

import numpy as np
import pandas as pd
import pytest

import mito_linescan.plot_peak_distance_analysis as ppda

# A clean two-peak signal so every "file" yields a consecutive distance and the
# run reaches completion instead of raising "no pairs".
_D = np.arange(21, dtype=float)
_SI = np.array(
    [0, 0, 0, 0.2, 0.6, 1.0, 0.6, 0.2, 0, 0, 0,
     0, 0, 0.2, 0.6, 1.0, 0.6, 0.2, 0, 0, 0],
    dtype=float,
)


def test_discovery_is_sorted_regardless_of_glob_order(monkeypatch, tmp_path):
    files = [
        str(tmp_path / "imgA_mito_2.csv"),
        str(tmp_path / "imgA_mito_10.csv"),
        str(tmp_path / "imgB_mito_1.csv"),
    ]
    scrambled = [files[2], files[0], files[1]]

    monkeypatch.setattr(ppda.glob, "glob", lambda *a, **k: list(scrambled))

    read_order = []

    def fake_read_csv(path, *a, **k):
        read_order.append(path)
        return pd.DataFrame({"Distance": _D, "Scan_Intensity": _SI})

    monkeypatch.setattr(ppda.pd, "read_csv", fake_read_csv)

    ppda.main.callback(
        csv_directory=str(tmp_path),
        input_pattern="*_mito_*.csv",
        output_directory=str(tmp_path),
        peak_min_intensity=0.3,
        peak_min_prominence=0.1,
        peak_prominence_wlen=10,
        bin_width=5.0,
        max_distance=0.0,
        recursive=False,
    )

    assert read_order == sorted(files)
