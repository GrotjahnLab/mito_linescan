"""Regression test for DEBT-3 / BP-10.

`select_threshold` and `select_threshold_gui` were dead code (superseded by
`select_mask_gui`); `select_threshold_gui` also referenced an undefined
`mito_cmap`, a latent NameError. They must be gone. The import-smoke in
test_import_smoke.py separately guards that the module still imports cleanly.
"""

import mito_linescan.mito_protein_line_scanner as line_scanner


def test_dead_threshold_helpers_removed():
    assert not hasattr(line_scanner, "select_threshold")
    assert not hasattr(line_scanner, "select_threshold_gui")


def test_replacement_still_present():
    # The live replacement stays.
    assert hasattr(line_scanner, "select_mask_gui")
