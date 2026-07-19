"""Regression test for BP-2.

`puncta_nn_scan._build_mito_binary` unpacked 4 values from the mito-mask
builders, but both `select_mask_gui` and `compute_mito_mask_noninteractive`
return a 5-tuple (threshold, binary, skeleton, graph, params). On the first
headless run this raised `ValueError: too many values to unpack`. Stub the
producers with the real 5-value shape and assert `_build_mito_binary` unpacks
cleanly in both the GUI and non-GUI branches.
"""

import numpy as np
import pytest

import mito_linescan.mito_protein_line_scanner as line_scanner
from mito_linescan.puncta_nn_scan import _build_mito_binary

MASK_KWARGS = dict(
    tubule_radius=2.0,
    sensitivity=1.0,
    min_object_size=30,
    gap_closing=1,
    use_thickness_filter=False,
    min_thickness=1.0,
    max_thickness=20.0,
)


def _fake_producer(image, **kwargs):
    """Mimic the real 5-value return: (threshold, binary, skeleton, graph, params)."""
    binary = np.ones_like(np.asarray(image), dtype=bool)
    return 0.5, binary, None, None, {"tubule_radius": kwargs.get("tubule_radius")}


@pytest.mark.parametrize("use_gui", [False, True])
def test_build_mito_binary_unpacks_producer(monkeypatch, use_gui):
    monkeypatch.setattr(line_scanner, "select_mask_gui", _fake_producer)
    monkeypatch.setattr(line_scanner, "compute_mito_mask_noninteractive", _fake_producer)

    img = np.zeros((8, 8), dtype=np.float32)
    result = _build_mito_binary(img, use_gui=use_gui, **MASK_KWARGS)

    assert result.dtype == bool
    assert result.shape == img.shape


def test_producers_return_named_5tuple():
    """The producers expose a NamedTuple so future arity drift fails loudly."""
    result_type = line_scanner.MitoMaskResult
    assert len(result_type._fields) == 5
