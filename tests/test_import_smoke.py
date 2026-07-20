"""Import-smoke regression tests.

Covers DEP-2/BP-11 (STED must import without `mrcfile` installed) and, per
DEBT-3/BP-10, that every workflow module imports without a NameError.
"""

import importlib
import sys

import pytest

WORKFLOW_MODULES = [
    "mito_linescan.mito_protein_localization",
    "mito_linescan.mito_mask",
    "mito_linescan.mito_mask_refine",
    "mito_linescan.mito_protein_line_scanner",
    "mito_linescan.mito_protein_omm_normal_scanner",
    "mito_linescan.analyze_omm_scans",
    "mito_linescan.plot_peak_distance_analysis",
    "mito_linescan.puncta_nn_scan",
    "mito_linescan.sted_colocalization_3d",
]


@pytest.mark.parametrize("module_name", WORKFLOW_MODULES)
def test_workflow_module_imports(module_name):
    """Every workflow module imports cleanly (no NameError, no missing dep)."""
    assert importlib.import_module(module_name) is not None


def test_sted_imports_without_mrcfile(monkeypatch):
    """DEP-2: importing the STED module must not require `mrcfile` at import
    time. Simulate an environment where `mrcfile` is not installed (as in the
    stock environment.yml) by making `import mrcfile` raise ImportError, then
    force a fresh import of the module."""
    monkeypatch.setitem(sys.modules, "mrcfile", None)  # -> ImportError on import
    monkeypatch.delitem(
        sys.modules, "mito_linescan.sted_colocalization_3d", raising=False
    )
    module = importlib.import_module("mito_linescan.sted_colocalization_3d")
    assert module is not None
