"""Regression test for DEBT-2 / DEP-8.

setup.py's console_scripts referenced `mito_protein_omm_localization`, a module
that does not exist, so `pip install` produced a broken command. Assert every
console_scripts target resolves to an importable module.
"""

import ast
import importlib
from pathlib import Path

import pytest

SETUP_PY = Path(__file__).resolve().parent.parent / "setup.py"


def _console_scripts():
    tree = ast.parse(SETUP_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "entry_points":
            return ast.literal_eval(node.value).get("console_scripts", [])
    return []


@pytest.mark.parametrize("entry", _console_scripts())
def test_console_script_module_importable(entry):
    module_path = entry.split("=", 1)[1].split(":", 1)[0]
    assert importlib.import_module(module_path) is not None
