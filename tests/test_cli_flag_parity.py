"""Regression test for DEBT-1 / BP-1.

`config_to_args` translates a YAML config section into CLI flags that are then
fed to the target workflow's click command. If the config emits a flag the
command doesn't define, the headless run dies with "No such option". This test
pins the invariant: every flag `config_to_args` emits for the
`network_line_scan` section must be a real option on the target command.
"""

from pathlib import Path

import yaml
import pytest

from mito_linescan.mito_protein_localization import config_to_args
from mito_linescan.mito_protein_line_scanner import main as network_line_scan_cmd

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _valid_flags(command):
    """All flag spellings a click command accepts (primary + secondary)."""
    flags = set()
    for param in command.params:
        flags.update(param.opts)
        flags.update(param.secondary_opts)
    return flags


def _emitted_flags():
    config = yaml.safe_load((EXAMPLES_DIR / "config_headless.yaml").read_text())
    args = config_to_args(config["network_line_scan"])
    return [tok for tok in args if tok.startswith("--")]


@pytest.mark.parametrize("flag", _emitted_flags())
def test_network_line_scan_flag_exists(flag):
    assert flag in _valid_flags(network_line_scan_cmd), (
        f"config_to_args emitted {flag!r}, which is not an option on "
        f"network_line_scan's command"
    )
