"""TEST-2 — CLI flag parity (grows the Phase-1 DEBT-1/BP-1 seed).

`config_to_args` translates a YAML config section into CLI flags fed to the
target workflow's click command. If a section emits a flag the command doesn't
define, the headless run dies with "No such option". This pins the invariant
for **every** section that goes through `config_to_args` (7 of them), against
the shipped example config, so config drift like the DEBT-1 `--use-gui`
mismatch fails the suite.
"""

from pathlib import Path

import importlib
import yaml
import pytest

from mito_linescan.mito_protein_localization import config_to_args

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
CONFIG_PATH = EXAMPLES_DIR / "config.yml"

# config section -> module whose `main` click command consumes config_to_args.
# `analyze_omm_scans` is intentionally absent: its dispatcher reads keys
# directly and calls process_directory, it does not go through config_to_args.
SECTION_TO_MODULE = {
    "draw_mask": "mito_linescan.mito_mask",
    "refine_mask": "mito_linescan.mito_mask_refine",
    "omm_normal_scan": "mito_linescan.mito_protein_omm_normal_scanner",
    "network_line_scan": "mito_linescan.mito_protein_line_scanner",
    "puncta_nn_scan": "mito_linescan.puncta_nn_scan",
    "sted_colocalization_3d": "mito_linescan.sted_colocalization_3d",
    "plot_peak_distance_analysis": "mito_linescan.plot_peak_distance_analysis",
}


def _valid_flags(module_name):
    module = importlib.import_module(module_name)
    flags = set()
    for param in module.main.params:
        flags.update(param.opts)
        flags.update(param.secondary_opts)
    return flags


def _cases():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    cases = []
    for section, module_name in SECTION_TO_MODULE.items():
        if section not in config:
            continue
        for tok in config_to_args(config[section]):
            if tok.startswith("--"):
                cases.append((section, module_name, tok))
    return cases


def test_all_seven_config_sections_present_in_example():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    missing = [s for s in SECTION_TO_MODULE if s not in config]
    assert not missing, f"example config is missing sections: {missing}"


@pytest.mark.parametrize(
    "section,module_name,flag",
    _cases(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_config_flag_exists_on_command(section, module_name, flag):
    assert flag in _valid_flags(module_name), (
        f"config section {section!r} emitted {flag!r}, not an option on "
        f"{module_name}.main"
    )
