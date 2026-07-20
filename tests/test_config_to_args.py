"""TEST-6 — config_to_args serialization branches.

Pins the behavior of each branch in `config_to_args`: boolean True/False,
None-skip, and scalar str/int/float. Also characterizes the known
list-mis-serialization limitation so a future fix is a deliberate, visible
change rather than a silent one.
"""

from mito_linescan.mito_protein_localization import config_to_args


def test_bool_true_emits_bare_flag():
    assert config_to_args({"use_gui": True}) == ["--use-gui"]


def test_bool_false_emits_no_flag():
    assert config_to_args({"use_gui": False}) == ["--no-use-gui"]


def test_none_is_skipped():
    assert config_to_args({"mask_channel": None}) == []


def test_underscores_become_hyphens():
    assert config_to_args({"min_path_length": 30}) == ["--min-path-length", "30"]


def test_str_int_float_scalars():
    args = config_to_args({"run_name": "run1", "scan_width": 4, "sensitivity": 1.5})
    assert args == [
        "--run-name", "run1",
        "--scan-width", "4",
        "--sensitivity", "1.5",
    ]


def test_ordering_is_preserved():
    args = config_to_args({"a": 1, "b": True, "c": "x"})
    assert args == ["--a", "1", "--b", "--c", "x"]


def test_list_value_is_mis_serialized_known_limitation():
    # KNOWN LIMITATION: a list value is str()'d into a single token
    # ("[1, 2, 3]") rather than repeated as multiple values. No config section
    # currently uses list values; this characterizes the branch so any future
    # fix is intentional. See PHASE1_REPORT / backlog.
    assert config_to_args({"channels": [1, 2, 3]}) == ["--channels", "[1, 2, 3]"]
