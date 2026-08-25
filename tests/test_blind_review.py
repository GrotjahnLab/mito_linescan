"""Tests for the display-free logic of the blind-review utility.

No GUI is exercised here — only the pure functions the matplotlib shell wraps:
auto-contrast, channel/z-stack plane selection, deterministic shuffling, and
merging calls into the master genotype sheet.
"""

import numpy as np
import pandas as pd
import pytest

import mito_linescan.blind_review as br


# --- auto_contrast -------------------------------------------------------

def test_auto_contrast_known_array():
    # 0..100 inclusive; zeros are excluded, so percentiles run over 1..100.
    img = np.arange(0, 101, dtype=np.float64).reshape(1, 101)
    vmin, vmax = br.auto_contrast(img, p_low=1.0, p_high=99.5)
    nonzero = np.arange(1, 101, dtype=np.float64)
    assert vmin == pytest.approx(np.percentile(nonzero, 1.0))
    assert vmax == pytest.approx(np.percentile(nonzero, 99.5))
    assert vmin < vmax


def test_auto_contrast_uses_all_pixels_when_no_zeros():
    img = np.arange(1, 101, dtype=np.float64)
    vmin, vmax = br.auto_contrast(img, p_low=1.0, p_high=99.5)
    assert vmin == pytest.approx(np.percentile(img, 1.0))
    assert vmax == pytest.approx(np.percentile(img, 99.5))


def test_auto_contrast_constant_image():
    img = np.full((10, 10), 7.0)
    vmin, vmax = br.auto_contrast(img)
    assert vmin < vmax  # must never collapse


def test_auto_contrast_all_zero_image():
    img = np.zeros((8, 8))
    vmin, vmax = br.auto_contrast(img)
    assert vmin < vmax


# --- select_review_plane -------------------------------------------------

def test_plane_2d_unchanged():
    img = np.random.rand(50, 60)
    out = br.select_review_plane(img)
    assert out.shape == (50, 60)
    assert np.array_equal(out, img)


def test_plane_channel_first_axis_default_last():
    # (C, Y, X) with 2 channels — smallest axis is 0; default is the last channel.
    arr = np.zeros((2, 30, 40))
    arr[0] = 1.0
    arr[1] = 2.0
    out = br.select_review_plane(arr)
    assert out.shape == (30, 40)
    assert np.all(out == 2.0)  # last channel by default


def test_plane_channel_last_axis_default_last():
    # (Y, X, C) with 2 channels — smallest axis is 2; default is the last channel.
    arr = np.zeros((30, 40, 2))
    arr[..., 0] = 5.0
    arr[..., 1] = 9.0
    out = br.select_review_plane(arr)
    assert out.shape == (30, 40)
    assert np.all(out == 9.0)  # last channel by default


def test_plane_explicit_channel_index():
    # Explicit non-default index still works and is clamped when out of range.
    arr = np.zeros((3, 10, 10))
    arr[0] = 1.0
    arr[1] = 2.0
    arr[2] = 3.0
    assert np.all(br.select_review_plane(arr, channel=0) == 1.0)
    assert np.all(br.select_review_plane(arr, channel=1) == 2.0)
    assert np.all(br.select_review_plane(arr, channel=99) == 3.0)  # clamped to last


def test_plane_zstack_max_projection():
    # (Z, Y, X) with many z slices — treated as a z-series, max-projected.
    arr = np.zeros((20, 10, 10))
    arr[7, 3, 3] = 42.0
    out = br.select_review_plane(arr)
    assert out.shape == (10, 10)
    assert out[3, 3] == 42.0
    assert out.max() == 42.0


def test_plane_bad_dims_raise():
    with pytest.raises(ValueError):
        br.select_review_plane(np.zeros((2, 2, 2, 2, 2)))


# --- shuffle determinism -------------------------------------------------

FILES = [f"S9MEF_{i:02d}.tif" for i in range(1, 21)]


def test_shuffle_is_permutation():
    out = br.shuffle_files(FILES, seed=123)
    assert sorted(out) == sorted(FILES)
    assert len(out) == len(FILES)


def test_shuffle_same_seed_same_order():
    a = br.shuffle_files(FILES, br.seed_from_name("Alice"))
    b = br.shuffle_files(FILES, br.seed_from_name("Alice"))
    assert a == b


def test_shuffle_different_name_different_order():
    a = br.shuffle_files(FILES, br.seed_from_name("Alice"))
    b = br.shuffle_files(FILES, br.seed_from_name("Bob"))
    assert a != b


def test_seed_from_name_stable():
    assert br.seed_from_name("Alice") == br.seed_from_name("Alice")
    assert br.seed_from_name("Alice") != br.seed_from_name("Bob")


# --- sanitize_reviewer ---------------------------------------------------

def test_sanitize_reviewer():
    assert br.sanitize_reviewer("Jane Doe") == "jane_doe"
    assert br.sanitize_reviewer("A. B-C!!") == "a_b_c"
    assert br.sanitize_reviewer("   ") == "reviewer"


# --- merge_calls_into_master ---------------------------------------------

def _master(stems):
    return pd.DataFrame({"FILE": stems})


def test_merge_adds_column_joined_on_stem():
    master = _master(["S9MEF_01", "S9MEF_02", "S9MEF_03"])
    calls = {"S9MEF_01": "MorphologyA", "S9MEF_03": "MorphologyB"}
    merged, extras = br.merge_calls_into_master(master, calls, "alice")
    assert list(merged.columns) == ["FILE", "alice"]  # appended at right
    assert merged.loc[merged.FILE == "S9MEF_01", "alice"].item() == "MorphologyA"
    assert merged.loc[merged.FILE == "S9MEF_03", "alice"].item() == "MorphologyB"
    # unscored row -> empty
    assert merged.loc[merged.FILE == "S9MEF_02", "alice"].item() == ""
    assert extras == []


def test_merge_preserves_existing_column_order():
    master = pd.DataFrame({"FILE": ["S9MEF_01"], "bob": ["MorphologyA"], "note": ["x"]})
    calls = {"S9MEF_01": "MorphologyB"}
    merged, _ = br.merge_calls_into_master(master, calls, "alice")
    assert list(merged.columns) == ["FILE", "bob", "note", "alice"]


def test_merge_extra_files_warned_not_added_by_default():
    master = _master(["S9MEF_01"])
    calls = {"S9MEF_01": "MorphologyA", "S9MEF_99": "MorphologyB"}
    merged, extras = br.merge_calls_into_master(master, calls, "alice")
    assert extras == ["S9MEF_99"]
    assert "S9MEF_99" not in set(merged.FILE)


def test_merge_extra_files_added_with_flag():
    master = _master(["S9MEF_01"])
    calls = {"S9MEF_01": "MorphologyA", "S9MEF_99": "MorphologyB"}
    merged, extras = br.merge_calls_into_master(
        master, calls, "alice", allow_new_rows=True)
    assert extras == ["S9MEF_99"]
    assert "S9MEF_99" in set(merged.FILE)
    assert merged.loc[merged.FILE == "S9MEF_99", "alice"].item() == "MorphologyB"


def test_merge_existing_column_raises_without_overwrite():
    master = pd.DataFrame({"FILE": ["S9MEF_01"], "alice": ["MorphologyA"]})
    with pytest.raises(ValueError):
        br.merge_calls_into_master(master, {"S9MEF_01": "MorphologyB"}, "alice")


def test_merge_existing_column_overwrites_with_flag():
    master = pd.DataFrame({"FILE": ["S9MEF_01"], "alice": ["MorphologyA"]})
    merged, _ = br.merge_calls_into_master(
        master, {"S9MEF_01": "MorphologyB"}, "alice", overwrite_column=True)
    assert merged.loc[merged.FILE == "S9MEF_01", "alice"].item() == "MorphologyB"


def test_merge_missing_file_column_raises():
    with pytest.raises(ValueError):
        br.merge_calls_into_master(pd.DataFrame({"X": [1]}), {}, "alice")


# --- build_reviewer_results ----------------------------------------------

def test_build_reviewer_results():
    records = [
        {"FILE": "S9MEF_01", "call": "MorphologyA"},
        {"FILE": "S9MEF_02", "call": "MorphologyB"},
    ]
    df = br.build_reviewer_results(records, "Jane Doe")
    assert list(df.columns) == ["FILE", "Jane Doe"]
    assert df.loc[df.FILE == "S9MEF_01", "Jane Doe"].item() == "MorphologyA"
    assert df.loc[df.FILE == "S9MEF_02", "Jane Doe"].item() == "MorphologyB"


def test_build_reviewer_results_empty():
    df = br.build_reviewer_results([], "alice")
    assert list(df.columns) == ["FILE", "alice"]
    assert len(df) == 0


# --- ground-truth scoring ------------------------------------------------

def test_load_ground_truth_tab_separated(tmp_path):
    # New-style labels pass through unchanged; separator is auto-detected.
    p = tmp_path / "gt.csv"
    p.write_text("BLINDED\tGround_Truth\nS9MEF_01\tMorphologyA\nS9MEF_02\tMorphologyB\n")
    truth = br.load_ground_truth(str(p))
    assert truth == {"S9MEF_01": "MorphologyA", "S9MEF_02": "MorphologyB"}


def test_load_ground_truth_positional_fallback(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("stem,truth\nS9MEF_01,MorphologyA\nS9MEF_02,MorphologyB\n")
    truth = br.load_ground_truth(str(p))
    assert truth == {"S9MEF_01": "MorphologyA", "S9MEF_02": "MorphologyB"}


def test_load_ground_truth_legacy_wt_ko_aliased(tmp_path):
    # Legacy WT/KO sheets (any case) are mapped onto the morphology labels.
    p = tmp_path / "gt.csv"
    p.write_text("BLINDED\tGround_Truth\nS9MEF_01\tWT\nS9MEF_02\tko\n")
    truth = br.load_ground_truth(str(p))
    assert truth == {"S9MEF_01": "MorphologyA", "S9MEF_02": "MorphologyB"}


def test_normalize_call_aliases():
    assert br.normalize_call("WT") == "MorphologyA"
    assert br.normalize_call("ko") == "MorphologyB"
    assert br.normalize_call("MorphologyA") == "MorphologyA"
    assert br.normalize_call("IDK") == "IDK"


def _rec(stem, call):
    return {"FILE": stem, "call": call}


def test_score_against_truth_basic():
    records = [_rec("A", "MorphologyA"), _rec("B", "MorphologyB"), _rec("C", "MorphologyA")]
    truth = {"A": "MorphologyA", "B": "MorphologyA", "C": "MorphologyA"}
    res = br.score_against_truth(records, truth)
    assert res["n_compared"] == 3
    assert res["n_correct"] == 2  # A, C right; B wrong
    assert res["accuracy"] == pytest.approx(200.0 / 3)


def test_score_ignores_files_absent_from_truth():
    records = [_rec("A", "MorphologyA"), _rec("Z", "MorphologyB")]
    truth = {"A": "MorphologyA"}
    res = br.score_against_truth(records, truth)
    assert res["n_compared"] == 1
    assert res["n_correct"] == 1
    assert res["accuracy"] == 100.0


def test_score_idk_counts_as_incorrect():
    records = [_rec("A", "IDK"), _rec("B", "MorphologyB")]
    truth = {"A": "MorphologyA", "B": "MorphologyB"}
    res = br.score_against_truth(records, truth)
    assert res["n_compared"] == 2
    assert res["n_idk"] == 1
    assert res["n_correct"] == 1  # only B
    assert res["accuracy"] == 50.0


def test_score_no_overlap_is_zero():
    res = br.score_against_truth([_rec("X", "MorphologyA")], {"A": "MorphologyA"})
    assert res["n_compared"] == 0
    assert res["accuracy"] == 0.0


def test_score_call_case_insensitive():
    # Legacy lowercase "wt" call is aliased+matched against a MorphologyA truth.
    res = br.score_against_truth([_rec("A", "wt")], {"A": "MorphologyA"})
    assert res["n_correct"] == 1


@pytest.mark.parametrize("acc,mood", [
    (100.0, "happy"), (95.0, "happy"), (80.0, "happy"),
    (60.0, "meh"), (30.0, "sad"), (0.0, "sad"),
])
def test_funny_verdict_moods(acc, mood):
    message, got_mood = br.funny_verdict(acc)
    assert got_mood == mood
    assert isinstance(message, str) and message


# --- discovery -----------------------------------------------------------

def test_discover_tiffs_sorted(tmp_path):
    for name in ["b.tif", "a.tif", "c.tif"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "note.txt").write_bytes(b"")
    found = br.discover_tiffs(str(tmp_path))
    assert [br.file_stem(f) for f in found] == ["a", "b", "c"]


def test_file_stem_strips_ome():
    assert br.file_stem("/x/y/S9MEF_01.ome.tif") == "S9MEF_01"
    assert br.file_stem("/x/y/S9MEF_02.tif") == "S9MEF_02"
