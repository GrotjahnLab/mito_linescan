#!/usr/bin/env python3
"""Standalone blind-review utility for genotype scoring.

This tool is **not** part of the analysis pipeline. It lives in the package so
that the manual scoring procedure used by reviewers is documented and
reproducible. Nothing in the pipeline imports it, and it imports nothing from
the pipeline modules.

A reviewer is shown one grayscale image at a time, in a per-reviewer
randomized order, with the real filename hidden. For each image they call
``WT``, ``KO`` or ``IDK`` (by button or keyboard). Calls are written to a
per-reviewer CSV keyed by file stem and merged as a new column into the master
genotype sheet.

The scientific/decision logic is factored into pure functions at the top of the
module (discovery, plane selection, auto-contrast, shuffle, merge) so it can be
unit-tested without a display. The matplotlib GUI at the bottom is a thin shell
over those functions.

Channel / z-stack rule (see ``select_review_plane``)
----------------------------------------------------
Only one 2D plane is ever shown. For a 3D array we look at the smallest axis:
if it is no larger than ``CHANNEL_MAX_SIZE`` we treat that axis as channels and
take the **first** channel; otherwise the array is treated as a z-series and we
take a **max-intensity projection** along the leading axis. 2D arrays are used
as-is. This mirrors the small-leading-dim guard used in
``mask_thickness_util._extract_mask`` but generalizes it to either axis order.
"""

import glob
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import click
import numpy as np
import pandas as pd
import tifffile as tf

__version__ = "0.1"

# --- module-level tunables ------------------------------------------------
# Robust percentile stretch used to auto-pick display limits per image.
DEFAULT_PLOW = 1.0     # vmin percentile
DEFAULT_PHIGH = 99.5   # vmax percentile
# A 3D axis this size or smaller is assumed to be channels rather than z.
CHANNEL_MAX_SIZE = 4

# Valid scoring calls.
CALLS = ("WT", "KO", "IDK")

# Columns of the per-reviewer CSV, in order.
REVIEWER_CSV_COLUMNS = [
    "FILE", "call", "reviewer", "presentation_index",
    "vmin", "vmax", "seconds_on_image", "timestamp_utc",
]

# Column of the master genotype sheet that holds file stems.
MASTER_KEY = "FILE"


# =========================================================================
# Pure, display-free logic (unit-tested)
# =========================================================================

def discover_tiffs(directory, pattern="*.tif"):
    """Return a sorted list of TIFF paths in ``directory``.

    Uses the same ``sorted(glob.glob(...))`` discovery as the other batch
    tools (e.g. ``mask_thickness_util``, ``mito_protein_omm_normal_scanner``)
    so ordering is deterministic regardless of filesystem order.
    """
    return sorted(glob.glob(os.path.join(directory, pattern)))


def file_stem(path):
    """File stem used to join against the master sheet, e.g. ``S9MEF_01``.

    Strips a trailing ``.ome`` too, matching the convention elsewhere in the
    package where ``foo.ome.tif`` has stem ``foo``.
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    if stem.lower().endswith(".ome"):
        stem = stem[:-len(".ome")]
    return stem


def select_review_plane(arr, channel=0, channel_max=CHANNEL_MAX_SIZE):
    """Reduce an input array to the single 2D plane to display.

    See the module docstring for the full rule. Returns a 2D ``numpy`` array.

    - 2D ``(Y, X)``: returned unchanged.
    - 3D: the smallest axis is inspected. If its length ``<= channel_max`` that
      axis is treated as channels and channel ``channel`` (default the first)
      is taken. Otherwise the array is a z-series and a max-intensity
      projection along the leading axis is returned.
    - Anything else raises ``ValueError``.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        ch_axis = int(np.argmin(arr.shape))
        if arr.shape[ch_axis] <= channel_max:
            idx = min(channel, arr.shape[ch_axis] - 1)
            return np.take(arr, idx, axis=ch_axis)
        # z-series: collapse the leading (z) axis with a max projection.
        return arr.max(axis=0)
    raise ValueError(
        f"Unsupported image dimensionality: shape {arr.shape} (ndim={arr.ndim})"
    )


def auto_contrast(image, p_low=DEFAULT_PLOW, p_high=DEFAULT_PHIGH):
    """Robust percentile display limits for one image.

    Returns ``(vmin, vmax)`` with ``vmin < vmax`` guaranteed.

    Percentiles are computed over the nonzero pixels (microscopy backgrounds
    are typically exact zero); if the image has no zeros we use all pixels. If
    the chosen percentiles collapse (``vmin >= vmax``) we fall back to the full
    min/max, and if that also collapses (constant image) we widen ``vmax`` by
    one so the display range is always valid.
    """
    data = np.asarray(image, dtype=np.float64)
    flat = data.ravel()
    nonzero = flat[flat != 0]
    sample = nonzero if nonzero.size > 0 else flat

    vmin = float(np.percentile(sample, p_low))
    vmax = float(np.percentile(sample, p_high))

    if not vmin < vmax:
        # Percentiles collapsed — fall back to full data range.
        vmin = float(flat.min())
        vmax = float(flat.max())
    if not vmin < vmax:
        # Still collapsed (constant image) — widen so vmin < vmax holds.
        vmax = vmin + 1.0
    return vmin, vmax


def clamp_limits(vmin, vmax):
    """Enforce ``vmin < vmax`` for slider-driven limits without crashing.

    If ``vmin >= vmax`` the max is nudged just above the min.
    """
    vmin = float(vmin)
    vmax = float(vmax)
    if vmin >= vmax:
        vmax = vmin + np.finfo(np.float64).eps + abs(vmin) * 1e-6
    return vmin, vmax


def seed_from_name(name):
    """Deterministic non-negative 32-bit seed derived from a reviewer name.

    Uses a stable hash (not Python's salted ``hash``) so a given reviewer gets
    the same presentation order across runs and machines.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def shuffle_files(files, seed):
    """Return a shuffled copy of ``files`` using ``numpy.random.default_rng``.

    The result is always a permutation of the input.
    """
    rng = np.random.default_rng(seed)
    order = list(files)
    rng.shuffle(order)
    return order


def sanitize_reviewer(name):
    """Filesystem-safe form of a reviewer name: lowercase, non-alnum -> ``_``."""
    out = []
    for ch in name.strip().lower():
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out).strip("_")
    # Collapse runs of underscores for readability.
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "reviewer"


def merge_calls_into_master(master_df, calls, reviewer,
                            allow_new_rows=False, overwrite_column=False):
    """Merge a reviewer's calls into the master genotype dataframe.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Must contain a ``FILE`` column of stems. Existing column order is
        preserved; the new reviewer column is appended at the right.
    calls : dict
        Mapping of file stem -> call string (e.g. ``{"S9MEF_01": "WT"}``).
    reviewer : str
        Original reviewer name; becomes the new column header.
    allow_new_rows : bool
        If True, stems present in ``calls`` but absent from the master sheet
        are appended as new rows. If False they are ignored (the caller warns).
    overwrite_column : bool
        If a column named ``reviewer`` already exists, overwrite it only when
        this is True; otherwise raise ``ValueError``.

    Returns
    -------
    (pandas.DataFrame, list)
        The merged dataframe and the list of extra stems (in ``calls`` but not
        in the original master sheet).
    """
    if MASTER_KEY not in master_df.columns:
        raise ValueError(f"Master sheet has no '{MASTER_KEY}' column.")
    if reviewer in master_df.columns and not overwrite_column:
        raise ValueError(
            f"Column '{reviewer}' already exists in the master sheet. "
            f"Pass --overwrite-column to replace it."
        )

    df = master_df.copy()
    master_stems = set(df[MASTER_KEY].astype(str))
    extras = [stem for stem in calls if stem not in master_stems]

    if allow_new_rows and extras:
        new_rows = pd.DataFrame({MASTER_KEY: extras})
        df = pd.concat([df, new_rows], ignore_index=True)

    # Build the reviewer column aligned to the (possibly extended) FILE order.
    df[reviewer] = df[MASTER_KEY].astype(str).map(calls).fillna("")
    return df, extras


def build_reviewer_dataframe(records):
    """Assemble the per-reviewer CSV dataframe with fixed column order."""
    df = pd.DataFrame(records, columns=REVIEWER_CSV_COLUMNS)
    return df


def build_reviewer_results(records, reviewer):
    """Genotype-results dataframe for one reviewer: FILE stem + call column.

    This is the standalone per-reviewer results table (as opposed to the shared
    master sheet). The call column is named after the reviewer so the file is
    self-describing. Used when no master sheet is supplied; when a master is
    supplied the results are taken from the merged sheet instead so they stay
    aligned to the master's FILE list.
    """
    df = pd.DataFrame({MASTER_KEY: [r["FILE"] for r in records]})
    df[reviewer] = [r["call"] for r in records]
    return df


def atomic_write_csv(df, path):
    """Write ``df`` to ``path`` atomically (temp file in same dir + replace)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# =========================================================================
# GUI shell (thin wrapper over the pure functions above)
# =========================================================================

def _load_plane(path, channel=0):
    """Read a TIFF and reduce it to the 2D display plane."""
    arr = tf.imread(path)
    return select_review_plane(arr, channel=channel)


def run_gui(files, reviewer, p_low, p_high, on_record):
    """Show images one at a time and collect calls. Single forward pass.

    ``on_record(stem, call, presentation_index, vmin, vmax, seconds)`` is
    invoked for each scored image. Returns when the window is closed or every
    image has been scored. Import of matplotlib is deferred so the pure
    functions and CLI discovery/merge paths work in headless environments.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    state = {"idx": 0, "shown_at": None, "vmin": 0.0, "vmax": 1.0,
             "plane": None, "data_lo": 0.0, "data_hi": 1.0}

    fig = plt.figure(figsize=(8, 8))
    ax_img = fig.add_axes([0.08, 0.28, 0.84, 0.64])
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    im = ax_img.imshow(np.zeros((2, 2)), cmap="gray",
                       interpolation="nearest", aspect="equal")

    ax_min = fig.add_axes([0.15, 0.20, 0.70, 0.03])
    ax_max = fig.add_axes([0.15, 0.15, 0.70, 0.03])
    s_min = Slider(ax_min, "min", 0.0, 1.0, valinit=0.0)
    s_max = Slider(ax_max, "max", 0.0, 1.0, valinit=1.0)

    def _apply_limits(update_sliders=False):
        vmin, vmax = clamp_limits(s_min.val, s_max.val)
        state["vmin"], state["vmax"] = vmin, vmax
        im.set_clim(vmin, vmax)
        if update_sliders:
            s_min.eventson = s_max.eventson = False
            s_min.set_val(vmin)
            s_max.set_val(vmax)
            s_min.eventson = s_max.eventson = True
        fig.canvas.draw_idle()

    def _reconfigure_sliders(lo, hi):
        # Point sliders at the current image's raw intensity range.
        for s, ax in ((s_min, ax_min), (s_max, ax_max)):
            s.valmin, s.valmax = lo, hi
            ax.set_xlim(lo, hi)

    def _show_current():
        idx = state["idx"]
        if idx >= len(files):
            plt.close(fig)
            return
        plane = _load_plane(files[idx])
        state["plane"] = plane
        lo, hi = float(np.min(plane)), float(np.max(plane))
        if lo >= hi:
            hi = lo + 1.0
        state["data_lo"], state["data_hi"] = lo, hi
        _reconfigure_sliders(lo, hi)

        vmin, vmax = auto_contrast(plane, p_low, p_high)
        im.set_data(plane)
        im.set_extent((-0.5, plane.shape[1] - 0.5, plane.shape[0] - 0.5, -0.5))
        s_min.eventson = s_max.eventson = False
        s_min.set_val(vmin)
        s_max.set_val(vmax)
        s_min.eventson = s_max.eventson = True
        state["vmin"], state["vmax"] = vmin, vmax
        im.set_clim(vmin, vmax)
        ax_img.set_title(f"Image {idx + 1} / {len(files)}   —   reviewer: {reviewer}")
        state["shown_at"] = time.monotonic()
        fig.canvas.draw_idle()

    def _record_and_advance(call):
        idx = state["idx"]
        if idx >= len(files):
            return
        seconds = time.monotonic() - (state["shown_at"] or time.monotonic())
        on_record(file_stem(files[idx]), call, idx,
                  state["vmin"], state["vmax"], seconds)
        state["idx"] += 1
        _show_current()

    def _reset_contrast():
        if state["plane"] is None:
            return
        vmin, vmax = auto_contrast(state["plane"], p_low, p_high)
        s_min.eventson = s_max.eventson = False
        s_min.set_val(vmin)
        s_max.set_val(vmax)
        s_min.eventson = s_max.eventson = True
        _apply_limits()

    s_min.on_changed(lambda _v: _apply_limits())
    s_max.on_changed(lambda _v: _apply_limits())

    ax_wt = fig.add_axes([0.15, 0.04, 0.20, 0.07])
    ax_ko = fig.add_axes([0.40, 0.04, 0.20, 0.07])
    ax_idk = fig.add_axes([0.65, 0.04, 0.20, 0.07])
    b_wt = Button(ax_wt, "WT", color="#4caf50", hovercolor="#66bb6a")
    b_ko = Button(ax_ko, "KO", color="#e53935", hovercolor="#ef5350")
    b_idk = Button(ax_idk, "IDK", color="#9e9e9e", hovercolor="#bdbdbd")
    b_wt.on_clicked(lambda _e: _record_and_advance("WT"))
    b_ko.on_clicked(lambda _e: _record_and_advance("KO"))
    b_idk.on_clicked(lambda _e: _record_and_advance("IDK"))

    def _on_key(event):
        key = (event.key or "").lower()
        if key in ("1", "w"):
            _record_and_advance("WT")
        elif key in ("2", "k"):
            _record_and_advance("KO")
        elif key in ("3", "i"):
            _record_and_advance("IDK")
        elif key == "r":
            _reset_contrast()

    fig.canvas.mpl_connect("key_press_event", _on_key)

    _show_current()
    if state["idx"] < len(files):  # nothing scored yet / images remain
        plt.show(block=True)


# =========================================================================
# CLI
# =========================================================================

@click.command()
@click.option("--input-directory", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Directory of TIFFs to review.")
@click.option("--reviewer", default="",
              help="Reviewer name. If omitted you are prompted before the "
                   "first image is shown.")
@click.option("--genotype-csv", default="", type=str,
              help="Master genotype sheet with a single FILE column of stems. "
                   "If given, the reviewer's calls are merged in as a new column.")
@click.option("--output-directory", default="", type=str,
              help="Where to write outputs (default: --input-directory).")
@click.option("--seed", default=None, type=int,
              help="Shuffle seed. Default derives a stable seed from the "
                   "reviewer name.")
@click.option("--input-pattern", default="*.tif", show_default=True,
              help="Glob pattern for TIFFs within --input-directory.")
@click.option("--p-low", default=DEFAULT_PLOW, show_default=True, type=float,
              help="Lower percentile for auto-contrast vmin.")
@click.option("--p-high", default=DEFAULT_PHIGH, show_default=True, type=float,
              help="Upper percentile for auto-contrast vmax.")
@click.option("--allow-new-rows", is_flag=True, default=False,
              help="Append master-sheet rows for input files not already listed.")
@click.option("--overwrite-column", is_flag=True, default=False,
              help="Overwrite an existing reviewer column in the master sheet.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Exercise discovery + merge with stub 'IDK' calls, no GUI. "
                   "Useful for headless smoke testing.")
def main(input_directory, reviewer, genotype_csv, output_directory, seed,
         input_pattern, p_low, p_high, allow_new_rows, overwrite_column,
         dry_run):
    """Blindly score TIFFs as WT / KO / IDK and record the calls.

    This utility is standalone and is not part of the analysis pipeline.
    """
    if not reviewer.strip():
        reviewer = click.prompt("Reviewer name").strip()
    if not reviewer.strip():
        raise click.ClickException("A reviewer name is required.")

    out_dir = output_directory or input_directory
    os.makedirs(out_dir, exist_ok=True)

    files = discover_tiffs(input_directory, input_pattern)
    if not files:
        raise click.ClickException(
            f"No TIFFs matching '{input_pattern}' in {input_directory}")

    if seed is None:
        seed = seed_from_name(reviewer)
    order = shuffle_files(files, seed)

    click.echo(f"Reviewer: {reviewer}")
    click.echo(f"Discovered {len(files)} image(s); seed={seed}")

    records = []
    start_time = datetime.now(timezone.utc)

    def _record(stem, call, pres_idx, vmin, vmax, seconds):
        records.append({
            "FILE": stem,
            "call": call,
            "reviewer": reviewer,
            "presentation_index": pres_idx,
            "vmin": vmin,
            "vmax": vmax,
            "seconds_on_image": round(seconds, 3),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })

    if dry_run:
        # Score nothing for real; stub every image as IDK so the merge and
        # round-trip can be exercised without a display.
        for i, path in enumerate(order):
            _record(file_stem(path), "IDK", i, 0.0, 1.0, 0.0)
        click.echo(f"[dry-run] stubbed {len(records)} call(s) as IDK")
    else:
        run_gui(order, reviewer, p_low, p_high, _record)

    end_time = datetime.now(timezone.utc)

    scored = len(records)
    skipped = len(files) - scored
    click.echo(f"Scored {scored} image(s); skipped {skipped}.")

    slug = sanitize_reviewer(reviewer)
    reviewer_csv = os.path.join(out_dir, f"{slug}.csv")
    session_json = os.path.join(out_dir, f"{slug}_session.json")

    atomic_write_csv(build_reviewer_dataframe(records), reviewer_csv)
    click.echo(f"Wrote {reviewer_csv}")

    session = {
        "reviewer": reviewer,
        "seed": int(seed),
        "input_directory": os.path.abspath(input_directory),
        "input_pattern": input_pattern,
        "p_low": p_low,
        "p_high": p_high,
        "channel_max_size": CHANNEL_MAX_SIZE,
        "tool_version": __version__,
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "n_discovered": len(files),
        "n_scored": scored,
        "n_skipped": skipped,
    }
    with open(session_json, "w") as fh:
        json.dump(session, fh, indent=2)
    click.echo(f"Wrote {session_json}")

    results_csv = os.path.join(out_dir, f"{slug}_genotype_results.csv")
    if genotype_csv:
        merged = _merge_into_master(genotype_csv, records, reviewer,
                                    allow_new_rows, overwrite_column)
        # Standalone per-reviewer results: the master's FILE list with only
        # this reviewer's calls, so it stays aligned to the master.
        results = merged[[MASTER_KEY, reviewer]].copy()
    else:
        results = build_reviewer_results(records, reviewer)

    atomic_write_csv(results, results_csv)
    click.echo(f"Wrote {results_csv}")


def _merge_into_master(genotype_csv, records, reviewer,
                       allow_new_rows, overwrite_column):
    """Back up and atomically update the master sheet with the new column.

    Returns the merged dataframe.
    """
    if not os.path.exists(genotype_csv):
        raise click.ClickException(f"Master sheet not found: {genotype_csv}")

    master = pd.read_csv(genotype_csv, dtype=str).fillna("")
    calls = {r["FILE"]: r["call"] for r in records}

    try:
        merged, extras = merge_calls_into_master(
            master, calls, reviewer,
            allow_new_rows=allow_new_rows,
            overwrite_column=overwrite_column,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc))

    if extras:
        if allow_new_rows:
            click.echo(f"Added {len(extras)} new row(s) for unlisted files: "
                       f"{', '.join(sorted(extras))}")
        else:
            click.echo("WARNING: these reviewed files are absent from the "
                       "master sheet and were NOT added (use --allow-new-rows "
                       f"to add them): {', '.join(sorted(extras))}")

    # Back up the original before the first write.
    backup = genotype_csv + ".bak"
    if not os.path.exists(backup):
        import shutil
        shutil.copy2(genotype_csv, backup)
        click.echo(f"Backed up original to {backup}")

    atomic_write_csv(merged, genotype_csv)
    click.echo(f"Merged column '{reviewer}' into {genotype_csv}")
    return merged


if __name__ == "__main__":
    main()
