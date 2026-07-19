#!/usr/bin/env python3

"""
Standalone utility: given a DIRECTORY of TIFF mask files, compute the
local thickness of each mask, save a per-file 2-channel TIFF next to the
input, and write a single summary CSV covering the whole batch.

Per-file output (next to each input mask):
    {input_stem}_with_thickness.tif      2-channel ImageJ hyperstack
        channel 0 : the input mask (float32, 0.0 outside, 1.0 inside)
        channel 1 : local thickness  (float32, voxels; 0.0 outside the mask)

Summary output (one row per input file, in --output-csv or by default
{input_dir}/mask_thickness_summary.csv):
    filename, n_voxels_in_mask, mask_fraction,
    mean_thickness_voxels, median_thickness_voxels,
    std_thickness_voxels, max_thickness_voxels,
    output_path

Thickness stats are computed over MASK voxels only — background zeros never
enter the average, so the "mean" is the real interior thickness.

Thickness is computed via the `localthickness` PyPI package (same fallback
logic as `network_line_scan`'s thickness filter). Not wired into the
`mito_protein_localization` dispatcher — this is a side utility. Invoke
directly:

    mask_thickness --input-dir examples/SEPT9_REPRESENTATIVE/manual_masks/

or without the console_scripts shim:

    python -m mito_linescan.mask_thickness_util --input-dir <dir>

Input dimensionality is auto-detected per file:
  - 2D (H, W)         : mask directly
  - 3D (Z, H, W)      : volumetric mask (leading dim > 4)
  - 3D (C, H, W)      : `--channel` picks the mask (leading dim ≤ 4)
  - 4D (C,Z,H,W) / (Z,C,H,W) : `--channel` picks the mask; heuristic /
                               `--channel-axis` decides which axis is C

Failures on individual files are logged and skipped rather than aborting
the whole batch — the summary CSV includes every file that succeeded.
"""

import csv
import glob
import os

import click
import numpy as np
import tifffile as tf


# -------- local thickness helper -----------------------------------------

def _local_thickness(binary):
    """Local thickness map of a 2D or 3D binary array (in voxel units).

    Uses the `localthickness` PyPI package with the same fallback logic as
    `network_line_scan`. Raises `click.ClickException` (rather than
    returning None) so this side-utility fails loudly if the package isn't
    installed — you can always add the required dependency without hunting
    through logs.
    """
    try:
        import localthickness as lt
    except ImportError as exc:
        raise click.ClickException(
            f"The `localthickness` PyPI package is required "
            f"({exc}). Install with `pip install localthickness` "
            f"or add it to your conda env."
        )

    bn = np.ascontiguousarray(binary.astype(np.uint8))
    try:
        thk = lt.local_thickness(bn)
    except AttributeError:
        try:
            thk = lt.local_thickness_2d(bn)
        except Exception as exc:
            raise click.ClickException(
                f"localthickness call failed on shape {bn.shape}: {exc}"
            )
    except Exception as exc:
        # Some versions of localthickness only support 2D; for a 3D mask
        # they raise on the 3D call. Fall back to slice-by-slice.
        if bn.ndim == 3:
            click.echo(
                f"    [thickness] 3D call raised ({exc!s}); falling back to "
                f"per-slice 2D thickness. Note: this measures per-slice "
                f"thickness, not true 3D thickness."
            )
            try:
                slices = [lt.local_thickness(bn[z]) for z in range(bn.shape[0])]
            except AttributeError:
                slices = [lt.local_thickness_2d(bn[z]) for z in range(bn.shape[0])]
            thk = np.stack(slices, axis=0)
        else:
            raise click.ClickException(
                f"localthickness call failed on shape {bn.shape}: {exc}"
            )

    thk = np.asarray(thk, dtype=np.float32)
    if thk.shape != bn.shape:
        raise click.ClickException(
            f"localthickness returned shape {thk.shape}, "
            f"expected {bn.shape}."
        )
    return thk


# -------- input shape handling -------------------------------------------

def _extract_mask(arr, channel, channel_axis):
    """Pick the mask plane out of the input array.

    Returns (mask, was_volumetric) where:
      - `mask` is a 2D (H, W) or 3D (Z, H, W) array containing the mask
      - `was_volumetric` says whether the mask spans multiple Z slices,
        used to decide the output axes ('CYX' vs 'ZCYX').

    `channel_axis` overrides the automatic detection; pass None for auto.
    """
    if arr.ndim == 2:
        return arr, False
    if arr.ndim == 3:
        if channel_axis is not None:
            return np.take(arr, channel, axis=channel_axis), False
        # Ambiguous: (C, H, W) or (Z, H, W). Heuristic: small leading dim = channels.
        if arr.shape[0] <= 4:
            return np.take(arr, channel, axis=0), False
        return arr, True
    if arr.ndim == 4:
        if channel_axis is not None:
            axis = channel_axis
        elif arr.shape[0] <= 4:
            axis = 0
        elif arr.shape[1] <= 4:
            axis = 1
        else:
            raise click.ClickException(
                f"Ambiguous 4D shape {arr.shape}: can't tell which axis "
                f"is channels. Pass --channel-axis explicitly (0..3)."
            )
        return np.take(arr, channel, axis=axis), True
    raise click.ClickException(
        f"Unsupported input dimensionality: shape {arr.shape}"
    )


# -------- per-file driver -----------------------------------------------

def _process_one_file(input_path, channel, channel_axis):
    """Run the full extract -> thickness -> save pipeline on one TIFF.

    Returns a dict of summary stats suitable for CSV output, or None if the
    file could not be processed. Any error is logged to stderr but not
    raised (so a bad file doesn't abort the batch).
    """
    try:
        arr = tf.imread(input_path)
    except Exception as exc:
        click.echo(f"    ERROR: could not read {input_path}: {exc}")
        return None

    click.echo(f"    shape={arr.shape}, dtype={arr.dtype}, "
               f"min={arr.min()}, max={arr.max()}")

    try:
        mask, was_volumetric = _extract_mask(arr, channel, channel_axis)
    except click.ClickException as exc:
        click.echo(f"    ERROR extracting mask: {exc.message}")
        return None
    binary = (mask > 0)

    n_voxels = int(binary.sum())
    if n_voxels == 0:
        click.echo("    WARNING: mask is empty; writing thickness=0 and skipping stats.")
        thk = np.zeros_like(mask, dtype=np.float32)
        stats = dict(mean=0.0, median=0.0, std=0.0, max=0.0)
    else:
        try:
            thk = _local_thickness(binary)
        except click.ClickException as exc:
            click.echo(f"    ERROR computing thickness: {exc.message}")
            return None
        vals = thk[binary]
        stats = dict(
            mean=float(vals.mean()),
            median=float(np.median(vals)),
            std=float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            max=float(vals.max()),
        )
    click.echo(
        f"    thickness (voxels, mask only): "
        f"mean={stats['mean']:.3f} median={stats['median']:.3f} "
        f"std={stats['std']:.3f} max={stats['max']:.3f}"
    )

    # Assemble 2-channel ImageJ hyperstack.
    mask_f = binary.astype(np.float32)
    if was_volumetric:
        combined = np.stack([mask_f, thk], axis=1)  # (Z, 2, Y, X)
        axes = 'ZCYX'
    else:
        combined = np.stack([mask_f, thk], axis=0)  # (2, Y, X)
        axes = 'CYX'

    # Output path: {input_stem}_with_thickness.tif in the same directory.
    stem = os.path.splitext(input_path)[0]
    if stem.lower().endswith('.ome'):
        stem = stem[:-len('.ome')]
    out_path = f"{stem}_with_thickness.tif"

    try:
        tf.imwrite(out_path, combined, imagej=True,
                   metadata={'axes': axes})
    except Exception as exc:
        click.echo(f"    ERROR writing {out_path}: {exc}")
        return None
    click.echo(f"    wrote -> {out_path}  (axes={axes}, dtype={combined.dtype})")

    return {
        'filename': os.path.basename(input_path),
        'n_voxels_in_mask': n_voxels,
        'mask_fraction': float(binary.mean()),
        'mean_thickness_voxels': stats['mean'],
        'median_thickness_voxels': stats['median'],
        'std_thickness_voxels': stats['std'],
        'max_thickness_voxels': stats['max'],
        'output_path': out_path,
    }


# -------- CLI ------------------------------------------------------------

@click.command()
@click.option('--input-dir', required=True,
              type=click.Path(exists=True, file_okay=False),
              help='Input directory containing mask TIFF files.')
@click.option('--input-pattern', default='*.tif', show_default=True,
              help='Glob pattern for mask TIFFs within --input-dir.')
@click.option('--output-csv', default='', type=str,
              help='Path to the summary CSV. Defaults to '
                   '{input-dir}/mask_thickness_summary.csv.')
@click.option('--channel', default=0, type=int, show_default=True,
              help='Channel index of the mask (used only when the input '
                   'has a channel axis).')
@click.option('--channel-axis', default=None, type=int,
              help='Force which axis is channels (see module docstring).')
@click.option('--recursive/--no-recursive', default=False,
              help='Recurse into subdirectories of --input-dir.')
def main(input_dir, input_pattern, output_csv, channel, channel_axis,
         recursive):
    """Batch-compute local thickness for every mask TIFF in a directory."""
    if recursive:
        pattern = os.path.join(input_dir, '**', input_pattern)
        files = sorted(glob.glob(pattern, recursive=True))
    else:
        files = sorted(glob.glob(os.path.join(input_dir, input_pattern)))

    # Filter out any {stem}_with_thickness.tif we might have written on a
    # previous run — we don't want to reprocess our own outputs.
    files = [f for f in files
             if not os.path.basename(f).lower().endswith('_with_thickness.tif')]

    if not files:
        click.echo(f"No files matched {os.path.join(input_dir, input_pattern)}")
        return

    click.echo(f"Processing {len(files)} file(s) from {input_dir}")

    rows = []
    for i, path in enumerate(files, 1):
        click.echo(f"[{i}/{len(files)}] {os.path.basename(path)}")
        row = _process_one_file(path, channel, channel_axis)
        if row is not None:
            rows.append(row)

    if not rows:
        click.echo("No files processed successfully; skipping summary CSV.")
        return

    if not output_csv:
        output_csv = os.path.join(input_dir, 'mask_thickness_summary.csv')

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    click.echo(f"\nSummary CSV -> {output_csv}")
    click.echo(f"Processed {len(rows)}/{len(files)} file(s) successfully.")

    # A brief batch-wide summary of the mean-thickness column so the user
    # gets a single "average thickness across all masks" line.
    means = np.array([r['mean_thickness_voxels'] for r in rows], dtype=float)
    click.echo(f"Batch mean of per-file mean-thickness: {means.mean():.3f} voxels")
    click.echo(f"Batch median of per-file mean-thickness: "
               f"{float(np.median(means)):.3f} voxels")


if __name__ == '__main__':
    main()
