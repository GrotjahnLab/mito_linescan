#!/usr/bin/env python3

"""
Compute the offset between mitochondria peaks and protein/target peaks along
each per-mito line scan, and plot the distribution (box + violin) grouped by
the first N characters of the source image name.

Input: a DIRECTORY of per-mito CSVs produced by `network_line_scan`. Each CSV
must have columns:  Distance, Mito_Intensity, Scan_Intensity.

For each CSV:
  - peaks are detected on Mito_Intensity and Scan_Intensity with the configured
    intensity (height) + prominence thresholds
  - every protein/target peak is paired with the *nearest* mito peak along the
    Distance axis
  - one row is emitted per paired protein peak, with columns:
        image_name, mito_id,
        mito_peak_distance, target_peak_distance,
        mito_peak_intensity, mito_peak_prominence,
        target_peak_intensity, target_peak_prominence
This gives the existing groupby + outlier + box/violin code the exact same
column shape it had when consuming `analyze_omm_scans` scan_data.csv, so the
plot itself is unchanged — only the data ingestion is new.
"""

import glob
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences


# Per-mito CSV filename pattern emitted by network_line_scan.
DEFAULT_INPUT_PATTERN = '*_mito_*.csv'
# Window length (in samples) used for peak_prominences. Matches the line
# scanner's per-mito plotting convention.
PROMINENCE_WLEN = 10


def _parse_image_name(path):
    """Recover the source image name from a per-mito CSV path by stripping
    the trailing `_mito_<N>.csv` suffix."""
    base = os.path.basename(path)
    base = base[:-4] if base.lower().endswith('.csv') else base
    idx = base.rfind('_mito_')
    return base[:idx] if idx > 0 else base


def _parse_mito_id(path):
    """Recover the mito index from a per-mito CSV path. Returns -1 if absent."""
    base = os.path.basename(path)
    base = base[:-4] if base.lower().endswith('.csv') else base
    parts = base.split('_mito_')
    if len(parts) < 2:
        return -1
    tail = parts[-1].split('_')[0]
    try:
        return int(tail)
    except ValueError:
        return -1


def _find_filtered_peaks(y, min_intensity, min_prominence, wlen=PROMINENCE_WLEN):
    """Detect peaks above `min_intensity` with prominence >= `min_prominence`.
    Returns (peak_indices, peak_prominences) — both same length, both in the
    original sample-index space."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return np.array([], dtype=int), np.array([])
    peaks, _ = find_peaks(y, height=min_intensity if min_intensity is not None else None)
    if peaks.size == 0:
        return peaks, np.array([])
    proms = peak_prominences(y, peaks, wlen=wlen)[0]
    if min_prominence is not None:
        mask = proms >= min_prominence
        return peaks[mask], proms[mask]
    return peaks, proms


def build_scan_data_from_line_scan_dir(csv_directory, input_pattern,
                                       min_mito_intensity, min_mito_prominence,
                                       min_target_intensity, min_target_prominence,
                                       recursive=False):
    """Read every per-mito CSV in `csv_directory`, detect peaks on both
    channels, pair each protein peak to the nearest mito peak, and assemble
    a DataFrame matching the schema the downstream plotting code expects.
    """
    if recursive:
        files = sorted(glob.glob(
            os.path.join(csv_directory, '**', input_pattern), recursive=True))
    else:
        files = sorted(glob.glob(os.path.join(csv_directory, input_pattern)))
    if not files:
        raise click.ClickException(
            f"No CSV files matched {input_pattern!r} under {csv_directory!r}"
        )

    print(f"Found {len(files)} CSV files in {csv_directory}")

    rows = []
    n_short = n_no_mito = n_no_scan = 0
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"  WARN: failed to read {f}: {exc}")
            continue
        needed = {'Distance', 'Mito_Intensity', 'Scan_Intensity'}
        if not needed.issubset(df.columns):
            print(f"  WARN: {os.path.basename(f)} missing {needed - set(df.columns)}; skipping.")
            continue
        d = df['Distance'].to_numpy(dtype=float)
        mi = df['Mito_Intensity'].to_numpy(dtype=float)
        si = df['Scan_Intensity'].to_numpy(dtype=float)
        if d.size < 3:
            n_short += 1
            continue

        mito_idx, mito_proms = _find_filtered_peaks(
            mi, min_mito_intensity, min_mito_prominence)
        scan_idx, scan_proms = _find_filtered_peaks(
            si, min_target_intensity, min_target_prominence)

        if mito_idx.size == 0:
            n_no_mito += 1
            continue
        if scan_idx.size == 0:
            n_no_scan += 1
            continue

        img_name = _parse_image_name(f)
        mito_id = _parse_mito_id(f)
        mito_d = d[mito_idx]
        scan_d = d[scan_idx]

        # For each protein/target peak, pair with the *nearest* mito peak.
        for j, s_pos in enumerate(scan_d):
            k = int(np.argmin(np.abs(mito_d - s_pos)))
            rows.append({
                'image_name': img_name,
                'mito_id': mito_id,
                'mito_peak_distance': float(mito_d[k]),
                'mito_peak_intensity': float(mi[mito_idx[k]]),
                'mito_peak_prominence': float(mito_proms[k]),
                'target_peak_distance': float(s_pos),
                'target_peak_intensity': float(si[scan_idx[j]]),
                'target_peak_prominence': float(scan_proms[j]),
            })

    print(f"  Skipped (short):      {n_short}")
    print(f"  Skipped (no mito pk): {n_no_mito}")
    print(f"  Skipped (no scan pk): {n_no_scan}")
    print(f"  Paired peaks total:   {len(rows)}")

    if not rows:
        raise click.ClickException(
            "No peaks found in any CSV. Loosen min_*_intensity / min_*_prominence."
        )
    return pd.DataFrame(rows)


def identify_outliers(df, column='peak_distance'):
    """
    IQR-based outlier detection on `column`, per group.

    Args:
        df: DataFrame with 'group', the chosen `column`, and 'image_name' columns
        column: name of the numeric column to scan for outliers

    Returns:
        List of dicts with group, image_name, value, lower_bound, upper_bound.
    """
    outliers = []

    groups = sorted(df['group'].unique())
    for group in groups:
        group_data = df[df['group'] == group]
        values = group_data[column]

        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        group_outliers = group_data[(group_data[column] < lower_bound) |
                                    (group_data[column] > upper_bound)]

        for _, row in group_outliers.iterrows():
            outliers.append({
                'group': group,
                'image_name': row['image_name'],
                column: row[column],
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
            })

    return outliers


def analyze_peak_distances(csv_directory, input_pattern=DEFAULT_INPUT_PATTERN,
                           output_dir=None, group_by_chars=5,
                           min_mito_intensity=None, max_mito_intensity=None,
                           min_target_intensity=None, max_target_intensity=None,
                           min_mito_prominence=None, max_mito_prominence=None,
                           min_target_prominence=None, max_target_prominence=None,
                           bin_width=1.0, max_abs_distance=0.0,
                           recursive=False):
    """
    Build the mito-vs-protein peak-distance dataframe from a directory of
    network_line_scan per-mito CSVs, then plot the distribution (box + violin)
    grouped by the first N characters of image_name.

    Args:
        csv_directory: Directory containing per-mito CSVs from network_line_scan.
        input_pattern: Glob pattern within the directory (default: '*_mito_*.csv').
        output_dir: Where to save outputs (defaults to csv_directory).
        group_by_chars: Number of starting characters of image_name to group by.
        min_*_intensity / min_*_prominence: peak-detection thresholds applied
            during ingestion. Max_* thresholds are applied as POST-FILTER
            constraints on the resulting peak attributes (same semantics as
            the previous OMM-based version of this script).
        recursive: If True, recurse into subdirectories when globbing.
    """
    # Resolve output directory before any I/O.
    if output_dir is None or output_dir == '':
        output_dir = csv_directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1) ingest: detect peaks per CSV with the min thresholds, then pair.
    df = build_scan_data_from_line_scan_dir(
        csv_directory,
        input_pattern=input_pattern,
        min_mito_intensity=min_mito_intensity,
        min_mito_prominence=min_mito_prominence,
        min_target_intensity=min_target_intensity,
        min_target_prominence=min_target_prominence,
        recursive=recursive,
    )
    original_count = len(df)

    # 2) post-filter: max thresholds apply after pairing.
    print("Applying max-thresholds:")
    if max_mito_intensity is not None:
        print(f"  Mito intensity <= {max_mito_intensity}")
        df = df[df['mito_peak_intensity'] <= max_mito_intensity]
    if max_target_intensity is not None:
        print(f"  Target intensity <= {max_target_intensity}")
        df = df[df['target_peak_intensity'] <= max_target_intensity]
    if max_mito_prominence is not None:
        print(f"  Mito prominence <= {max_mito_prominence}")
        df = df[df['mito_peak_prominence'] <= max_mito_prominence]
    if max_target_prominence is not None:
        print(f"  Target prominence <= {max_target_prominence}")
        df = df[df['target_peak_prominence'] <= max_target_prominence]

    filtered_count = len(df)
    print(f"\nPaired peaks: {original_count} ingested -> {filtered_count} after max-filters")

    if filtered_count == 0:
        print("No data remaining after applying thresholds!")
        return

    # Extract the specified number of characters from image_name as group
    df['group'] = df['image_name'].str[:group_by_chars]

    # Absolute distance between the paired mito peak and protein/target peak,
    # in pixels along the path. We deliberately take the absolute value: the
    # sign of (mito - target) on a line scan just reflects which direction
    # the path happened to be traced, so it has no biological meaning here.
    # (Carrying a signed difference was an OMM-scan convention that does not
    # transfer to the network line scan output.)
    df['peak_distance'] = np.abs(
        df['mito_peak_distance'] - df['target_peak_distance']
    )

    # Persist the assembled scan_data for downstream inspection.
    scan_data_path = os.path.join(output_dir, 'scan_data.csv')
    df.to_csv(scan_data_path, index=False)
    print(f"Saved assembled scan_data.csv: {scan_data_path}")

    print(f"Groups: {df['group'].unique()}")
    print(f"\nSummary statistics (peak_distance, px):")
    print(df.groupby('group')['peak_distance'].describe())

    # Identify and print outliers (on the absolute distance)
    outliers = identify_outliers(df, column='peak_distance')
    if outliers:
        print(f"\n\nOUTLIERS DETECTED ({len(outliers)} total):")
        print("-" * 80)
        for outlier in outliers:
            print(f"Group: {outlier['group']:5s} | Image: {outlier['image_name']:40s} | "
                  f"Value: {outlier['peak_distance']:8.2f} | "
                  f"Bounds: [{outlier['lower_bound']:.2f}, {outlier['upper_bound']:.2f}]")

        # Save outliers to CSV file
        outliers_df = pd.DataFrame(outliers)
        outliers_file = os.path.join(output_dir, 'peak_distance_outliers.csv')
        outliers_df.to_csv(outliers_file, index=False)
        print(f"\nOutliers saved to {outliers_file}")
    else:
        print("\nNo outliers detected.")


    # Histogram of |mito_peak_distance - target_peak_distance|, pooled across
    # all paired peaks. Median + mean overlaid. The x-axis cap defaults to
    # the data's p99 unless the caller pinned it via `max_abs_distance`.
    vals = df['peak_distance'].to_numpy(dtype=float)   # already >= 0
    if max_abs_distance and max_abs_distance > 0:
        upper = float(max_abs_distance)
    else:
        upper = float(np.percentile(vals, 99))
    upper = max(upper, float(bin_width))
    bins = np.arange(0.0, upper + bin_width, bin_width)

    clipped = vals[vals <= upper]
    n_clipped = vals.size - clipped.size

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(clipped, bins=bins, edgecolor='black', alpha=0.85)
    ax.axvline(float(np.median(vals)), color='red', linestyle='--',
               linewidth=1.5, label=f'median = {np.median(vals):.2f}')
    ax.axvline(float(vals.mean()), color='orange', linestyle=':',
               linewidth=1.5, label=f'mean = {vals.mean():.2f}')
    ax.set_xlabel('|Mito peak − Target peak| distance along path (px)')
    ax.set_ylabel('Count')
    title = (f'Mito-to-target peak distance  (n={vals.size} paired peaks from '
             f"{df['mito_id'].nunique()} tracks, {df['image_name'].nunique()} images)")
    if n_clipped:
        title += f'  ·  {n_clipped} values > {upper:.1f} px not shown'
    ax.set_title(title)
    ax.set_xlim(0, upper)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_file = os.path.join(output_dir, 'peak_distance_histogram.png')
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"\nSaved histogram to {output_file}")


@click.command()
@click.option('--csv-directory', type=click.Path(exists=True, file_okay=False),
              required=True,
              help='Directory containing per-mito CSVs from network_line_scan '
                   '(columns: Distance, Mito_Intensity, Scan_Intensity).')
@click.option('--input-pattern', type=str, default=DEFAULT_INPUT_PATTERN,
              show_default=True,
              help='Glob pattern for the per-mito CSV files inside --csv-directory.')
@click.option('--output-directory', type=click.Path(), default=None,
              help='Output directory for plots and assembled scan_data.csv '
                   '(defaults to --csv-directory).')
@click.option('--group-by-chars', type=int, default=5, show_default=True,
              help='Number of starting characters of image_name to group by.')
@click.option('--min-mito-intensity', type=float, default=0.3, show_default=True,
              help='Minimum mito peak intensity (height threshold during ingestion).')
@click.option('--max-mito-intensity', type=float, default=None,
              help='Maximum mito peak intensity (post-pair filter).')
@click.option('--min-target-intensity', type=float, default=0.2, show_default=True,
              help='Minimum target peak intensity (height threshold during ingestion).')
@click.option('--max-target-intensity', type=float, default=None,
              help='Maximum target peak intensity (post-pair filter).')
@click.option('--min-mito-prominence', type=float, default=0.08, show_default=True,
              help='Minimum mito peak prominence (filter during ingestion).')
@click.option('--max-mito-prominence', type=float, default=None,
              help='Maximum mito peak prominence (post-pair filter).')
@click.option('--min-target-prominence', type=float, default=0.05, show_default=True,
              help='Minimum target peak prominence (filter during ingestion).')
@click.option('--max-target-prominence', type=float, default=None,
              help='Maximum target peak prominence (post-pair filter).')
@click.option('--bin-width', type=float, default=1.0, show_default=True,
              help='Histogram bin width (px).')
@click.option('--max-abs-distance', type=float, default=0.0, show_default=True,
              help='Histogram x-axis cap |x| <= this. 0 = auto from data (p99).')
@click.option('--recursive/--no-recursive', default=False,
              help='Recurse into subdirectories of --csv-directory.')
def main(csv_directory, input_pattern, output_directory, group_by_chars,
         min_mito_intensity, max_mito_intensity,
         min_target_intensity, max_target_intensity,
         min_mito_prominence, max_mito_prominence,
         min_target_prominence, max_target_prominence,
         bin_width, max_abs_distance,
         recursive):
    """
    Plot the offset between mitochondria peaks and protein/target peaks along
    network_line_scan per-mito CSVs, grouped by image-name prefix.

    Example:
        python plot_peak_distance_analysis.py --csv-directory run1_output
        python plot_peak_distance_analysis.py --csv-directory run1_output --group-by-chars 3
        python plot_peak_distance_analysis.py --csv-directory run1_output \\
            --min-mito-intensity 0.1 --min-target-intensity 0.1
    """
    analyze_peak_distances(
        csv_directory,
        input_pattern=input_pattern,
        output_dir=output_directory,
        group_by_chars=group_by_chars,
        min_mito_intensity=min_mito_intensity,
        max_mito_intensity=max_mito_intensity,
        min_target_intensity=min_target_intensity,
        max_target_intensity=max_target_intensity,
        min_mito_prominence=min_mito_prominence,
        max_mito_prominence=max_mito_prominence,
        min_target_prominence=min_target_prominence,
        max_target_prominence=max_target_prominence,
        bin_width=bin_width,
        max_abs_distance=max_abs_distance,
        recursive=recursive,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
