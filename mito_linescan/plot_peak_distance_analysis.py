#!/usr/bin/env python3

"""
Pool every per-mito line-scan CSV under an input directory, detect peaks on
the Scan_Intensity column (protein/target channel) with user-supplied
intensity + prominence thresholds, compute the distance between
*consecutive* peaks within each track, and write a histogram of all the
distances pooled across every track.

CSV input format (produced by mito_protein_line_scanner.py):
  columns: Distance, Mito_Intensity, Scan_Intensity
  one row per sampled point along a mitochondrial path; intensities are
  normalised to [0, 1] inside the line scanner.

Outputs (in --output-directory, defaults to --csv-directory):
  peak_distance_histogram.png      pooled histogram of consecutive distances
  peak_distances.csv               every distance value (one row per pair)
  peak_distance_per_track.csv      per-track summary
  peak_distance_summary.txt        global summary stats

Distances are in whatever units the CSV's Distance column is in (pixels by
default for the line scanner).
"""

import glob
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences


DEFAULT_INPUT_PATTERN = '*_mito_*.csv'


# -------- helpers --------------------------------------------------------

def _parse_image_name(path):
    """Strip the trailing `_mito_<N>.csv` to recover the source image name."""
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


def find_filtered_peaks(y, min_intensity, min_prominence, wlen):
    """Detect peaks with `height >= min_intensity` and
    `prominence >= min_prominence`."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(y, height=min_intensity)
    if peaks.size == 0:
        return peaks
    proms = peak_prominences(y, peaks, wlen=wlen)[0]
    return peaks[proms >= min_prominence]


def consecutive_distances(distance, peak_indices):
    """Return distance[peak[i+1]] - distance[peak[i]] for sorted peaks."""
    if peak_indices.size < 2:
        return np.array([], dtype=float)
    d = np.asarray(distance, dtype=float)
    # defensive sort in case the line scanner ever emits non-monotonic Distance
    order = np.argsort(d)
    d_sorted = d[order]
    inv = np.empty_like(order); inv[order] = np.arange(order.size)
    peak_sorted = np.sort(inv[peak_indices])
    return np.diff(d_sorted[peak_sorted])


# -------- CLI ------------------------------------------------------------

@click.command()
@click.option('--csv-directory', required=True,
              type=click.Path(exists=True, file_okay=False),
              help='Directory of per-mito line-scan CSVs '
                   '(columns: Distance, Mito_Intensity, Scan_Intensity).')
@click.option('--input-pattern', default=DEFAULT_INPUT_PATTERN, show_default=True,
              help='Glob within --csv-directory.')
@click.option('--output-directory', default='', type=click.Path(),
              help='Where to write outputs (default = --csv-directory).')
@click.option('--peak-min-intensity', default=0.3, show_default=True, type=float,
              help='Minimum Scan_Intensity (0-1) to count as a peak.')
@click.option('--peak-min-prominence', default=0.1, show_default=True, type=float,
              help='Minimum peak prominence (0-1).')
@click.option('--peak-prominence-wlen', default=10, show_default=True, type=int,
              help='`wlen` passed to scipy.signal.peak_prominences.')
@click.option('--bin-width', default=5.0, show_default=True, type=float,
              help='Histogram bin width in px.')
@click.option('--max-distance', default=0.0, show_default=True, type=float,
              help='Histogram x-axis cap; 0 = auto from data (p99).')
@click.option('--recursive/--no-recursive', default=False,
              help='Recurse into subdirectories of --csv-directory.')
def main(csv_directory, input_pattern, output_directory,
         peak_min_intensity, peak_min_prominence, peak_prominence_wlen,
         bin_width, max_distance, recursive):
    """Pool consecutive peak-to-peak distances on Scan_Intensity from every
    per-mito CSV under --csv-directory and plot a histogram."""
    out_dir = output_directory or csv_directory
    os.makedirs(out_dir, exist_ok=True)

    if recursive:
        csv_files = sorted(glob.glob(
            os.path.join(csv_directory, '**', input_pattern), recursive=True))
    else:
        csv_files = sorted(glob.glob(os.path.join(csv_directory, input_pattern)))

    if not csv_files:
        raise click.ClickException(
            f"No CSV files matched {input_pattern!r} under {csv_directory!r}"
        )
    click.echo(f"Found {len(csv_files)} CSV files in {csv_directory}")

    all_distances = []   # rows: {image_name, mito_id, distance}
    per_track = []       # one row per CSV
    n_short = 0
    n_no_pair = 0
    n_bad = 0

    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            click.echo(f"  WARN: failed to read {f}: {exc}")
            n_bad += 1
            continue
        if not {'Distance', 'Scan_Intensity'}.issubset(df.columns):
            click.echo(f"  WARN: {os.path.basename(f)} missing Distance/"
                       f"Scan_Intensity; skipping.")
            n_bad += 1
            continue

        img = _parse_image_name(f)
        mid = _parse_mito_id(f)
        d = df['Distance'].to_numpy(dtype=float)
        si = df['Scan_Intensity'].to_numpy(dtype=float)

        if si.size < 3:
            n_short += 1
            continue

        peaks = find_filtered_peaks(
            si,
            min_intensity=peak_min_intensity,
            min_prominence=peak_min_prominence,
            wlen=peak_prominence_wlen,
        )
        distances = consecutive_distances(d, peaks)

        per_track.append({
            'image_name': img,
            'mito_id': mid,
            'n_samples': int(si.size),
            'path_length': float(d.max() - d.min()) if d.size else 0.0,
            'n_peaks': int(peaks.size),
            'n_distances': int(distances.size),
            'mean_distance': float(distances.mean()) if distances.size else np.nan,
            'median_distance': (float(np.median(distances))
                                if distances.size else np.nan),
        })

        if distances.size == 0:
            n_no_pair += 1
            continue

        for v in distances:
            all_distances.append({'image_name': img, 'mito_id': mid,
                                  'distance': float(v)})

    if not all_distances:
        raise click.ClickException(
            "No consecutive peak pairs found. Loosen "
            "--peak-min-intensity / --peak-min-prominence."
        )

    distances_df = pd.DataFrame(all_distances)
    per_track_df = pd.DataFrame(per_track)

    distances_csv = os.path.join(out_dir, 'peak_distances.csv')
    per_track_csv = os.path.join(out_dir, 'peak_distance_per_track.csv')
    summary_txt = os.path.join(out_dir, 'peak_distance_summary.txt')
    hist_png = os.path.join(out_dir, 'peak_distance_histogram.png')

    distances_df.to_csv(distances_csv, index=False)
    per_track_df.to_csv(per_track_csv, index=False)

    vals = distances_df['distance'].to_numpy()
    summary = (
        f"Peak-distance summary\n"
        f"  Input dir:                {csv_directory}\n"
        f"  Pattern:                  {input_pattern}\n"
        f"  CSVs found:               {len(csv_files)}\n"
        f"  CSVs unreadable/bad:      {n_bad}\n"
        f"  CSVs too short:           {n_short}\n"
        f"  CSVs with <2 peaks:       {n_no_pair}\n"
        f"  Tracks contributing:      {(per_track_df['n_distances'] > 0).sum()}\n"
        f"  Total peaks detected:     {int(per_track_df['n_peaks'].sum())}\n"
        f"  Total consecutive pairs:  {vals.size}\n"
        f"  Thresholds:               intensity>={peak_min_intensity}  "
        f"prominence>={peak_min_prominence}  wlen={peak_prominence_wlen}\n"
        f"  Units:                    same as CSV 'Distance' column (px by default)\n"
        f"  Distance stats:\n"
        f"    mean   = {vals.mean():.3f}\n"
        f"    median = {np.median(vals):.3f}\n"
        f"    std    = {vals.std():.3f}\n"
        f"    p10    = {np.percentile(vals, 10):.3f}\n"
        f"    p25    = {np.percentile(vals, 25):.3f}\n"
        f"    p75    = {np.percentile(vals, 75):.3f}\n"
        f"    p90    = {np.percentile(vals, 90):.3f}\n"
        f"    p99    = {np.percentile(vals, 99):.3f}\n"
        f"    min    = {vals.min():.3f}\n"
        f"    max    = {vals.max():.3f}\n"
    )
    click.echo(summary)
    with open(summary_txt, 'w') as fh:
        fh.write(summary)

    # Histogram
    upper = float(max_distance) if max_distance > 0 else float(
        np.percentile(vals, 99))
    upper = max(upper, bin_width)
    bins = np.arange(0.0, upper + bin_width, bin_width)
    clipped = vals[vals <= upper]
    n_clipped = vals.size - clipped.size

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(clipped, bins=bins, edgecolor='black', alpha=0.85)
    ax.axvline(float(np.median(vals)), color='red', linestyle='--',
               linewidth=1.5, label=f'median = {np.median(vals):.2f}')
    ax.axvline(float(vals.mean()), color='orange', linestyle=':',
               linewidth=1.5, label=f'mean = {vals.mean():.2f}')
    ax.set_xlabel('Consecutive peak-to-peak distance (px)')
    ax.set_ylabel('Count')
    title = (f'Consecutive Scan_Intensity peak distances  '
             f"(n={vals.size} from {(per_track_df['n_distances'] > 0).sum()} tracks, "
             f"{per_track_df['image_name'].nunique()} images)\n"
             f'intensity ≥ {peak_min_intensity}, prominence ≥ {peak_min_prominence}')
    if n_clipped:
        title += f'  ·  {n_clipped} distances > {upper:.1f} px not shown'
    ax.set_title(title)
    ax.set_xlim(0, upper)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(hist_png, dpi=150)
    plt.close(fig)

    click.echo("Wrote:")
    click.echo(f"  {hist_png}")
    click.echo(f"  {distances_csv}")
    click.echo(f"  {per_track_csv}")
    click.echo(f"  {summary_txt}")


if __name__ == '__main__':
    main()
