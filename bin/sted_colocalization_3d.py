#!/usr/bin/env python3

"""
3D STED colocalization analysis.

Per multi-channel 3D TIFF in --input-dir:

  1. Read the image as (Z, C, Y, X) (auto-reshape from 3D if needed).
  2. Threshold each channel by its mean (below-mean -> 0). This is a
     visualization prep, not the binary mask, and matches what the original
     stand-alone script did.
  3. Optionally save each channel as a single-channel MRC file.
  4. Build 3D binary masks:
       - mito_mask_3d   : mito channel > Pth percentile, then dilated.
       - mtdna_mask_3d  : mtDNA channel > Pth percentile.
       - septin_mask_3d : septin channel > Pth percentile.
     Then derive:
       - area1_mask_3d  : (mtdna_mask dilated K times) ∩ mito_mask
                          ('within K voxels of mtDNA, inside mito')
       - area2_mask_3d  : mito_mask \\ area1_mask_3d
                          ('inside mito, away from mtDNA')
     Same logic, same defaults, as the original script.
  5. Compute per-area septin intensity stats:
       avg_septin_area1, avg_septin_area2, avg_septin_mito,
       avg_septin_outside_mito, Area1/Area2 and Area1/mito ratios.
  6. NEW: Find mtDNA nucleoid centroids (connected components of
     mtdna_mask_3d, dropping ones below --min-nucleoid-voxels). For each
     centroid, compute 3D radial intensity profiles (spherical shells out to
     --punct-scan-radius voxels) in ALL THREE channels using
     percentile-normalized [0,1] intensity. Save the per-image long-format
     CSV and (across all images) a pooled CSV + a 3-panel mean ± SEM plot.

Per-image outputs (in {output_dir}/{basename}{run_name}/):
  {basename}_{mtdna|mito|septin}.mrc   single-channel MRC exports (toggleable)
  {basename}_analysis.png              the original 2×3 central-slice viz
  {basename}_histogram.png             per-channel intensity histogram
  {basename}_radial_profiles.csv       long-format: image_name, nucleoid_id,
                                       z, y, x, on_mito, distance,
                                       mtdna_intensity, mito_intensity,
                                       septin_intensity
  {basename}_radial_profiles.png       per-image 3-panel mean ± SEM plot
                                       (mtDNA / mito / septin), same render
                                       as the pooled plot but over this
                                       image's nucleoids only
  per_nucleoid/nucleoid_{id:03d}_z{z}_y{y}_x{x}.png
                                       one 3-panel radial profile plot per
                                       detected mtDNA nucleoid (no error
                                       band — single sample); subdirectory
                                       to keep the per-image dir tidy when
                                       a single image has many nucleoids

Pooled outputs (in {output_dir}/):
  analysis_results.csv                 per-image Area1/Area2/ratio summary
                                       (same schema as the original script)
  septin_ratios_boxplot.png            boxplot of Area1/mito vs Area2/mito
  puncta_radial_profiles_pooled.csv    every per-image radial profile concatenated
  puncta_radial_profiles.png           three-panel mean ± SEM plot
                                       (mtDNA / mito / septin)

Distance is in voxels and assumes isotropic spacing. If your acquisition is
anisotropic (Z step ≠ XY pixel), scale the profile axis offline or call
`_radial_profile_3d` with a re-scaled coordinate grid.
"""

import csv
import glob
import os

import click
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage import measure


# -------- filename helpers (match the puncta_nn_scan convention) -----------

def _strip_tiff_ext(raw_name):
    """Strip the actual TIFF extension only (handles `.ome.tif/.ome.tiff`).

    Matches the rule used in puncta_nn_scan / mito_protein_line_scanner so
    that per-image directory names line up across workflows.
    """
    name_lower = raw_name.lower()
    if name_lower.endswith('.ome.tif'):
        return raw_name[:-len('.ome.tif')]
    if name_lower.endswith('.ome.tiff'):
        return raw_name[:-len('.ome.tiff')]
    if name_lower.endswith('.tiff'):
        return raw_name[:-len('.tiff')]
    if name_lower.endswith('.tif'):
        return raw_name[:-len('.tif')]
    return os.path.splitext(raw_name)[0]


# -------- IO + channel separation -----------------------------------------

def read_3d_tiff(path, expected_channels=3):
    """Load a 3D STED TIFF and return a list of (Z, Y, X) arrays, one per
    channel.

    Accepts:
      - 4D `(Z, C, Y, X)` directly.
      - 3D `(Z*C, Y, X)` where Z*C is a multiple of expected_channels (we
        reshape under the assumption that channels are interleaved across Z).

    Raises ValueError for anything else.
    """
    with tifffile.TiffFile(path) as tif:
        data = tif.asarray()

    if data.ndim == 4:
        z_slices, num_channels, height, width = data.shape
        channel_data = [data[:, ch, :, :] for ch in range(num_channels)]
    elif data.ndim == 3:
        total, height, width = data.shape
        if total % expected_channels != 0:
            raise ValueError(
                f"Total slices ({total}) is not divisible by "
                f"{expected_channels} channels for {path}"
            )
        z_slices = total // expected_channels
        num_channels = expected_channels
        reshaped = data.reshape(z_slices, num_channels, height, width)
        channel_data = [reshaped[:, ch, :, :] for ch in range(num_channels)]
    else:
        raise ValueError(f"Unexpected TIFF shape for {path}: {data.shape}")

    return channel_data, (z_slices, num_channels, height, width)


def threshold_above_mean(ch):
    """Return a copy of `ch` with everything ≤ mean(ch) set to zero.

    This is the visualization preprocessing the original script did. It's not
    the binary mask — that's done later via percentile thresholds.
    """
    mean_val = float(np.mean(ch))
    out = ch.copy()
    out[ch <= mean_val] = 0
    return out, mean_val


def save_channel_mrcs(output_path, basename, channel_arrays, channel_names):
    """Write each (Z, Y, X) channel array as a single-channel float32 MRC."""
    for ch_name, ch_data in zip(channel_names, channel_arrays):
        out_file = os.path.join(output_path,
                                f"{basename}_{ch_name}.mrc")
        with mrcfile.new(out_file, overwrite=True) as mrc:
            mrc.set_data(ch_data.astype(np.float32))


# -------- 3D masks + Area1/Area2 (preserves original behavior) -------------

def compute_mito_mask_3d(mito_ch, threshold_percentile, dilation):
    """Build the 3D mito binary mask: threshold at the Pth percentile of the
    non-zero voxels, then binary-dilate `dilation` times so the mask captures
    the immediate periphery (matches the original script)."""
    nonzero = mito_ch[mito_ch > 0]
    threshold = float(np.percentile(nonzero, threshold_percentile)) if nonzero.size else 0.0
    mask = mito_ch > threshold
    if dilation > 0:
        mask = ndimage.binary_dilation(mask, iterations=int(dilation))
    return mask, threshold


def compute_percentile_mask_3d(ch, percentile):
    """Generic helper: binary mask at Pth percentile of nonzero voxels."""
    nonzero = ch[ch > 0]
    threshold = float(np.percentile(nonzero, percentile)) if nonzero.size else 0.0
    return ch > threshold, threshold


def compute_area_masks(mtdna_mask, mito_mask, mtdna_dilation):
    """Area1 = (mtDNA dilated K times) ∩ mito_mask.
       Area2 = mito_mask \\ Area1.

    Matches the original script's notion of 'within K voxels of mtDNA' vs
    'inside mito, away from mtDNA'.
    """
    area1 = ndimage.binary_dilation(mtdna_mask, iterations=int(mtdna_dilation)) & mito_mask
    area2 = mito_mask & (~area1)
    return area1, area2


def area_septin_stats(septin_ch, area1, area2, mito_mask):
    """Replicates the original script's per-area average septin intensities
    and the Area1/Area2 ratio."""
    in_area1 = septin_ch[area1]
    in_area2 = septin_ch[area2]
    in_mito = septin_ch[mito_mask]
    outside_mito = septin_ch[~mito_mask]

    avg_a1 = float(np.mean(in_area1)) if in_area1.size else 0.0
    avg_a2 = float(np.mean(in_area2)) if in_area2.size else 0.0
    avg_mito = float(np.mean(in_mito)) if in_mito.size else 0.0
    avg_out = float(np.mean(outside_mito)) if outside_mito.size else 0.0
    ratio = (avg_a1 / avg_a2) if avg_a2 > 0 else float('nan')
    return {
        'avg_septin_area1': avg_a1,
        'avg_septin_area2': avg_a2,
        'avg_septin_mito': avg_mito,
        'avg_septin_outside_mito': avg_out,
        'ratio_area1_area2': ratio,
    }


# -------- 3D radial intensity profile around mtDNA centroids ---------------

def find_nucleoid_centroids(mtdna_mask_3d, min_voxels):
    """Connected-component centroids of the 3D mtDNA binary mask.

    Drops components below `min_voxels`. Returns an (N, 3) array of (z, y, x)
    centroids in image coordinates (floats rounded to ints for indexing).
    """
    labeled = measure.label(mtdna_mask_3d, connectivity=3)
    if labeled.max() == 0:
        return np.empty((0, 3), dtype=np.int64)
    props = measure.regionprops(labeled)
    coords = []
    for p in props:
        if p.area >= int(min_voxels):
            z, y, x = p.centroid
            coords.append((int(round(z)), int(round(y)), int(round(x))))
    if not coords:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(coords, dtype=np.int64)


def _normalize_3d(ch, mask=None):
    """Normalize a 3D array to [0, 1] using percentile clipping on `mask`
    (or all voxels if mask is None). Mirrors `_normalize_for_puncta` in the
    2D pipeline so profiles in 2D and 3D are comparable in spirit."""
    arr = np.asarray(ch, dtype=np.float32)
    pool = arr[mask] if (mask is not None and mask.any()) else arr
    p_lo = float(np.percentile(pool, 1))
    p_hi = float(np.percentile(pool, 99.9))
    denom = max(p_hi - p_lo, 1e-9)
    return np.clip((arr - p_lo) / denom, 0.0, 1.0).astype(np.float32)


def _radial_profile_3d(image, z0, y0, x0, radius):
    """Mean intensity vs integer 3D distance from `(z0, y0, x0)` to `radius`.

    Returns a length-(radius+1) float32 array. Bin `r` contains voxels whose
    Euclidean distance to the center, rounded to the nearest integer, equals
    `r`. NaN at radii whose shell has no voxels inside the volume bounds.

    Same idea as the 2D `_radial_profile` but with a third axis. We compute
    one volumetric distance map, ravel it, then `np.bincount` over rounded
    distances for the per-shell mean.
    """
    Z, H, W = image.shape
    z_min = max(0, int(z0) - radius)
    z_max = min(Z, int(z0) + radius + 1)
    y_min = max(0, int(y0) - radius)
    y_max = min(H, int(y0) + radius + 1)
    x_min = max(0, int(x0) - radius)
    x_max = min(W, int(x0) + radius + 1)

    if z_max <= z_min or y_max <= y_min or x_max <= x_min:
        return np.full(radius + 1, np.nan, dtype=np.float32)

    sub = image[z_min:z_max, y_min:y_max, x_min:x_max].astype(np.float32)
    zz, yy, xx = np.mgrid[z_min:z_max, y_min:y_max, x_min:x_max]
    d = np.sqrt((zz - z0) ** 2 + (yy - y0) ** 2 + (xx - x0) ** 2).ravel()
    d_int = np.rint(d).astype(np.int64)
    valid = d_int <= radius
    d_int = d_int[valid]
    vals = sub.ravel()[valid]

    n_bins = radius + 1
    counts = np.bincount(d_int, minlength=n_bins)[:n_bins]
    sums = np.bincount(d_int, weights=vals, minlength=n_bins)[:n_bins]
    with np.errstate(invalid='ignore', divide='ignore'):
        profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return profile.astype(np.float32)


def compute_radial_profiles_3d(
    coords, on_mito,
    mtdna_n, mito_n, septin_n,
    radius, basename,
):
    """Per-nucleoid 3D radial intensity profiles in all three channels.

    Returns a long-format DataFrame with one row per (nucleoid, distance):
        image_name, nucleoid_id, z, y, x, on_mito, distance,
        mtdna_intensity, mito_intensity, septin_intensity
    """
    cols = ['image_name', 'nucleoid_id', 'z', 'y', 'x', 'on_mito', 'distance',
            'mtdna_intensity', 'mito_intensity', 'septin_intensity']
    if coords.shape[0] == 0:
        return pd.DataFrame(columns=cols)

    n = coords.shape[0]
    nb = radius + 1
    mt = np.empty((n, nb), dtype=np.float32)
    mi = np.empty((n, nb), dtype=np.float32)
    sp = np.empty((n, nb), dtype=np.float32)
    for i in range(n):
        z, y, x = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
        mt[i] = _radial_profile_3d(mtdna_n, z, y, x, radius)
        mi[i] = _radial_profile_3d(mito_n, z, y, x, radius)
        sp[i] = _radial_profile_3d(septin_n, z, y, x, radius)

    nucleoid_id = np.repeat(np.arange(n, dtype=np.int64), nb)
    distance = np.tile(np.arange(nb, dtype=np.int64), n)
    z_rep = np.repeat(coords[:, 0].astype(np.int64), nb)
    y_rep = np.repeat(coords[:, 1].astype(np.int64), nb)
    x_rep = np.repeat(coords[:, 2].astype(np.int64), nb)
    on_rep = np.repeat(np.asarray(on_mito, dtype=bool), nb)

    return pd.DataFrame({
        'image_name': basename,
        'nucleoid_id': nucleoid_id,
        'z': z_rep,
        'y': y_rep,
        'x': x_rep,
        'on_mito': on_rep,
        'distance': distance,
        'mtdna_intensity': mt.ravel(),
        'mito_intensity': mi.ravel(),
        'septin_intensity': sp.ravel(),
    })


# -------- visualization (preserves original 2x3 panel + histogram) ---------

def render_analysis_png(
    out_path, mtdna_ch, mito_ch, septin_ch,
    mtdna_mask, mito_mask, area1, area2, central_z,
):
    """2x3 central-slice panel: matches the original script's plot exactly."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (0,0) mtDNA channel with Area1 overlay
    ax = axes[0, 0]
    im = ax.imshow(mtdna_ch[central_z], cmap='Blues', alpha=0.8)
    ax.contour(area1[central_z], colors='red', linewidths=2, levels=[0.5])
    ax.contourf(area1[central_z].astype(float), levels=[0.5, 1.5],
                colors=['red'], alpha=0.2)
    ax.set_title(f'mtDNA - Central Slice z={central_z}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixels)'); ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Intensity')
    ax.legend(handles=[mpatches.Patch(color='red', alpha=0.2,
                                       label='Area 1 (≤K voxel from mtDNA)')],
              loc='upper right')

    # (0,1) mtDNA binary mask
    ax = axes[0, 1]
    im = ax.imshow(mtdna_mask[central_z].astype(int), cmap='Blues', alpha=0.8)
    ax.set_title(f'mtDNA Binary Mask - z={central_z}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixels)'); ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Binary')

    # (0,2) mito channel
    ax = axes[0, 2]
    im = ax.imshow(mito_ch[central_z], cmap='Greens', alpha=0.8)
    ax.set_title(f'mito - Central Slice z={central_z}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixels)'); ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Intensity')

    # (1,0) mito binary mask
    ax = axes[1, 0]
    im = ax.imshow(mito_mask[central_z].astype(int), cmap='Greens', alpha=0.8)
    ax.set_title(f'mito Binary Mask - z={central_z}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixels)'); ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Binary')

    # (1,1) septin with Area1 + Area2 overlay
    ax = axes[1, 1]
    im = ax.imshow(septin_ch[central_z], cmap='Reds', alpha=0.8)
    ax.contourf(area1[central_z].astype(float), levels=[0.5, 1.5],
                colors=['cyan'], alpha=0.3)
    ax.contourf(area2[central_z].astype(float), levels=[0.5, 1.5],
                colors=['green'], alpha=0.3)
    ax.set_title(f'septin with Regions - z={central_z}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixels)'); ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Intensity')
    ax.legend(handles=[
        mpatches.Patch(color='cyan', alpha=0.3,
                       label='Area 1 (≤K voxel from mtDNA)'),
        mpatches.Patch(color='green', alpha=0.3,
                       label='Area 2 (mito, >K voxel)'),
    ], loc='upper right')

    # (1,2) septin + Area2 only
    ax = axes[1, 2]
    im = ax.imshow(septin_ch[central_z], cmap='Reds', alpha=0.8)
    ax.contourf(area2[central_z].astype(float), levels=[0.5, 1.5],
                colors=['green'], alpha=0.3)
    ax.set_title(f'Area 2 + septin - z={central_z}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X (pixels)'); ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Intensity')
    ax.legend(handles=[mpatches.Patch(color='green', alpha=0.3,
                                       label='Area 2 (mito, >K voxel from mtDNA)')],
              loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_histogram_png(out_path, mtdna_ch, mito_ch, septin_ch):
    """Per-channel intensity histogram, log y-axis. Matches the original."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cap = max(float(mtdna_ch.max()), float(mito_ch.max()), float(septin_ch.max()))
    bins = np.linspace(0, cap, 50)
    ax.hist(mtdna_ch.flatten(), bins=bins, alpha=0.5,
            label='mtDNA', color='blue', edgecolor='blue')
    ax.hist(mito_ch.flatten(), bins=bins, alpha=0.5,
            label='mito', color='green', edgecolor='green')
    ax.hist(septin_ch.flatten(), bins=bins, alpha=0.5,
            label='septin', color='red', edgecolor='red')
    ax.set_xlabel('Intensity'); ax.set_ylabel('Frequency')
    ax.set_title('Intensity Distribution - All Channels (Thresholded)',
                 fontsize=12, fontweight='bold')
    ax.legend(); ax.set_yscale('log'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_ratios_boxplot(out_path, ratios_a1_mito, ratios_a2_mito):
    """Boxplot of Area1/mito and Area2/mito ratios across all images.

    Matches the original script's boxplot output (including the lightblue +
    lightgreen colors)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([ratios_a1_mito, ratios_a2_mito],
                    labels=['Area 1/mito', 'Area 2/mito'],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    ax.set_ylabel('Septin Intensity Ratio', fontsize=12)
    ax.set_title('Distribution of Septin Intensity Ratios Across All Images',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_single_nucleoid_radial(out_path, single_df, radius, label):
    """3-panel mtDNA / mito / septin radial profile for ONE nucleoid.

    Unlike the pooled/mean plot, this shows the raw radial profile of a
    single mtDNA nucleoid: one line per channel with markers at each integer
    distance bin. No error band because there's no variance over nucleoids
    here (it's a single sample). Useful for spotting outlier nucleoids and
    QC'ing individual profiles before pooling.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    panels = [
        (axes[0], 'mtdna_intensity',  'mtDNA channel',  'tab:blue'),
        (axes[1], 'mito_intensity',   'Mito channel',   'tab:green'),
        (axes[2], 'septin_intensity', 'Septin channel', 'tab:red'),
    ]
    distances = single_df['distance'].values
    for ax, col, title, color in panels:
        ax.plot(distances, single_df[col].values,
                color=color, linewidth=1.6, marker='o', markersize=3)
        ax.set_xlabel('Distance from centroid (voxels)')
        ax.set_ylabel('Intensity (normalized)')
        ax.set_title(title)
        ax.set_xlim(0, radius)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Radial profile - {label}')
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_radial_profiles_png(out_path, pooled_df, radius):
    """Three-panel mean ± SEM plot of mtDNA / mito / septin radial profiles.

    Single curve per panel (pooled across all nucleoids). Matches the 2D
    puncta_nn_scan output's visual style but with one group instead of
    on/off, since we did not introduce a puncta-detection-driven on/off
    classification in 3D.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        (axes[0], 'mtdna_intensity',  'mtDNA channel',  'tab:blue'),
        (axes[1], 'mito_intensity',   'Mito channel',   'tab:green'),
        (axes[2], 'septin_intensity', 'Septin channel', 'tab:red'),
    ]
    n_nucleoids = pooled_df[['image_name', 'nucleoid_id']].drop_duplicates().shape[0]
    for ax, col, title, color in panels:
        grouped = pooled_df.groupby('distance')[col]
        mean = grouped.mean()
        sem = grouped.sem(ddof=1).fillna(0.0)
        ax.plot(mean.index, mean.values, color=color, linewidth=1.8,
                label=f'n={n_nucleoids} nucleoids')
        ax.fill_between(mean.index,
                        (mean - sem).values, (mean + sem).values,
                        color=color, alpha=0.25, linewidth=0)
        ax.set_xlabel('Distance from mtDNA centroid (voxels)')
        ax.set_ylabel('Mean intensity (normalized)')
        ax.set_title(title)
        ax.set_xlim(0, radius)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'3D radial intensity profile around mtDNA nucleoids '
                 f'(mean ± SEM, R={radius} voxels)')
    fig.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# -------- per-image driver -------------------------------------------------

def process_one_image_3d(
    image_path,
    *,
    output_dir,
    run_name,
    mtdna_channel,
    mito_channel,
    protein_channel,
    mito_threshold_percentile,
    mito_dilation,
    mtdna_threshold_percentile,
    mtdna_dilation,
    septin_threshold_percentile,
    punct_scan_radius,
    min_nucleoid_voxels,
    save_channel_mrcs_flag,
    save_analysis_png_flag,
    save_histogram_png_flag,
    save_per_nucleoid_png_flag,
):
    """Run the full 3D pipeline for one image and return:
       (area_summary_row, radial_df)

    Either may be None if the image was skipped (bad shape, missing channels).
    """
    raw_name = os.path.basename(image_path)
    basename = _strip_tiff_ext(raw_name)
    image_out_dir = os.path.join(output_dir, basename + run_name)
    os.makedirs(image_out_dir, exist_ok=True)

    click.echo(f"\n{'='*80}\nProcessing: {raw_name}\n{'='*80}")

    try:
        channel_data, (z_slices, num_channels, height, width) = read_3d_tiff(image_path)
    except ValueError as exc:
        click.echo(f"  ERROR: {exc}")
        return None, None

    # Map the requested channel indices to the actual arrays.
    if max(mtdna_channel, mito_channel, protein_channel) >= num_channels:
        click.echo(
            f"  WARNING: image has {num_channels} channels but channels "
            f"mtdna={mtdna_channel}, mito={mito_channel}, "
            f"protein={protein_channel}; skipping."
        )
        return None, None

    mtdna_raw = channel_data[mtdna_channel]
    mito_raw = channel_data[mito_channel]
    septin_raw = channel_data[protein_channel]

    # ---- visualization-prep thresholding (kept identical to original) ----
    mtdna_ch, mtdna_avg = threshold_above_mean(mtdna_raw)
    mito_ch, mito_avg = threshold_above_mean(mito_raw)
    septin_ch, septin_avg = threshold_above_mean(septin_raw)
    click.echo(f"  mean-thresholded channels: "
               f"mtDNA avg={mtdna_avg:.2f}, mito avg={mito_avg:.2f}, "
               f"septin avg={septin_avg:.2f}")

    # ---- Single-channel MRC exports (existing behavior) -----------------
    if save_channel_mrcs_flag:
        # Original used raw (pre-mean-threshold) channels; we mirror that.
        save_channel_mrcs(
            image_out_dir, basename,
            [channel_data[protein_channel], channel_data[mito_channel],
             channel_data[mtdna_channel]],
            ['septin', 'mito', 'mtDNA'],
        )
        click.echo(f"  wrote per-channel MRC files")

    # ---- 3D masks --------------------------------------------------------
    mito_mask_3d, mito_thr = compute_mito_mask_3d(
        mito_ch, mito_threshold_percentile, mito_dilation
    )
    mtdna_mask_3d, mtdna_thr = compute_percentile_mask_3d(
        mtdna_ch, mtdna_threshold_percentile
    )
    septin_mask_3d, septin_thr = compute_percentile_mask_3d(
        septin_ch, septin_threshold_percentile
    )
    click.echo(
        f"  thresholds (Pth percentile): "
        f"mito@{mito_threshold_percentile}={mito_thr:.1f}, "
        f"mtDNA@{mtdna_threshold_percentile}={mtdna_thr:.1f}, "
        f"septin@{septin_threshold_percentile}={septin_thr:.1f}"
    )

    area1_mask_3d, area2_mask_3d = compute_area_masks(
        mtdna_mask_3d, mito_mask_3d, mtdna_dilation
    )
    click.echo(
        f"  mask voxels: mito={int(mito_mask_3d.sum())}, "
        f"mtDNA={int(mtdna_mask_3d.sum())}, "
        f"Area1={int(area1_mask_3d.sum())}, Area2={int(area2_mask_3d.sum())}"
    )

    # ---- Septin stats per area (existing per-image summary row) ---------
    stats = area_septin_stats(septin_ch, area1_mask_3d, area2_mask_3d, mito_mask_3d)
    area_row = {
        'filename': raw_name,
        'z_slices': z_slices,
        'height': height,
        'width': width,
        'area1_voxels': int(area1_mask_3d.sum()),
        'area2_voxels': int(area2_mask_3d.sum()),
        'mito_voxels': int(mito_mask_3d.sum()),
        'outside_mito_voxels': int((~mito_mask_3d).sum()),
        **stats,
    }
    click.echo(
        f"  avg septin: Area1={stats['avg_septin_area1']:.2f}, "
        f"Area2={stats['avg_septin_area2']:.2f}, "
        f"mito={stats['avg_septin_mito']:.2f}, "
        f"outside={stats['avg_septin_outside_mito']:.2f}"
    )

    # ---- Visualizations --------------------------------------------------
    central_z = z_slices // 2
    if save_analysis_png_flag:
        render_analysis_png(
            os.path.join(image_out_dir, f"{basename}_analysis.png"),
            mtdna_ch, mito_ch, septin_ch,
            mtdna_mask_3d, mito_mask_3d, area1_mask_3d, area2_mask_3d,
            central_z,
        )
        click.echo(f"  wrote analysis PNG (central slice z={central_z})")
    if save_histogram_png_flag:
        render_histogram_png(
            os.path.join(image_out_dir, f"{basename}_histogram.png"),
            mtdna_ch, mito_ch, septin_ch,
        )
        click.echo(f"  wrote channel histogram PNG")

    # ---- NEW: per-nucleoid 3D radial intensity profiles -----------------
    centroids = find_nucleoid_centroids(mtdna_mask_3d, min_nucleoid_voxels)
    click.echo(f"  found {centroids.shape[0]} mtDNA nucleoid centroids "
               f"(min_voxels={min_nucleoid_voxels})")

    if centroids.shape[0] == 0:
        radial_df = None
    else:
        # Normalize each channel to [0, 1] over its own voxels so profiles
        # are comparable across images and between channels. Use the raw
        # (pre-mean-threshold) arrays so the normalization isn't biased by
        # the visualization-prep zeroing.
        mtdna_n = _normalize_3d(mtdna_raw)
        mito_n = _normalize_3d(mito_raw)
        septin_n = _normalize_3d(septin_raw)

        on_mito = mito_mask_3d[centroids[:, 0], centroids[:, 1], centroids[:, 2]]
        radial_df = compute_radial_profiles_3d(
            centroids, on_mito, mtdna_n, mito_n, septin_n,
            radius=int(punct_scan_radius), basename=basename,
        )
        radial_csv = os.path.join(image_out_dir,
                                   f"{basename}_radial_profiles.csv")
        radial_df.to_csv(radial_csv, index=False)
        click.echo(f"  wrote radial profiles ({len(radial_df)} rows) -> "
                   f"{os.path.basename(radial_csv)}")

        # Per-image radial profile PNG (mean ± SEM across this image's
        # nucleoids). Reuses the same render helper as the pooled output so
        # the two plots are visually consistent.
        try:
            radial_png = os.path.join(image_out_dir,
                                       f"{basename}_radial_profiles.png")
            render_radial_profiles_png(radial_png, radial_df,
                                        int(punct_scan_radius))
            click.echo(f"  wrote per-image radial profile plot -> "
                       f"{os.path.basename(radial_png)}")
        except Exception as exc:
            click.echo(f"  per-image radial PNG render failed: {exc}")

        # Per-nucleoid radial profile PNGs. One file per detected nucleoid,
        # named with the centroid coordinates so individual nucleoids are
        # easy to cross-reference with the CSV. Lives in a subdirectory so
        # it doesn't clutter the per-image output dir for high-nucleoid
        # counts.
        if save_per_nucleoid_png_flag:
            per_nuc_dir = os.path.join(image_out_dir, 'per_nucleoid')
            os.makedirs(per_nuc_dir, exist_ok=True)
            n_ok = 0
            n_fail = 0
            for nuc_id in radial_df['nucleoid_id'].unique():
                sub = radial_df[radial_df['nucleoid_id'] == nuc_id]
                if sub.empty:
                    continue
                z = int(sub['z'].iloc[0])
                y = int(sub['y'].iloc[0])
                x = int(sub['x'].iloc[0])
                on = bool(sub['on_mito'].iloc[0])
                label = (f"id={int(nuc_id):d} "
                         f"z={z} y={y} x={x} "
                         f"{'on-mito' if on else 'off-mito'}")
                out_path = os.path.join(
                    per_nuc_dir,
                    f"nucleoid_{int(nuc_id):03d}_z{z}_y{y}_x{x}.png"
                )
                try:
                    render_single_nucleoid_radial(
                        out_path, sub, int(punct_scan_radius), label,
                    )
                    n_ok += 1
                except Exception as exc:
                    n_fail += 1
                    if n_fail <= 3:
                        click.echo(f"  per-nucleoid PNG {nuc_id} failed: {exc}")
            click.echo(f"  wrote {n_ok} per-nucleoid radial PNGs to "
                       f"per_nucleoid/ ({n_fail} failed)" if n_fail
                       else f"  wrote {n_ok} per-nucleoid radial PNGs to "
                            f"per_nucleoid/")

    return area_row, radial_df


# -------- pooled outputs ---------------------------------------------------

def _save_area_results_csv(results, output_dir):
    """Write the original analysis_results.csv (per-image Area1/Area2 stats)."""
    if not results:
        return
    out = os.path.join(output_dir, 'analysis_results.csv')
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    click.echo(f"Wrote area results CSV -> {out}")


def _save_septin_ratios_boxplot(results, output_dir, mtdna_threshold, mtdna_dilation):
    """Boxplot of per-image Area1/mito and Area2/mito ratios."""
    a1_over_mito = []
    a2_over_mito = []
    for r in results:
        m = r['avg_septin_mito']
        if m > 0:
            a1_over_mito.append(r['avg_septin_area1'] / m)
            a2_over_mito.append(r['avg_septin_area2'] / m)
    if not a1_over_mito or not a2_over_mito:
        click.echo("Not enough data for septin ratios boxplot.")
        return
    out = os.path.join(output_dir,
                       f"septin_ratios_boxplot_{mtdna_threshold}_{mtdna_dilation}.png")
    render_ratios_boxplot(out, a1_over_mito, a2_over_mito)
    click.echo(f"Wrote septin ratios boxplot -> {out}")
    click.echo(
        f"  Area1/mito: mean={np.mean(a1_over_mito):.3f} "
        f"median={np.median(a1_over_mito):.3f} std={np.std(a1_over_mito):.3f}"
    )
    click.echo(
        f"  Area2/mito: mean={np.mean(a2_over_mito):.3f} "
        f"median={np.median(a2_over_mito):.3f} std={np.std(a2_over_mito):.3f}"
    )


def _save_radial_profile_outputs_3d(per_image_dfs, output_dir, radius):
    """Pool per-image 3D radial profile CSVs + write the mean ± SEM plot."""
    if not per_image_dfs:
        click.echo("No radial profiles to pool; skipping radial outputs.")
        return
    pooled = pd.concat(per_image_dfs, ignore_index=True)
    pooled_csv = os.path.join(output_dir, 'puncta_radial_profiles_pooled.csv')
    pooled.to_csv(pooled_csv, index=False)
    click.echo(f"Wrote pooled radial CSV -> {pooled_csv}")

    render_radial_profiles_png(
        os.path.join(output_dir, 'puncta_radial_profiles.png'),
        pooled, radius,
    )
    click.echo(f"Wrote pooled radial profile plot")


# -------- CLI --------------------------------------------------------------

@click.command()
@click.option('--input-dir', default='', type=str,
              help='Input directory containing 3D multi-channel TIFF files.')
@click.option('--input-pattern', default='*.tif', show_default=True,
              help='Glob pattern for input TIFFs within --input-dir.')
@click.option('--output-dir', default='', type=str,
              help='Where per-image dirs + pooled outputs go. '
                   'Defaults to --input-dir.')
@click.option('--run-name', default='run1', show_default=True,
              help='Suffix appended to each per-image output directory.')
@click.option('--mtdna-channel', default=0, type=int, show_default=True,
              help='0-based mtDNA channel index (default matches the 3D STED '
                   'file convention: channel 0 = mtDNA).')
@click.option('--mito-channel', default=1, type=int, show_default=True,
              help='0-based mito channel index.')
@click.option('--protein-channel', default=2, type=int, show_default=True,
              help='0-based protein/septin channel index.')
@click.option('--mito-threshold-percentile', default=30, type=float, show_default=True,
              help='Percentile of nonzero mito voxels used as the binary '
                   'threshold for the mito mask.')
@click.option('--mito-dilation', default=3, type=int, show_default=True,
              help='Binary-dilation iterations applied to the mito mask.')
@click.option('--mtdna-threshold-percentile', default=99, type=float, show_default=True,
              help='Percentile of nonzero mtDNA voxels used as the binary '
                   'threshold for the mtDNA mask.')
@click.option('--mtdna-dilation', default=3, type=int, show_default=True,
              help='Binary-dilation iterations applied to mtDNA when building '
                   'Area 1 (within K voxels of mtDNA).')
@click.option('--septin-threshold-percentile', default=95, type=float, show_default=True,
              help='Percentile of nonzero septin voxels used as the binary '
                   'threshold for the septin mask (used for stats reporting).')
@click.option('--punct-scan-radius', default=20, type=int, show_default=True,
              help='Radius (voxels) for the 3D radial intensity profile '
                   'around each mtDNA nucleoid centroid.')
@click.option('--min-nucleoid-voxels', default=5, type=int, show_default=True,
              help='Drop mtDNA connected components smaller than this many '
                   'voxels before computing centroids.')
@click.option('--save-channel-mrcs/--no-save-channel-mrcs', default=True,
              help='Write {basename}_{septin|mito|mtDNA}.mrc files per image.')
@click.option('--save-analysis-png/--no-save-analysis-png', default=True,
              help='Write the 2x3 central-slice analysis PNG per image.')
@click.option('--save-histogram-png/--no-save-histogram-png', default=True,
              help='Write the per-channel intensity histogram PNG per image.')
@click.option('--save-per-nucleoid-png/--no-save-per-nucleoid-png', default=True,
              help='Write one radial-profile PNG per detected mtDNA nucleoid '
                   'into {basename}{run_name}/per_nucleoid/. Lots of files '
                   'for high-nucleoid images; disable if you only want the '
                   'aggregated per-image and pooled plots.')
def main(input_dir, input_pattern, output_dir, run_name,
         mtdna_channel, mito_channel, protein_channel,
         mito_threshold_percentile, mito_dilation,
         mtdna_threshold_percentile, mtdna_dilation,
         septin_threshold_percentile,
         punct_scan_radius, min_nucleoid_voxels,
         save_channel_mrcs, save_analysis_png, save_histogram_png,
         save_per_nucleoid_png):
    """Analyze 3D STED colocalization: mito/mtDNA/septin masks + Area1/Area2
    septin densities + per-nucleoid 3D radial intensity profiles."""
    if not input_dir:
        raise click.ClickException("--input-dir is required")

    out_dir = output_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)

    image_list = sorted(
        list(glob.glob(os.path.join(input_dir, input_pattern))) +
        list(glob.glob(os.path.join(input_dir, '*.tiff')))
    )
    # De-duplicate (in case --input-pattern already matches *.tif AND we
    # accidentally double-count under *.tiff).
    seen = set()
    image_list = [p for p in image_list if not (p in seen or seen.add(p))]
    if not image_list:
        click.echo(f"No TIFFs found at {os.path.join(input_dir, input_pattern)}")
        return

    click.echo(f"Found {len(image_list)} TIFF(s).")
    click.echo(f"  mtdna_channel={mtdna_channel}, mito_channel={mito_channel}, "
               f"protein_channel={protein_channel}")
    click.echo(f"  mito thr={mito_threshold_percentile}%, dilation={mito_dilation}")
    click.echo(f"  mtDNA thr={mtdna_threshold_percentile}%, dilation={mtdna_dilation}")
    click.echo(f"  septin thr={septin_threshold_percentile}%")
    click.echo(f"  radial scan radius={punct_scan_radius} voxels, "
               f"min_nucleoid_voxels={min_nucleoid_voxels}")

    area_rows = []
    radial_dfs = []
    for image_path in image_list:
        try:
            area_row, radial_df = process_one_image_3d(
                image_path,
                output_dir=out_dir,
                run_name=run_name,
                mtdna_channel=mtdna_channel,
                mito_channel=mito_channel,
                protein_channel=protein_channel,
                mito_threshold_percentile=mito_threshold_percentile,
                mito_dilation=mito_dilation,
                mtdna_threshold_percentile=mtdna_threshold_percentile,
                mtdna_dilation=mtdna_dilation,
                septin_threshold_percentile=septin_threshold_percentile,
                punct_scan_radius=punct_scan_radius,
                min_nucleoid_voxels=min_nucleoid_voxels,
                save_channel_mrcs_flag=save_channel_mrcs,
                save_analysis_png_flag=save_analysis_png,
                save_histogram_png_flag=save_histogram_png,
                save_per_nucleoid_png_flag=save_per_nucleoid_png,
            )
        except Exception as exc:
            click.echo(f"  ERROR processing {image_path}: {exc}")
            continue

        if area_row is not None:
            area_rows.append(area_row)
        if radial_df is not None and len(radial_df):
            radial_dfs.append(radial_df)

    # Pooled outputs.
    _save_area_results_csv(area_rows, out_dir)
    _save_septin_ratios_boxplot(area_rows, out_dir,
                                 mtdna_threshold_percentile, mtdna_dilation)
    _save_radial_profile_outputs_3d(radial_dfs, out_dir,
                                     radius=int(punct_scan_radius))
    click.echo(f"\nProcessed {len(image_list)} TIFF(s). Outputs in {out_dir}")


if __name__ == '__main__':
    main()
