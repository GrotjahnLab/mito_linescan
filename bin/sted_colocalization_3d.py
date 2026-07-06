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
     percentile-normalized [0,1] intensity. The radial averages are
     restricted to voxels INSIDE a separately-thresholded mito mask
     (--radial-scan-mito-threshold-percentile, default 99 + dilation), so
     "intensity vs distance" reflects mito interior only — voxels outside
     the mito (background, cytosol) are excluded from both the sum and the
     count, and shells with no in-mask voxels yield NaN. Save the per-image
     long-format CSV and (across all images) a pooled CSV + a 3-panel mean
     ± SEM plot.

Per-image outputs (in {output_dir}/{basename}{run_name}/):
  {basename}_{mtdna|mito|septin}.mrc   single-channel MRC exports (toggleable)
  {basename}_analysis.png              the original 2×3 central-slice viz
  {basename}_histogram.png             per-channel intensity histogram
  {basename}_radial_profiles.csv       long-format: image_name, nucleoid_id,
                                       z, y, x, on_mito, distance_um,
                                       mtdna_intensity, mito_intensity,
                                       septin_intensity
  {basename}_half_max_distances.csv    wide-format, one row per nucleoid:
                                       image_name, nucleoid_id, z, y, x,
                                       on_mito, mtdna_half_max_um,
                                       mito_half_max_um, septin_half_max_um
                                       (distance where each channel's
                                       radial profile first crosses half
                                       of its peak, linearly interpolated
                                       between the two bracketing bins)
  {basename}_radial_profiles.png       per-image 3-panel mean ± SEM plot
                                       (mtDNA / mito / septin), same render
                                       as the pooled plot but over this
                                       image's nucleoids only
  per_nucleoid/{basename}_{run_name}_nucleoid_{id:03d}_z{z}_y{y}_x{x}.{png,svg}
                                       2x3 plot per detected mtDNA nucleoid:
                                         top row    = radial profile in
                                                      mtDNA / mito / septin
                                                      (single-sample lines)
                                         bottom row = black-background
                                                      true-color crops at
                                                      the centroid's Z slice:
                                          [under mtDNA]  mtDNA+septin
                                          [under mito]   mtDNA alone
                                          [under septin] septin alone
                                       Each bottom panel has the scan-time
                                       mito mask drawn as a magenta contour
                                       (mito outline), a yellow '+' at the
                                       centroid, and a dashed yellow scan-
                                       radius circle. Each nucleoid is
                                       saved as BOTH PNG (raster) and SVG
                                       (vector, editable in
                                       Illustrator/Inkscape). Lives in a
                                       subdirectory to keep the per-image
                                       dir tidy when an image has many
                                       nucleoids.

Pooled outputs (in {output_dir}/):
  analysis_results.csv                 per-image Area1/Area2/ratio summary
                                       (same schema as the original script)
  septin_ratios_boxplot.png            boxplot of Area1/mito vs Area2/mito
  puncta_radial_profiles_pooled.csv    every per-image radial profile concatenated
  puncta_radial_profiles.png / .svg    three-panel mean ± SEM plot
                                       (mtDNA / mito / septin) with a
                                       vertical dashed line at each
                                       channel's half-max distance
                                       computed on the mean curve
  puncta_half_max_distances_pooled.csv every per-image half-max CSV concatenated

Distance is reported in MICRONS. The radial profile uses the true
anisotropic Euclidean distance using the voxel sizes supplied via the
`voxel_size_*_nm` config keys (defaults: 25, 25, 50 nm — typical for 3D
STED). Bin width = min(vx_um, vy_um), i.e. the finest lateral resolution.
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


def _radial_profile_3d(image, z0, y0, x0, radius_voxels,
                       vx_um, vy_um, vz_um, mask=None):
    """Mean intensity vs PHYSICAL (micron) distance from the centroid.

    Returns a length-(radius_voxels+1) float32 array. Bin `r` corresponds to
    physical distance `r * bin_width_um` µm, where `bin_width_um = min(vx, vy)`
    (typically the lateral pixel size, the finest resolution). For each
    voxel in the cubic ±radius_voxels window, the physical Euclidean
    distance is

        d_um = sqrt((Δz * vz_um)² + (Δy * vy_um)² + (Δx * vx_um)²)

    which correctly accounts for anisotropic voxel spacing — a unit step in
    Z covers a different physical distance than a unit step in XY when
    vz_um ≠ vx_um.

    Voxels with `round(d_um / bin_width_um) > radius_voxels` are dropped
    (they're beyond the physical scan radius even if they fall inside the
    cubic voxel window). Shells with no in-mask voxels yield NaN. NaN
    handling at the pandas pooling stage is unchanged.
    """
    Z, H, W = image.shape
    z_min = max(0, int(z0) - radius_voxels)
    z_max = min(Z, int(z0) + radius_voxels + 1)
    y_min = max(0, int(y0) - radius_voxels)
    y_max = min(H, int(y0) + radius_voxels + 1)
    x_min = max(0, int(x0) - radius_voxels)
    x_max = min(W, int(x0) + radius_voxels + 1)

    if z_max <= z_min or y_max <= y_min or x_max <= x_min:
        return np.full(radius_voxels + 1, np.nan, dtype=np.float32)

    sub = image[z_min:z_max, y_min:y_max, x_min:x_max].astype(np.float32)
    zz, yy, xx = np.mgrid[z_min:z_max, y_min:y_max, x_min:x_max]

    # Physical Euclidean distance per voxel, in microns. Anisotropic.
    d_um = np.sqrt(
        ((zz - z0) * vz_um) ** 2 +
        ((yy - y0) * vy_um) ** 2 +
        ((xx - x0) * vx_um) ** 2
    ).ravel()

    # Bin into integer multiples of bin_width_um (= min lateral voxel size,
    # typically vx for vx == vy). Bins 0..radius_voxels span 0..max_distance_um.
    bin_width_um = float(min(vx_um, vy_um))
    if bin_width_um <= 0:
        return np.full(radius_voxels + 1, np.nan, dtype=np.float32)
    d_int = np.rint(d_um / bin_width_um).astype(np.int64)

    valid = d_int <= radius_voxels
    if mask is not None:
        sub_mask = mask[z_min:z_max, y_min:y_max, x_min:x_max].astype(bool).ravel()
        valid = valid & sub_mask

    d_int = d_int[valid]
    vals = sub.ravel()[valid]

    n_bins = radius_voxels + 1
    counts = np.bincount(d_int, minlength=n_bins)[:n_bins]
    sums = np.bincount(d_int, weights=vals, minlength=n_bins)[:n_bins]
    with np.errstate(invalid='ignore', divide='ignore'):
        profile = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return profile.astype(np.float32)


def compute_radial_profiles_3d(
    coords, on_mito,
    mtdna_n, mito_n, septin_n,
    radius_voxels, basename,
    vx_um, vy_um, vz_um,
    scan_mask=None,
):
    """Per-nucleoid 3D radial intensity profiles in all three channels.

    Returns a long-format DataFrame with one row per (nucleoid, distance_um):
        image_name, nucleoid_id, z, y, x, on_mito, distance_um,
        mtdna_intensity, mito_intensity, septin_intensity

    `distance_um` is the physical distance from the centroid in MICRONS,
    accounting for anisotropic voxel spacing (vx_um, vy_um, vz_um) — see
    `_radial_profile_3d`. There are `radius_voxels + 1` bins per nucleoid;
    bin `r` corresponds to `r * min(vx_um, vy_um)` µm.

    `scan_mask` (optional, same shape as the channel volumes) restricts
    every shell average to voxels INSIDE the mask. Typically the caller
    passes a stricter mito mask here so the radial profile reports only
    intensity inside the mitochondrion.
    """
    cols = ['image_name', 'nucleoid_id', 'z', 'y', 'x', 'on_mito',
            'distance_um',
            'mtdna_intensity', 'mito_intensity', 'septin_intensity']
    if coords.shape[0] == 0:
        return pd.DataFrame(columns=cols)

    n = coords.shape[0]
    nb = int(radius_voxels) + 1
    bin_width_um = float(min(vx_um, vy_um))
    distance_axis_um = np.arange(nb, dtype=np.float32) * bin_width_um

    mt = np.empty((n, nb), dtype=np.float32)
    mi = np.empty((n, nb), dtype=np.float32)
    sp = np.empty((n, nb), dtype=np.float32)
    for i in range(n):
        z, y, x = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
        mt[i] = _radial_profile_3d(mtdna_n, z, y, x, int(radius_voxels),
                                    vx_um, vy_um, vz_um, mask=scan_mask)
        mi[i] = _radial_profile_3d(mito_n, z, y, x, int(radius_voxels),
                                    vx_um, vy_um, vz_um, mask=scan_mask)
        sp[i] = _radial_profile_3d(septin_n, z, y, x, int(radius_voxels),
                                    vx_um, vy_um, vz_um, mask=scan_mask)

    nucleoid_id = np.repeat(np.arange(n, dtype=np.int64), nb)
    distance_um = np.tile(distance_axis_um, n)
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
        'distance_um': distance_um,
        'mtdna_intensity': mt.ravel(),
        'mito_intensity': mi.ravel(),
        'septin_intensity': sp.ravel(),
    })


# -------- half-max distance ------------------------------------------------

def _half_max_distance(distances, intensities):
    """First distance at which the profile crosses below max/2.

    Given paired (distance, intensity) arrays, find the smallest distance
    where the intensity has fallen below half of the profile's peak value.
    The crossing is linearly interpolated between the two bracketing bins
    for sub-bin precision.

    Returns:
        Interpolated crossing distance in the units of `distances`, or NaN
        if the profile has fewer than 2 valid (non-NaN) samples, its peak
        is ≤ 0, or it never falls below max/2 within the sampled range.
    """
    d = np.asarray(distances, dtype=np.float64)
    i = np.asarray(intensities, dtype=np.float64)
    mask = ~np.isnan(i)
    if mask.sum() < 2:
        return np.nan
    d = d[mask]
    i = i[mask]
    max_i = float(np.max(i))
    if max_i <= 0.0:
        return np.nan
    half = max_i / 2.0
    below = i < half
    if not np.any(below):
        return np.nan
    first_below = int(np.argmax(below))
    if first_below == 0:
        return float(d[0])
    d1, d2 = float(d[first_below - 1]), float(d[first_below])
    i1, i2 = float(i[first_below - 1]), float(i[first_below])
    if i1 == i2:
        return d1
    frac = (i1 - half) / (i1 - i2)
    return d1 + frac * (d2 - d1)


def compute_half_max_distances(radial_df):
    """Per-nucleoid half-maximum distance for each channel.

    Given the long-format radial profile DataFrame produced by
    `compute_radial_profiles_3d`, compute one row per unique nucleoid with
    the half-max distance in each of the three channels. Rows are ordered
    by (image_name, nucleoid_id).

    Returned schema:
        image_name, nucleoid_id, z, y, x, on_mito,
        mtdna_half_max_um, mito_half_max_um, septin_half_max_um
    """
    cols = ['image_name', 'nucleoid_id', 'z', 'y', 'x', 'on_mito',
            'mtdna_half_max_um', 'mito_half_max_um', 'septin_half_max_um']
    if len(radial_df) == 0:
        return pd.DataFrame(columns=cols)

    rows = []
    for (image, nid), grp in radial_df.groupby(['image_name', 'nucleoid_id']):
        d = grp['distance_um'].values
        rows.append({
            'image_name': image,
            'nucleoid_id': int(nid),
            'z': int(grp['z'].iloc[0]),
            'y': int(grp['y'].iloc[0]),
            'x': int(grp['x'].iloc[0]),
            'on_mito': bool(grp['on_mito'].iloc[0]),
            'mtdna_half_max_um': _half_max_distance(d, grp['mtdna_intensity'].values),
            'mito_half_max_um': _half_max_distance(d, grp['mito_intensity'].values),
            'septin_half_max_um': _half_max_distance(d, grp['septin_intensity'].values),
        })
    return pd.DataFrame(rows, columns=cols)


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


def _clim_percentile(arr, lo=1.0, hi=99.5):
    """Robust (vmin, vmax) for imshow display via percentiles of non-zero
    pixels.

    Edge cases handled:
      - All-zero crop -> (0, 1) so imshow doesn't crash.
      - Only one (or very few, all-equal) nonzero values -> stretch the
        range from 0 up to that value rather than collapsing to a near-zero
        window. Important for sparse channels where there are only a handful
        of bright voxels in the local crop; otherwise the RGB composite
        renders them as black.
    """
    vals = arr[arr > 0]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(vals, lo))
    vmax = float(np.percentile(vals, hi))
    if vmax <= vmin:
        if vmax > 0:
            return 0.0, vmax
        return 0.0, 1.0
    return vmin, vmax


def _make_rgb_composite(yellow_ch=None, cyan_ch=None, magenta_ch=None):
    """Build a true-color RGB image (black background) from up to three
    fluorescence channels rendered in SECONDARY (CMY) colors.

    Mapping:
      yellow_ch  -> R + G  (so signal-only pixels look pure yellow)
      cyan_ch    -> G + B  (so signal-only pixels look pure cyan)
      magenta_ch -> R + B  (so signal-only pixels look pure magenta)

    Where two channels overlap, the additive combination of their secondary
    colors produces a clearly different hue, which is much easier to read
    than overlapping primary colors:
      yellow + cyan    -> white     (R=1, G=2->clipped to 1, B=1)
      yellow + magenta -> orange/red-leaning
      cyan + magenta   -> magenta/blue-leaning

    Each input is independently percentile-clipped to [0, 1] before being
    additively summed into its two color channels, then the whole image is
    clipped to [0, 1]. Missing channels stay at zero (black), so the
    background is genuine black rather than the white floor that sequential
    matplotlib colormaps like 'Blues' / 'Reds' produce at zero intensity.

    Returns a float32 (H, W, 3) array suitable for `imshow`.
    """
    shape = None
    for ch in (yellow_ch, cyan_ch, magenta_ch):
        if ch is not None:
            shape = ch.shape
            break
    if shape is None:
        return np.zeros((1, 1, 3), dtype=np.float32)

    rgb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)

    def _add_to_slots(ch, *slots):
        if ch is None:
            return
        vmin, vmax = _clim_percentile(ch)
        denom = max(vmax - vmin, 1e-9)
        v = np.clip((ch.astype(np.float32) - vmin) / denom, 0.0, 1.0)
        for s in slots:
            rgb[..., s] += v

    _add_to_slots(yellow_ch, 0, 1)   # R + G
    _add_to_slots(cyan_ch, 1, 2)     # G + B
    _add_to_slots(magenta_ch, 0, 2)  # R + B

    return np.clip(rgb, 0.0, 1.0)


def render_single_nucleoid_radial(
    out_path, single_df, radius, label,
    mtdna_vol, mito_vol, septin_vol, z, y, x,
    scan_mask=None,
):
    """2x3 plot for ONE nucleoid:
       Top row    = mtDNA / mito / septin radial profile (single sample).
       Bottom row = black-background image crops at the centroid's Z slice,
                    sized to the scan window:
                      under mtDNA profile  : mtDNA (blue) + septin (red)
                      under mito profile   : mtDNA only (blue)
                      under septin profile : septin only (red)

    Every bottom panel is a true-color RGB composite (no sequential cmaps
    with white background) so empty regions are genuine black. On top of
    each, the scan-time mito-mask boundary is drawn as a thick lime contour
    — voxels inside that contour are the ones that actually contributed to
    the radial averages plotted above. A yellow '+' marks the centroid; a
    dashed yellow circle marks the scan radius.

    `*_vol` are the (Z, Y, X) channel arrays used for display only — pass
    the raw channels (pre-mean-threshold) so the visualization isn't biased
    by the zeroing step. `mito_vol` is still accepted for signature
    stability across callers, even though the new bottom row only renders
    mtDNA and septin signal directly (the mito information is conveyed by
    the green mask contour).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # ---- Top row: radial profile per channel -----------------------------
    panels_top = [
        (axes[0, 0], 'mtdna_intensity',  'mtDNA channel',  'gold'),
        (axes[0, 1], 'mito_intensity',   'Mito channel',   'tab:green'),
        (axes[0, 2], 'septin_intensity', 'Septin channel', 'darkcyan'),
    ]
    distances = single_df['distance_um'].values
    max_d_um = float(np.max(distances)) if distances.size else 1.0
    for ax, col, title, color in panels_top:
        y_values = single_df[col].values
        ax.plot(distances, y_values,
                color=color, linewidth=1.6, marker='o', markersize=3)
        # Vertical dashed line at the half-max distance for THIS nucleoid
        # and THIS channel. Text annotation shows the numeric value inline
        # near the top of the line.
        hm = _half_max_distance(distances, y_values)
        if not np.isnan(hm):
            ax.axvline(hm, color=color, linestyle='--',
                       linewidth=1.2, alpha=0.75)
            y_top = np.nanmax(y_values) if np.any(~np.isnan(y_values)) else 1.0
            ax.text(hm, y_top, f' HM={hm:.3f} µm',
                    color=color, fontsize=8, va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='none',
                              alpha=0.7, pad=1))
        ax.set_xlabel('Distance from centroid (µm)')
        ax.set_ylabel('Intensity (normalized)')
        ax.set_title(title)
        ax.set_xlim(0, max_d_um)
        ax.grid(True, alpha=0.3)

    # ---- Bottom row: image-context crops centered on the nucleoid --------
    Z, H, W = mtdna_vol.shape
    z_disp = int(np.clip(z, 0, Z - 1))
    y_min = max(0, int(y) - radius)
    y_max = min(H, int(y) + radius + 1)
    x_min = max(0, int(x) - radius)
    x_max = min(W, int(x) + radius + 1)

    mtdna_crop = mtdna_vol[z_disp, y_min:y_max, x_min:x_max]
    mito_crop = mito_vol[z_disp, y_min:y_max, x_min:x_max]
    septin_crop = septin_vol[z_disp, y_min:y_max, x_min:x_max]

    # The centroid in cropped (local) coordinates. Because we clipped the
    # crop to image bounds, the centroid is not necessarily at (radius, radius)
    # for nucleoids near the volume edge — use the actual offset.
    cy_local = int(y) - y_min
    cx_local = int(x) - x_min

    # Optional: 2D slice of the scan mask over the crop region. Drawn as a
    # green contour on each panel so the user can see which voxels actually
    # contributed to the radial averages plotted above.
    if scan_mask is not None:
        mask_crop = scan_mask[z_disp, y_min:y_max, x_min:x_max]
    else:
        mask_crop = None

    def _draw_panel(ax, title, rgb_img):
        """Display a black-background RGB composite, overlay the mito-mask
        contour (lime green), and add the centroid marker + scan-radius
        circle in yellow."""
        ax.set_facecolor('black')
        ax.imshow(rgb_img, interpolation='nearest')
        if mask_crop is not None and mask_crop.any() and not mask_crop.all():
            ax.contour(mask_crop.astype(float), levels=[0.5],
                       colors='magenta', linewidths=1.8, alpha=0.95)
        ax.plot(cx_local, cy_local, '+',
                markeredgecolor='yellow', markersize=12, markeredgewidth=1.6)
        ax.add_patch(mpatches.Circle(
            (cx_local, cy_local), radius,
            edgecolor='yellow', facecolor='none',
            linewidth=0.8, linestyle='--', alpha=0.8,
        ))
        ax.set_title(f'{title}  (z={z_disp})', fontsize=10, color='black')
        ax.set_xticks([]); ax.set_yticks([])

    # Build true-color RGB composites for the three bottom-row panels.
    # Channels are rendered in CMY (secondary) colors so the overlap of
    # mtDNA (yellow) and septin (cyan) is clearly visible as white:
    #   mtDNA  -> yellow (R + G)
    #   septin -> cyan   (G + B)
    rgb_mtdna_septin = _make_rgb_composite(
        yellow_ch=mtdna_crop, cyan_ch=septin_crop,
    )
    rgb_mtdna_only = _make_rgb_composite(yellow_ch=mtdna_crop)
    rgb_septin_only = _make_rgb_composite(cyan_ch=septin_crop)

    _draw_panel(axes[1, 0], 'mtDNA + septin (mito outline)', rgb_mtdna_septin)
    _draw_panel(axes[1, 1], 'mtDNA (mito outline)',          rgb_mtdna_only)
    _draw_panel(axes[1, 2], 'septin (mito outline)',         rgb_septin_only)

    fig.suptitle(f'Radial profile + image context - {label}')
    fig.tight_layout()
    # Save both PNG (raster, fast to preview) and SVG (vector, editable in
    # Illustrator/Inkscape and infinitely scalable). The SVG path is derived
    # from the PNG path by swapping the extension, so the two files live
    # side-by-side with the same basename.
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    svg_path = os.path.splitext(out_path)[0] + '.svg'
    plt.savefig(svg_path, bbox_inches='tight')
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
        (axes[0], 'mtdna_intensity',  'mtDNA channel',  'gold'),
        (axes[1], 'mito_intensity',   'Mito channel',   'tab:green'),
        (axes[2], 'septin_intensity', 'Septin channel', 'darkcyan'),
    ]
    n_nucleoids = pooled_df[['image_name', 'nucleoid_id']].drop_duplicates().shape[0]
    max_d_um = float(pooled_df['distance_um'].max()) if len(pooled_df) else 1.0
    for ax, col, title, color in panels:
        grouped = pooled_df.groupby('distance_um')[col]
        mean = grouped.mean()
        sem = grouped.sem(ddof=1).fillna(0.0)
        # Half-max distance computed on the AGGREGATE MEAN curve — matches
        # what you'd read visually off the plotted line.
        hm = _half_max_distance(mean.index.values, mean.values)
        line_label = f'n={n_nucleoids} nucleoids'
        if not np.isnan(hm):
            line_label += f'  |  HM = {hm:.3f} µm'
        ax.plot(mean.index, mean.values, color=color, linewidth=1.8,
                label=line_label)
        ax.fill_between(mean.index,
                        (mean - sem).values, (mean + sem).values,
                        color=color, alpha=0.25, linewidth=0)
        # Vertical dashed line at the half-max distance of the mean curve.
        if not np.isnan(hm):
            ax.axvline(hm, color=color, linestyle='--',
                       linewidth=1.2, alpha=0.7)
        ax.set_xlabel('Distance from mtDNA centroid (µm)')
        ax.set_ylabel('Mean intensity (normalized)')
        ax.set_title(title)
        ax.set_xlim(0, max_d_um)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'3D radial intensity profile around mtDNA nucleoids '
                 f'(mean ± SEM, R={max_d_um:.3f} µm)')
    fig.tight_layout()
    # Save both PNG (raster) and SVG (vector) for consistency with the
    # per-nucleoid plots.
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    svg_path = os.path.splitext(out_path)[0] + '.svg'
    plt.savefig(svg_path, bbox_inches='tight')
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
    radial_scan_mito_threshold_percentile,
    radial_scan_mito_dilation,
    voxel_size_x_nm,
    voxel_size_y_nm,
    voxel_size_z_nm,
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

    half_max_df = None  # populated in the else-branch below when we have data
    if centroids.shape[0] == 0:
        radial_df = None
        radial_scan_mask = None
    else:
        # Normalize each channel to [0, 1] over its own voxels so profiles
        # are comparable across images and between channels. Use the raw
        # (pre-mean-threshold) arrays so the normalization isn't biased by
        # the visualization-prep zeroing.
        mtdna_n = _normalize_3d(mtdna_raw)
        mito_n = _normalize_3d(mito_raw)
        septin_n = _normalize_3d(septin_raw)

        # Build a separate, scan-time mito mask (defaults: 99th-percentile
        # threshold + 3-voxel dilation -> strict "definitely inside mito")
        # so the radial average only sees voxels that actually belong to a
        # mitochondrion. This is independent of the Area1/Area2 mito mask;
        # the two analyses use different thresholds by design.
        radial_scan_mask, radial_scan_thr = compute_mito_mask_3d(
            mito_ch,
            radial_scan_mito_threshold_percentile,
            radial_scan_mito_dilation,
        )
        click.echo(
            f"  radial-scan mito mask: thr@"
            f"{radial_scan_mito_threshold_percentile}%={radial_scan_thr:.1f}, "
            f"dilation={radial_scan_mito_dilation}, "
            f"{int(radial_scan_mask.sum())} voxels "
            f"({100.0 * radial_scan_mask.sum() / radial_scan_mask.size:.2f}% of volume)"
        )

        # Voxel sizes: stored in nm in the config, convert to microns here
        # so the rest of the pipeline (distance axis, plot labels, CSV
        # column) speaks in physical microns.
        vx_um = float(voxel_size_x_nm) / 1000.0
        vy_um = float(voxel_size_y_nm) / 1000.0
        vz_um = float(voxel_size_z_nm) / 1000.0

        on_mito = mito_mask_3d[centroids[:, 0], centroids[:, 1], centroids[:, 2]]
        radial_df = compute_radial_profiles_3d(
            centroids, on_mito, mtdna_n, mito_n, septin_n,
            radius_voxels=int(punct_scan_radius), basename=basename,
            vx_um=vx_um, vy_um=vy_um, vz_um=vz_um,
            scan_mask=radial_scan_mask,
        )
        radial_csv = os.path.join(image_out_dir,
                                   f"{basename}_radial_profiles.csv")
        radial_df.to_csv(radial_csv, index=False)
        max_r_um = float(radial_df['distance_um'].max())
        click.echo(f"  wrote radial profiles ({len(radial_df)} rows, "
                   f"R={max_r_um:.3f} µm, bin={min(vx_um, vy_um):.3f} µm) -> "
                   f"{os.path.basename(radial_csv)}")

        # Per-nucleoid half-max distances (wide format, one row per nucleoid).
        # Independent from the radial CSV but derived from it — makes it easy
        # to compare "how localized is mtDNA vs septin around each nucleoid"
        # without pivoting the long-format table yourself.
        half_max_df = compute_half_max_distances(radial_df)
        half_max_csv = os.path.join(image_out_dir,
                                     f"{basename}_half_max_distances.csv")
        half_max_df.to_csv(half_max_csv, index=False)
        click.echo(f"  wrote half-max distances ({len(half_max_df)} nucleoids) "
                   f"-> {os.path.basename(half_max_csv)}")

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
                # Prefix the filename with image basename + run_name so the
                # file is self-identifying when moved or pooled outside its
                # per_nucleoid/ directory.
                out_path = os.path.join(
                    per_nuc_dir,
                    f"{basename}_{run_name}_"
                    f"nucleoid_{int(nuc_id):03d}_z{z}_y{y}_x{x}.png"
                )
                try:
                    # Pass the RAW channel volumes (pre-mean-threshold) so
                    # the image-context crops aren't biased by the
                    # visualization zeroing step. The radial profile values
                    # in `sub` are unchanged. Also pass the scan-time mito
                    # mask so its boundary is drawn on each crop — voxels
                    # inside that contour are the ones that actually
                    # contributed to the radial averages above.
                    render_single_nucleoid_radial(
                        out_path, sub, int(punct_scan_radius), label,
                        mtdna_raw, mito_raw, septin_raw, z, y, x,
                        scan_mask=radial_scan_mask,
                    )
                    n_ok += 1
                except Exception as exc:
                    n_fail += 1
                    if n_fail <= 3:
                        click.echo(f"  per-nucleoid PNG {nuc_id} failed: {exc}")
            click.echo(f"  wrote {n_ok} per-nucleoid figures (PNG + SVG each) "
                       f"to per_nucleoid/ ({n_fail} failed)" if n_fail
                       else f"  wrote {n_ok} per-nucleoid figures "
                            f"(PNG + SVG each) to per_nucleoid/")

    return area_row, radial_df, half_max_df


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


def _save_half_max_pooled(per_image_half_max_dfs, output_dir):
    """Pool per-image half-max DataFrames into
    `puncta_half_max_distances_pooled.csv` and print a brief console summary
    so the user has an at-a-glance mean/median for each channel."""
    if not per_image_half_max_dfs:
        click.echo("No half-max distances to pool; skipping.")
        return
    pooled = pd.concat(per_image_half_max_dfs, ignore_index=True)
    out = os.path.join(output_dir, 'puncta_half_max_distances_pooled.csv')
    pooled.to_csv(out, index=False)
    click.echo(f"Wrote pooled half-max CSV -> {out}")

    def _summary(col):
        vals = pooled[col].dropna().values
        if vals.size == 0:
            return f'  {col}: no valid samples'
        return (f'  {col}: n={vals.size}, '
                f'mean={vals.mean():.3f} µm, '
                f'median={float(np.median(vals)):.3f} µm, '
                f'std={vals.std(ddof=1):.3f} µm')
    click.echo('  half-max distance summary (µm):')
    click.echo(_summary('mtdna_half_max_um'))
    click.echo(_summary('mito_half_max_um'))
    click.echo(_summary('septin_half_max_um'))


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
@click.option('--radial-scan-mito-threshold-percentile', default=99, type=float,
              show_default=True,
              help='Percentile of nonzero mito voxels used to build the '
                   'scan-time mito mask. Voxels outside this mask are '
                   'EXCLUDED from the radial intensity averages (so only '
                   'intensity *inside the mitochondrion* contributes). '
                   'Defaults to 99 (strict) and is independent of '
                   '--mito-threshold-percentile, which drives Area1/Area2.')
@click.option('--radial-scan-mito-dilation', default=3, type=int,
              show_default=True,
              help='Binary-dilation iterations applied to the scan-time mito '
                   'mask after percentile thresholding.')
@click.option('--voxel-size-x-nm', default=25.0, type=float, show_default=True,
              help='Lateral pixel size in nm (X axis). Reported distances '
                   'are converted to microns using these voxel sizes.')
@click.option('--voxel-size-y-nm', default=25.0, type=float, show_default=True,
              help='Lateral pixel size in nm (Y axis); usually equal to X.')
@click.option('--voxel-size-z-nm', default=50.0, type=float, show_default=True,
              help='Axial step in nm (Z axis). Typically larger than the '
                   'lateral pixel size for STED data — the radial profile '
                   'uses the true anisotropic Euclidean distance.')
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
         radial_scan_mito_threshold_percentile, radial_scan_mito_dilation,
         voxel_size_x_nm, voxel_size_y_nm, voxel_size_z_nm,
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
    click.echo(f"  radial-scan mito mask: "
               f"thr@{radial_scan_mito_threshold_percentile}%, "
               f"dilation={radial_scan_mito_dilation}")
    click.echo(f"  voxel size: x={voxel_size_x_nm:.1f} nm, "
               f"y={voxel_size_y_nm:.1f} nm, z={voxel_size_z_nm:.1f} nm "
               f"(distances reported in µm)")

    area_rows = []
    radial_dfs = []
    half_max_dfs = []
    for image_path in image_list:
        try:
            area_row, radial_df, half_max_df = process_one_image_3d(
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
                radial_scan_mito_threshold_percentile=radial_scan_mito_threshold_percentile,
                radial_scan_mito_dilation=radial_scan_mito_dilation,
                voxel_size_x_nm=voxel_size_x_nm,
                voxel_size_y_nm=voxel_size_y_nm,
                voxel_size_z_nm=voxel_size_z_nm,
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
        if half_max_df is not None and len(half_max_df):
            half_max_dfs.append(half_max_df)

    # Pooled outputs.
    _save_area_results_csv(area_rows, out_dir)
    _save_septin_ratios_boxplot(area_rows, out_dir,
                                 mtdna_threshold_percentile, mtdna_dilation)
    _save_radial_profile_outputs_3d(radial_dfs, out_dir,
                                     radius=int(punct_scan_radius))
    _save_half_max_pooled(half_max_dfs, out_dir)
    click.echo(f"\nProcessed {len(image_list)} TIFF(s). Outputs in {out_dir}")


if __name__ == '__main__':
    main()
