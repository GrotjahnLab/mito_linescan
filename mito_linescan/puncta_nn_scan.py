#!/usr/bin/env python3

"""
Puncta nearest-neighbor scanner.

For each multi-channel TIFF in --input-dir:

  1. Build (or load) a mitochondrial binary mask using the same ridge-filter
     pipeline as network_line_scan.
       - If `--mito-binary-dir` contains `{basename}_mito_binary.tif` (e.g.
         the export from `network_line_scan --binary-mask-dir-output`), we
         load it and skip mask-making entirely.
       - Otherwise we run the same mask-making flow as network_line_scan:
         (a) get a cell ROI lasso (loaded from `--mask-dir-input` if a saved
         one exists, else drawn interactively and persisted to
         `--mask-dir-output`); (b) normalize + histogram-equalize the mito
         channel inside the ROI; (c) ridge-filter mask GUI if `--use-gui`,
         else `compute_mito_mask_noninteractive` with the supplied CLI/config
         defaults. The resulting binary is then saved into
         `--mito-binary-dir` (when set) so subsequent runs reuse it.

  2. Detect puncta in --protein-channel using a Gaussian smooth followed by
     `skimage.feature.peak_local_max` with --punct-min-distance and either
     --punct-threshold-abs or --punct-threshold-rel (both are honored;
     peak_local_max combines them with max()).

  3. Label each punctum as on-mito (centroid inside the mito binary) or
     off-mito.

  4. Compute two nearest-neighbor distances per punctum:
       - nn_within: nearest neighbor within the same group (on -> on, off -> off).
         Useful for clustering analyses of each population.
       - nn_all   : nearest neighbor among ALL puncta in the image (on or off),
         then split by label downstream. Useful when you care about the
         mixed-population spacing.

Per-image outputs (written to {output_dir}/{basename}{run_name}/):
  {basename}_puncta.csv             one row per detected punctum
  {basename}_overlay.png            visual: puncta colored on (red) vs off (cyan)
  {basename}_radial_profiles.csv    long-format radial intensity profile:
                                    image_name, punctum_id, y, x, on_mito,
                                    distance, mtdna_intensity, mito_intensity,
                                    septin_intensity
                                    (rows = puncta x (radius+1) distance bins)

Pooled outputs (written to {output_dir}/):
  puncta_nn_pooled.csv          every row from every per-image CSV concatenated,
                                with an image_name column added
  puncta_nn_per_image.csv       per-image counts + median NN for on→on, off→off,
                                and all→all
  puncta_nn_histogram.png       Two-panel figure pooled across images:
                                 (1) overlaid histograms of on→on, off→off, all→all
                                 (2) ECDFs of the same three distributions
  puncta_nn_summary.txt         stats for on→on / off→off / all→all + pairwise
                                Mann-Whitney U and KS comparisons (on vs off,
                                on vs all, off vs all)
  puncta_radial_profiles_pooled.csv  every per-image radial profile concatenated
  puncta_radial_profiles.png         three-panel plot (mtDNA / mito / septin)
                                     of radial profiles, on-mito vs off-mito
                                     (mean ± SEM)

CSV schema (per-image, also embedded in the pooled CSV):
  punctum_id, y, x, intensity, on_mito, nn_within_distance, nn_all_distance

`nn_within_distance` is NaN when the punctum has no same-group neighbor
(e.g. only one on-mito punctum in the image). `nn_all_distance` is NaN when
the image has only a single punctum total.

Distances are in pixel units.
"""

import glob
import os

import click
import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, ks_2samp
from skimage.feature import peak_local_max


# -------- filename helpers (mirror network_line_scan basename rules) -------

def _strip_tiff_ext(raw_name):
    """Strip the actual TIFF (or .ome.tif/.ome.tiff) extension only.

    Filenames like ``..._52.1494_decon_NaN.ome.tif`` contain dots inside the
    name, so a naive ``split('.')[0]`` collapses different files to the same
    basename. We mirror the rule used inside mito_protein_line_scanner so the
    per-image directories line up across the two workflows.
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


# -------- mask sourcing ----------------------------------------------------

def _load_saved_mito_binary(mito_binary_dir, basename):
    """Try to load `{basename}_mito_binary.tif` from `mito_binary_dir`.

    Returns a boolean 2D array or None. We accept either 0/255 uint8 (which is
    what `network_line_scan --binary-mask-dir-output` writes) or any
    nonzero-as-True dtype.
    """
    if not mito_binary_dir:
        return None
    candidate = os.path.join(mito_binary_dir, f"{basename}_mito_binary.tif")
    if not os.path.exists(candidate):
        return None
    arr = tf.imread(candidate)
    return arr.astype(bool)


def _build_mito_binary(
    mito_img_eq,
    *,
    use_gui,
    tubule_radius,
    sensitivity,
    min_object_size,
    gap_closing,
    use_thickness_filter,
    min_thickness,
    max_thickness,
):
    """Run the ridge-filter mito-mask pipeline from mito_protein_line_scanner.

    Returns the boolean mito binary (post-thickness-filter). The line scanner
    module is imported lazily here so that just loading this module doesn't
    drag in matplotlib widgets etc.
    """
    pipeline_kwargs = dict(
        tubule_radius=tubule_radius,
        sensitivity=sensitivity,
        min_object_size=min_object_size,
        gap_closing=gap_closing,
        use_thickness_filter=use_thickness_filter,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
    )

    if use_gui:
        from mito_linescan.mito_protein_line_scanner import select_mask_gui
        _, mito_binary, _, _, _ = select_mask_gui(mito_img_eq, **pipeline_kwargs)
    else:
        from mito_linescan.mito_protein_line_scanner import compute_mito_mask_noninteractive
        _, mito_binary, _, _, _ = compute_mito_mask_noninteractive(
            mito_img_eq, **pipeline_kwargs
        )

    return np.asarray(mito_binary, dtype=bool)


def _get_cell_roi(mito_img, mask_dir_input, mask_dir_output, image_path):
    """Return a cell ROI mask. Mirrors the logic in network_line_scan.

    If `mask_dir_input` has a saved mask for this image, load it. Otherwise,
    launch the lasso GUI on the mito channel and save the result to
    `mask_dir_output` (only if that directory was provided).
    """
    from mito_linescan.mito_protein_line_scanner import lasso_select_cell

    base = os.path.basename(image_path)
    if mask_dir_input:
        candidate = os.path.join(mask_dir_input, base)
        if os.path.exists(candidate):
            return tf.imread(candidate).astype(bool)

    roi = lasso_select_cell(mito_img)
    if mask_dir_output:
        os.makedirs(mask_dir_output, exist_ok=True)
        tf.imwrite(os.path.join(mask_dir_output, base), roi.astype(np.uint8))
    return roi


def _prepare_mito_for_mask(mito_img, roi):
    """Same normalization pipeline used by network_line_scan before mask GUI.

    Rescale mito channel to [-1, 1], hist-equalize on the non-background pixels,
    then multiply by the cell ROI so the ridge filter only fires inside the cell.
    """
    from skimage import exposure
    mito_img = mito_img.astype(np.float32)
    rng = mito_img.max() - mito_img.min()
    if rng < 1e-12:
        return mito_img
    mito_norm = (mito_img - mito_img.min()) / rng * 2.0 - 1.0
    mito_eq = exposure.equalize_hist(mito_norm, nbins=256, mask=(mito_norm > -0.9))
    return mito_eq * roi


# -------- puncta detection --------------------------------------------------

def _normalize_for_puncta(protein_img, roi):
    """Normalize the protein channel to [0,1] using ROI percentiles.

    Returns (img_n, p_lo, p_hi). Sharing this between detect_puncta and the
    interactive GUI guarantees a punctum the user accepts in the GUI matches
    what `--no-punct-use-gui` with the same params would produce.
    """
    img = np.asarray(protein_img, dtype=np.float32)
    base = img[roi] if roi.any() else img
    p_lo = float(np.percentile(base, 1))
    p_hi = float(np.percentile(base, 99.9))
    denom = max(p_hi - p_lo, 1e-9)
    img_n = np.clip((img - p_lo) / denom, 0.0, 1.0).astype(np.float32)
    return img_n, p_lo, p_hi


def _detect_on_smoothed(smoothed, roi, *,
                         min_distance, threshold_abs, threshold_rel,
                         exclude_border):
    """peak_local_max on an already-smoothed [0,1] image, restricted to ROI.

    Returns (coords_yx, intensities)."""
    kwargs = dict(
        min_distance=int(max(1, min_distance)),
        exclude_border=int(max(0, exclude_border)),
    )
    if threshold_abs is not None and threshold_abs > 0:
        kwargs['threshold_abs'] = float(threshold_abs)
    if threshold_rel is not None and threshold_rel > 0:
        kwargs['threshold_rel'] = float(threshold_rel)

    coords = peak_local_max(smoothed, **kwargs)
    if coords.size == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=np.float32)

    keep = roi[coords[:, 0], coords[:, 1]]
    coords = coords[keep]
    if coords.size == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=np.float32)

    intensities = smoothed[coords[:, 0], coords[:, 1]].astype(np.float32)
    return coords, intensities


def detect_puncta(
    protein_img,
    roi,
    *,
    blur_sigma,
    min_distance,
    threshold_abs,
    threshold_rel,
    exclude_border,
):
    """Detect puncta as local maxima in the (smoothed) protein channel.

    Returns (coords_yx, intensities) where coords_yx is shape (N, 2) in
    (row, col) order matching skimage conventions, and intensities are sampled
    from the smoothed image at each peak.

    Detection is restricted to the cell ROI: peaks outside `roi` are dropped.
    """
    img_n, _, _ = _normalize_for_puncta(protein_img, roi)
    smoothed = (gaussian_filter(img_n, sigma=float(blur_sigma))
                if blur_sigma > 0 else img_n)
    return _detect_on_smoothed(
        smoothed, roi,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
    )


def select_puncta_gui(
    protein_img,
    roi,
    mito_binary=None,
    *,
    blur_sigma=1.0,
    min_distance=3,
    threshold_abs=0.0,
    threshold_rel=0.2,
    exclude_border=1,
):
    """Interactive puncta-detection GUI.

    Sliders for blur sigma, min distance, threshold_abs, threshold_rel,
    exclude_border, and contrast (vmin/vmax). The puncta scatter updates live
    on every change; when `mito_binary` is provided, puncta are colored red
    (on-mito) vs cyan (off-mito) in real time so the user can tune by eye.

    Returns (coords_yx, intensities, params), where `params` is a dict of the
    final slider values so the caller can echo or persist them.
    """
    from matplotlib.widgets import Slider, Button

    img = np.asarray(protein_img, dtype=np.float32)
    roi_arr = (np.asarray(roi, dtype=bool) if roi is not None
               else np.ones_like(img, dtype=bool))
    if roi_arr.shape != img.shape:
        raise ValueError(
            f"ROI shape {roi_arr.shape} != image shape {img.shape}")

    img_n, _, _ = _normalize_for_puncta(img, roi_arr)

    # State carried across slider callbacks.
    state = {
        'blur_sigma': float(blur_sigma),
        'min_distance': int(min_distance),
        'threshold_abs': float(threshold_abs),
        'threshold_rel': float(threshold_rel),
        'exclude_border': int(exclude_border),
        'smoothed': img_n if blur_sigma <= 0 else None,
        'last_blur': 0.0 if blur_sigma <= 0 else None,
        'coords': np.empty((0, 2), dtype=int),
        'intensities': np.empty((0,), dtype=np.float32),
    }

    def _ensure_smoothed():
        if state['last_blur'] != state['blur_sigma']:
            sig = state['blur_sigma']
            state['smoothed'] = (gaussian_filter(img_n, sigma=sig)
                                  if sig > 0 else img_n)
            state['last_blur'] = sig

    def _recompute():
        _ensure_smoothed()
        coords, intens = _detect_on_smoothed(
            state['smoothed'], roi_arr,
            min_distance=state['min_distance'],
            threshold_abs=state['threshold_abs'],
            threshold_rel=state['threshold_rel'],
            exclude_border=state['exclude_border'],
        )
        state['coords'] = coords
        state['intensities'] = intens

    # ---- Layout ----
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.subplots_adjust(bottom=0.30, top=0.95)

    vmin0 = float(np.percentile(img, 1))
    vmax0 = float(np.percentile(img, 99.5))
    im = ax.imshow(img, cmap='gray', vmin=vmin0, vmax=vmax0)
    ax.set_axis_off()

    if mito_binary is not None and np.any(mito_binary):
        ax.contour(np.asarray(mito_binary, dtype=float), levels=[0.5],
                   colors='yellow', linewidths=0.4, alpha=0.6)
    if not roi_arr.all():
        ax.contour(roi_arr.astype(float), levels=[0.5],
                   colors='white', linewidths=0.3, alpha=0.4)

    scat_on, = ax.plot([], [], 'o', mfc='none', mec='red',
                      mew=0.8, ms=6, linestyle='None')
    scat_off, = ax.plot([], [], 'o', mfc='none', mec='cyan',
                       mew=0.8, ms=6, linestyle='None')
    title = ax.set_title("Adjust sliders, click Done to accept")

    def _redraw():
        coords = state['coords']
        if coords.shape[0] and mito_binary is not None:
            on = np.asarray(mito_binary, dtype=bool)[coords[:, 0], coords[:, 1]]
        else:
            on = np.ones(coords.shape[0], dtype=bool)
        on_pts = coords[on]
        off_pts = coords[~on]
        if on_pts.size:
            scat_on.set_data(on_pts[:, 1], on_pts[:, 0])
        else:
            scat_on.set_data([], [])
        if off_pts.size:
            scat_off.set_data(off_pts[:, 1], off_pts[:, 0])
        else:
            scat_off.set_data([], [])
        total = coords.shape[0]
        title.set_text(
            f"Puncta: total={total}, on={int(on.sum())}, "
            f"off={int((~on).sum())}   "
            f"[σ={state['blur_sigma']:.2f}, min_d={state['min_distance']}, "
            f"abs={state['threshold_abs']:.2f}, rel={state['threshold_rel']:.2f}]"
        )
        fig.canvas.draw_idle()

    # Slider rows (top to bottom inside the bottom margin)
    s_blur = Slider(fig.add_axes([0.15, 0.235, 0.7, 0.025]),
                    'blur σ (px)', 0.0, 5.0, valinit=state['blur_sigma'])
    s_mind = Slider(fig.add_axes([0.15, 0.200, 0.7, 0.025]),
                    'min dist (px)', 1, 25,
                    valinit=state['min_distance'], valstep=1)
    s_tabs = Slider(fig.add_axes([0.15, 0.165, 0.7, 0.025]),
                    'threshold abs', 0.0, 1.0,
                    valinit=state['threshold_abs'])
    s_trel = Slider(fig.add_axes([0.15, 0.130, 0.7, 0.025]),
                    'threshold rel', 0.0, 1.0,
                    valinit=state['threshold_rel'])
    s_brd = Slider(fig.add_axes([0.15, 0.095, 0.7, 0.025]),
                   'border (px)', 0, 20,
                   valinit=state['exclude_border'], valstep=1)
    s_vmin = Slider(fig.add_axes([0.15, 0.050, 0.32, 0.020]),
                    'vmin', float(img.min()), float(img.max()), valinit=vmin0)
    s_vmax = Slider(fig.add_axes([0.53, 0.050, 0.32, 0.020]),
                    'vmax', float(img.min()), float(img.max()), valinit=vmax0)

    def _on_param(_=None):
        state['blur_sigma'] = float(s_blur.val)
        state['min_distance'] = int(s_mind.val)
        state['threshold_abs'] = float(s_tabs.val)
        state['threshold_rel'] = float(s_trel.val)
        state['exclude_border'] = int(s_brd.val)
        _recompute()
        _redraw()

    def _on_contrast(_=None):
        vmin, vmax = float(s_vmin.val), float(s_vmax.val)
        if vmin < vmax:
            im.set_clim(vmin, vmax)
            fig.canvas.draw_idle()

    for s in (s_blur, s_mind, s_tabs, s_trel, s_brd):
        s.on_changed(_on_param)
    for s in (s_vmin, s_vmax):
        s.on_changed(_on_contrast)

    b_done = Button(fig.add_axes([0.85, 0.965, 0.12, 0.03]), 'Done')
    b_done.on_clicked(lambda _ev: plt.close(fig))

    # Initial paint at the supplied defaults.
    _recompute()
    _redraw()
    plt.show()

    params = dict(
        blur_sigma=state['blur_sigma'],
        min_distance=state['min_distance'],
        threshold_abs=state['threshold_abs'],
        threshold_rel=state['threshold_rel'],
        exclude_border=state['exclude_border'],
    )
    return state['coords'], state['intensities'], params


# -------- NN computation ----------------------------------------------------

def _within_group_nn(coords):
    """For an (N, 2) array of points, return the nearest-neighbor distance
    for each point to any *other* point in the same array. NaN if N < 2."""
    if coords.shape[0] < 2:
        return np.full(coords.shape[0], np.nan, dtype=np.float32)
    tree = cKDTree(coords)
    # k=2: first hit is the point itself (distance 0), second is the nearest
    # distinct neighbor.
    dists, _ = tree.query(coords, k=2)
    return dists[:, 1].astype(np.float32)


def _all_group_nn(coords_all):
    """For an (N, 2) array of all puncta, return each point's NN distance to
    any other point. NaN if N < 2."""
    return _within_group_nn(coords_all)


# -------- radial intensity profiles around puncta ---------------------------

def _radial_profile(image, y0, x0, radius):
    """Mean intensity vs integer distance from `(y0, x0)` out to `radius` px.

    Returns a length-(radius+1) float32 array. Bin `r` contains pixels whose
    Euclidean distance to the center, rounded to the nearest integer, equals
    `r`. NaN at radii whose ring has no pixels inside the image bounds (only
    relevant for puncta near the edge).
    """
    H, W = image.shape
    y_min = max(0, int(y0) - radius)
    y_max = min(H, int(y0) + radius + 1)
    x_min = max(0, int(x0) - radius)
    x_max = min(W, int(x0) + radius + 1)

    if y_max <= y_min or x_max <= x_min:
        return np.full(radius + 1, np.nan, dtype=np.float32)

    sub = image[y_min:y_max, x_min:x_max].astype(np.float32)
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    d = np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2).ravel()
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


def compute_radial_profiles(
    coords, on_mito,
    mtdna_img_n, mito_img_n, septin_img_n,
    radius, basename,
):
    """Per-punctum radial intensity profiles in all three channels.

    Returns a long-format DataFrame with one row per (punctum, distance):
        image_name, punctum_id, y, x, on_mito, distance,
        mtdna_intensity, mito_intensity, septin_intensity

    All three intensity columns are sampled from images that the caller has
    already normalized to [0, 1] (typically via `_normalize_for_puncta`), so
    profiles are comparable across images and between channels.

    Note: if `mtdna_channel == mito_channel` in the config (i.e. you do not
    have a dedicated mtDNA stain), the `mtdna_intensity` and `mito_intensity`
    columns will be identical; set `mtdna_channel` to your DNA-stain channel
    to differentiate them.
    """
    cols = ['image_name', 'punctum_id', 'y', 'x', 'on_mito', 'distance',
            'mtdna_intensity', 'mito_intensity', 'septin_intensity']
    if coords.shape[0] == 0:
        return pd.DataFrame(columns=cols)

    n = coords.shape[0]
    nb = radius + 1
    mt = np.empty((n, nb), dtype=np.float32)
    mi = np.empty((n, nb), dtype=np.float32)
    sp = np.empty((n, nb), dtype=np.float32)
    for i in range(n):
        y, x = int(coords[i, 0]), int(coords[i, 1])
        mt[i] = _radial_profile(mtdna_img_n, y, x, radius)
        mi[i] = _radial_profile(mito_img_n, y, x, radius)
        sp[i] = _radial_profile(septin_img_n, y, x, radius)

    punctum_id = np.repeat(np.arange(n, dtype=np.int64), nb)
    distance = np.tile(np.arange(nb, dtype=np.int64), n)
    y_rep = np.repeat(coords[:, 0].astype(np.int64), nb)
    x_rep = np.repeat(coords[:, 1].astype(np.int64), nb)
    on_rep = np.repeat(np.asarray(on_mito, dtype=bool), nb)

    return pd.DataFrame({
        'image_name': basename,
        'punctum_id': punctum_id,
        'y': y_rep,
        'x': x_rep,
        'on_mito': on_rep,
        'distance': distance,
        'mtdna_intensity': mt.ravel(),
        'mito_intensity': mi.ravel(),
        'septin_intensity': sp.ravel(),
    })


def _save_radial_profile_outputs(per_image_radial_dfs, output_dir, radius):
    """Write puncta_radial_profiles_pooled.csv and the mean ± SEM PNG plot."""
    if not per_image_radial_dfs:
        click.echo("No radial profiles to pool; skipping radial outputs.")
        return

    pooled = pd.concat(per_image_radial_dfs, ignore_index=True)
    pooled_csv = os.path.join(output_dir, 'puncta_radial_profiles_pooled.csv')
    pooled.to_csv(pooled_csv, index=False)

    # Three-panel plot: mtDNA (left), mito (middle), septin (right). Each
    # panel shows mean ± SEM of the radial profile, split by on/off-mito.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    groups = [('on-mito',  True,  'tab:red'),
              ('off-mito', False, 'tab:cyan')]

    for ax, col, title in (
        (axes[0], 'mtdna_intensity',  'mtDNA channel'),
        (axes[1], 'mito_intensity',   'Mito channel'),
        (axes[2], 'septin_intensity', 'Septin channel'),
    ):
        any_drawn = False
        for label, want_on, color in groups:
            sub = pooled[pooled['on_mito'] == want_on]
            if sub.empty:
                continue
            # n_puncta = number of unique (image, punctum_id) pairs in this group
            n_punct = sub[['image_name', 'punctum_id']].drop_duplicates().shape[0]
            grouped = sub.groupby('distance')[col]
            mean = grouped.mean()
            sem = grouped.sem(ddof=1).fillna(0.0)
            ax.plot(mean.index, mean.values, color=color, linewidth=1.6,
                    label=f'{label} (n={n_punct})')
            ax.fill_between(mean.index,
                            (mean - sem).values, (mean + sem).values,
                            color=color, alpha=0.25, linewidth=0)
            any_drawn = True
        ax.set_xlabel('Distance from punctum center (px)')
        ax.set_ylabel('Mean intensity (normalized)')
        ax.set_title(title)
        ax.set_xlim(0, radius)
        ax.grid(True, alpha=0.3)
        if any_drawn:
            ax.legend(fontsize=8)

    fig.suptitle(f'Radial intensity profile around puncta '
                 f'(mean ± SEM, R={radius} px)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'puncta_radial_profiles.png'), dpi=200)
    plt.close(fig)


# -------- overlay rendering -------------------------------------------------

def _save_overlay(out_path, protein_img, mito_binary, coords, on_mito, roi):
    """Save a PNG showing puncta colored by on/off classification on top of
    the protein channel, with the mito binary mask outlined."""
    fig, ax = plt.subplots(figsize=(8, 8))
    p_lo = float(np.percentile(protein_img, 1))
    p_hi = float(np.percentile(protein_img, 99.5))
    ax.imshow(protein_img, cmap='gray', vmin=p_lo, vmax=p_hi)

    if mito_binary is not None and mito_binary.any():
        ax.contour(mito_binary.astype(float), levels=[0.5],
                   colors='yellow', linewidths=0.4, alpha=0.6)
    if roi is not None and not roi.all():
        ax.contour(roi.astype(float), levels=[0.5],
                   colors='white', linewidths=0.3, alpha=0.4)

    if coords.shape[0] > 0:
        on = coords[on_mito]
        off = coords[~on_mito]
        if on.shape[0] > 0:
            ax.scatter(on[:, 1], on[:, 0], s=18, edgecolor='red',
                       facecolor='none', linewidths=0.8, label=f'on  (n={on.shape[0]})')
        if off.shape[0] > 0:
            ax.scatter(off[:, 1], off[:, 0], s=18, edgecolor='cyan',
                       facecolor='none', linewidths=0.8, label=f'off (n={off.shape[0]})')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.7)

    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# -------- pooled outputs ----------------------------------------------------

def _save_pooled_outputs(per_image_dfs, output_dir):
    """Write puncta_nn_pooled.csv, puncta_nn_per_image.csv, the histogram PNG,
    and the summary .txt to output_dir."""
    if not per_image_dfs:
        click.echo("No per-image results to pool; skipping pooled outputs.")
        return

    pooled = pd.concat(per_image_dfs, ignore_index=True)
    pooled_csv = os.path.join(output_dir, 'puncta_nn_pooled.csv')
    pooled.to_csv(pooled_csv, index=False)

    # Three NN distributions pooled across all images:
    #   on->on   : within-group NN distance for on-mito puncta only
    #   off->off : within-group NN distance for off-mito puncta only
    #   all->all : every punctum's NN distance to any other punctum
    #              (the union of on->any and off->any, NOT split by label)
    on_mask = pooled['on_mito'].values.astype(bool)
    nn_on_on = pooled.loc[on_mask, 'nn_within_distance'].dropna().values
    nn_off_off = pooled.loc[~on_mask, 'nn_within_distance'].dropna().values
    nn_all_all = pooled['nn_all_distance'].dropna().values

    # per-image summary (now includes medians for all three groups)
    rows = []
    for image_name, df in pooled.groupby('image_name'):
        on = df['on_mito'].values.astype(bool)
        nn_on = df.loc[on, 'nn_within_distance'].dropna()
        nn_off = df.loc[~on, 'nn_within_distance'].dropna()
        nn_all = df['nn_all_distance'].dropna()
        rows.append({
            'image_name': image_name,
            'n_total': len(df),
            'n_on': int(on.sum()),
            'n_off': int((~on).sum()),
            'median_nn_on_on': float(nn_on.median()) if len(nn_on) else np.nan,
            'median_nn_off_off': float(nn_off.median()) if len(nn_off) else np.nan,
            'median_nn_all_all': float(nn_all.median()) if len(nn_all) else np.nan,
        })
    per_image = pd.DataFrame(rows).sort_values('image_name')
    per_image.to_csv(os.path.join(output_dir, 'puncta_nn_per_image.csv'),
                     index=False)

    # Drop empty distributions for histogram cap computation
    nonempty = [a for a in (nn_on_on, nn_off_off, nn_all_all) if a.size]
    if not nonempty:
        click.echo("All NN distances are NaN — skipping histogram.")
        return

    cap = float(np.percentile(np.concatenate(nonempty), 99))
    cap = max(cap, 1.0)
    bins = np.linspace(0, cap, 41)

    # Two panels:
    #   (1) overlaid histograms of on->on (red), off->off (cyan), all->all (gray step)
    #   (2) ECDFs of the same three (easier to read with overlapping distributions)
    groups = [
        ('on→on',   nn_on_on,   'tab:red',  'fill'),
        ('off→off', nn_off_off, 'tab:cyan', 'fill'),
        ('all→all', nn_all_all, '0.25',     'step'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for name, arr, color, style in groups:
        if not arr.size:
            continue
        label = f'{name} (n={arr.size}, med={np.median(arr):.1f})'
        if style == 'step':
            ax.hist(arr, bins=bins, histtype='step', linewidth=1.6,
                    color=color, label=label)
        else:
            ax.hist(arr, bins=bins, alpha=0.5, color=color, label=label)
    ax.set_xlabel('NN distance (px)')
    ax.set_ylabel('Puncta count')
    ax.set_title('Nearest-neighbor distance (pooled across images)')
    ax.legend(fontsize=8)

    ax = axes[1]
    for name, arr, color, _style in groups:
        if not arr.size:
            continue
        xs = np.sort(arr)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax.plot(xs, ys, color=color, linewidth=1.6, label=f'{name} (n={xs.size})')
    ax.set_xlabel('NN distance (px)')
    ax.set_ylabel('Cumulative fraction')
    ax.set_title('ECDF of NN distances')
    ax.set_xlim(0, cap)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'puncta_nn_histogram.png'), dpi=200)
    plt.close(fig)

    # Summary stats + pairwise Mann-Whitney U / KS comparisons.
    # Note: all->all overlaps with on->on and off->off (it pools everything),
    # so the on-vs-all / off-vs-all tests are NOT independent; they're useful
    # for asking "is on-mito clustering tighter than the overall population?"
    # rather than as a strict two-sample test.
    def _stats(label, arr):
        if arr.size == 0:
            return f"  {label:<10s} n=0"
        return (f"  {label:<10s} n={arr.size:<6d} "
                f"mean={arr.mean():.2f} median={np.median(arr):.2f} "
                f"std={arr.std():.2f} p25={np.percentile(arr, 25):.2f} "
                f"p75={np.percentile(arr, 75):.2f}")

    def _compare(name_a, a, name_b, b):
        if a.size < 2 or b.size < 2:
            return (f"  {name_a} vs {name_b}: skipped "
                    f"(need ≥2 samples on each side; got {a.size} and {b.size})")
        u, pu = mannwhitneyu(a, b, alternative='two-sided')
        ks, pks = ks_2samp(a, b)
        return (f"  {name_a} vs {name_b}:\n"
                f"    Mann-Whitney U: U={u:.1f}, p={pu:.3e}\n"
                f"    KS:             D={ks:.3f}, p={pks:.3e}")

    lines = [
        "Puncta nearest-neighbor summary",
        "=" * 60,
        f"Images processed     : {pooled['image_name'].nunique()}",
        f"Total puncta         : {len(pooled)}",
        f"On-mito puncta       : {int(on_mask.sum())}",
        f"Off-mito puncta      : {int((~on_mask).sum())}",
        "",
        "Per-group stats:",
        _stats('on→on',   nn_on_on),
        _stats('off→off', nn_off_off),
        _stats('all→all', nn_all_all),
        "",
        "Pairwise comparisons (two-sided):",
        _compare('on→on',   nn_on_on,   'off→off', nn_off_off),
        _compare('on→on',   nn_on_on,   'all→all', nn_all_all),
        _compare('off→off', nn_off_off, 'all→all', nn_all_all),
    ]
    with open(os.path.join(output_dir, 'puncta_nn_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


# -------- per-image driver --------------------------------------------------

def process_one_image(
    image_path,
    *,
    output_dir,
    run_name,
    mito_channel,
    protein_channel,
    mtdna_channel,
    mito_binary_dir,
    mask_dir_input,
    mask_dir_output,
    use_gui,
    tubule_radius,
    sensitivity,
    min_object_size,
    gap_closing,
    use_thickness_filter,
    min_thickness,
    max_thickness,
    punct_use_gui,
    punct_blur_sigma,
    punct_min_distance,
    punct_threshold_abs,
    punct_threshold_rel,
    punct_exclude_border,
    punct_scan_radius,
    save_overlay,
):
    """Run the full pipeline for one image and write per-image outputs.

    Returns a (puncta_df, radial_df) tuple so the caller can pool both the
    per-punctum NN table and the per-punctum radial intensity profiles across
    images. Either may be None if there were no detections.
    """
    raw_name = os.path.basename(image_path)
    basename = _strip_tiff_ext(raw_name)
    image_out_dir = os.path.join(output_dir, basename + run_name)
    os.makedirs(image_out_dir, exist_ok=True)

    img = tf.imread(image_path)
    # Reject anything that isn't a multi-channel stack. The first failure mode
    # is the user (or a previous run) leaving a `*_mito_binary.tif` in the
    # input dir; those are 2D and would crash on the channel indexing below.
    if img.ndim != 3:
        click.echo(
            f"  WARNING: expected 3D (C, Y, X) TIFF; got shape {img.shape}. "
            f"Skipping {basename} (most likely a single-channel mask file)."
        )
        return None, None
    n_channels = img.shape[0]
    if (mito_channel >= n_channels or protein_channel >= n_channels
            or mtdna_channel >= n_channels):
        click.echo(
            f"  WARNING: image has {n_channels} channels but mito_channel="
            f"{mito_channel}, protein_channel={protein_channel}, "
            f"mtdna_channel={mtdna_channel}. Skipping {basename}."
        )
        return None, None
    mito_img = img[mito_channel, :, :]
    protein_img = img[protein_channel, :, :]
    mtdna_img = img[mtdna_channel, :, :]

    # ---- Mito binary mask: prefer pre-saved, fall back to ridge-filter ----
    mito_binary = _load_saved_mito_binary(mito_binary_dir, basename)
    if mito_binary is None:
        # No saved binary -> walk the user through the exact same mask-making
        # flow as network_line_scan: lasso cell ROI (or load a saved one),
        # normalize+equalize the mito channel inside the ROI, then ridge-filter
        # GUI (or non-interactive equivalent) for the binary.
        roi = _get_cell_roi(mito_img, mask_dir_input, mask_dir_output, image_path)
        mito_eq = _prepare_mito_for_mask(mito_img, roi)
        mito_binary = _build_mito_binary(
            mito_eq,
            use_gui=use_gui,
            tubule_radius=tubule_radius,
            sensitivity=sensitivity,
            min_object_size=min_object_size,
            gap_closing=gap_closing,
            use_thickness_filter=use_thickness_filter,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
        )
        click.echo(f"  built mito binary on the fly ({int(mito_binary.sum())} px)")

        # Persist the newly-built binary so subsequent runs (and other
        # workflows like network_line_scan) can pick it up directly without
        # re-opening the GUI. This mirrors network_line_scan's
        # binary_mask_dir_output convention: uint8 0/255, photometric
        # minisblack, named `{basename}_mito_binary.tif`. We write into
        # `mito_binary_dir` when it's set (the same place the loader looks),
        # so that path is dual-purpose: load from here, save to here.
        if mito_binary_dir:
            try:
                os.makedirs(mito_binary_dir, exist_ok=True)
                out_path = os.path.join(mito_binary_dir,
                                        f"{basename}_mito_binary.tif")
                mito_binary_u8 = (np.asarray(mito_binary, dtype=bool)
                                  .astype(np.uint8) * 255)
                tf.imwrite(out_path, mito_binary_u8, photometric='minisblack')
                click.echo(f"  saved mito binary to {out_path}")
            except Exception as exc:
                click.echo(f"  WARNING: failed to save mito binary: {exc}")
    else:
        # When the binary is precomputed we still need a cell ROI so we don't
        # detect puncta outside the cell. If none is available we use the
        # whole image.
        if mask_dir_input:
            candidate = os.path.join(mask_dir_input, raw_name)
            if os.path.exists(candidate):
                roi = tf.imread(candidate).astype(bool)
            else:
                roi = np.ones_like(mito_img, dtype=bool)
        else:
            roi = np.ones_like(mito_img, dtype=bool)
        click.echo(f"  loaded mito binary from disk ({int(mito_binary.sum())} px)")

    # Defensive shape check: the pre-saved mask must match this image.
    if mito_binary.shape != protein_img.shape:
        click.echo(
            f"  WARNING: mito binary shape {mito_binary.shape} does not match "
            f"image shape {protein_img.shape}; skipping {basename}."
        )
        return None, None

    # ---- Puncta detection (interactive or headless) ----
    if punct_use_gui:
        coords, intensities, gui_params = select_puncta_gui(
            protein_img,
            roi,
            mito_binary=mito_binary,
            blur_sigma=punct_blur_sigma,
            min_distance=punct_min_distance,
            threshold_abs=punct_threshold_abs,
            threshold_rel=punct_threshold_rel,
            exclude_border=punct_exclude_border,
        )
        click.echo(
            f"  GUI accepted: σ={gui_params['blur_sigma']:.2f} "
            f"min_d={gui_params['min_distance']} "
            f"abs={gui_params['threshold_abs']:.2f} "
            f"rel={gui_params['threshold_rel']:.2f} "
            f"border={gui_params['exclude_border']}"
        )
    else:
        coords, intensities = detect_puncta(
            protein_img,
            roi,
            blur_sigma=punct_blur_sigma,
            min_distance=punct_min_distance,
            threshold_abs=punct_threshold_abs,
            threshold_rel=punct_threshold_rel,
            exclude_border=punct_exclude_border,
        )
    n = coords.shape[0]
    click.echo(f"  detected {n} puncta in protein channel")
    if n == 0:
        return None, None

    # ---- On/off classification (centroid inside mito_binary) ----
    on_mito = mito_binary[coords[:, 0], coords[:, 1]].astype(bool)

    # ---- NN distances ----
    nn_within = np.full(n, np.nan, dtype=np.float32)
    if on_mito.any():
        nn_within[on_mito] = _within_group_nn(coords[on_mito])
    if (~on_mito).any():
        nn_within[~on_mito] = _within_group_nn(coords[~on_mito])
    nn_all = _all_group_nn(coords)

    # ---- Per-image CSV ----
    df = pd.DataFrame({
        'image_name': basename,
        'punctum_id': np.arange(n),
        'y': coords[:, 0].astype(int),
        'x': coords[:, 1].astype(int),
        'intensity': intensities,
        'on_mito': on_mito,
        'nn_within_distance': nn_within,
        'nn_all_distance': nn_all,
    })
    df.to_csv(os.path.join(image_out_dir, f"{basename}_puncta.csv"),
              index=False)

    # ---- Overlay PNG ----
    if save_overlay:
        try:
            _save_overlay(
                os.path.join(image_out_dir, f"{basename}_overlay.png"),
                protein_img, mito_binary, coords, on_mito, roi,
            )
        except Exception as exc:
            click.echo(f"  overlay rendering failed: {exc}")

    # ---- Radial intensity profiles around each punctum -------------------
    # All three channels are normalized to [0, 1] using the same
    # ROI-percentile scheme as the puncta-detection path, so profiles are
    # comparable across images and between channels.
    mtdna_img_n, _, _ = _normalize_for_puncta(mtdna_img, roi)
    mito_img_n, _, _ = _normalize_for_puncta(mito_img, roi)
    protein_img_n, _, _ = _normalize_for_puncta(protein_img, roi)
    radial_df = compute_radial_profiles(
        coords, on_mito,
        mtdna_img_n, mito_img_n, protein_img_n,
        radius=int(punct_scan_radius), basename=basename,
    )
    radial_csv = os.path.join(image_out_dir, f"{basename}_radial_profiles.csv")
    radial_df.to_csv(radial_csv, index=False)
    click.echo(f"  wrote radial profiles ({len(radial_df)} rows) -> "
               f"{os.path.basename(radial_csv)}")

    return df, radial_df


# -------- CLI ---------------------------------------------------------------

@click.command()
@click.option('--input-dir', default='examples/', show_default=True,
              help='Input directory containing multi-channel TIFF images.')
@click.option('--input-pattern', default='*.tif', show_default=True,
              help='Glob pattern for input TIFFs within --input-dir.')
@click.option('--output-dir', default='', type=str,
              help='Where per-image dirs + pooled outputs go. '
                   'Defaults to --input-dir.')
@click.option('--run-name', default='run1', show_default=True,
              help='Suffix appended to each per-image output directory.')
@click.option('--mito-channel', default=0, type=int, show_default=True,
              help='0-based index of the mitochondrial channel.')
@click.option('--protein-channel', default=2, type=int, show_default=True,
              help='0-based index of the protein/septin channel.')
@click.option('--mtdna-channel', default=0, type=int, show_default=True,
              help='0-based index of the mtDNA channel for the radial '
                   'intensity profile. Defaults to the mito channel; set to a '
                   'different index if you have a dedicated mtDNA stain '
                   '(e.g. PicoGreen) on its own channel.')
@click.option('--mito-binary-dir', default='', type=str,
              help='Directory for `{basename}_mito_binary.tif`. Dual-purpose: '
                   'if a matching file is found we load it and skip the '
                   'ridge-filter step; if not, we run the same mask-making '
                   'flow as network_line_scan (lasso cell ROI -> ridge-filter '
                   'GUI/no-GUI) and SAVE the result here so subsequent runs '
                   'can pick it up. Leave empty to never load and never save.')
@click.option('--mask-dir-input', default='', type=str,
              help='Directory containing pre-saved cell ROI lasso masks '
                   '(one per image, same basename). Used when the mito binary '
                   'is not pre-saved.')
@click.option('--mask-dir-output', default='', type=str,
              help='Directory to write new cell ROI masks into when the user '
                   'has to lasso a fresh one.')
@click.option('--use-gui/--no-use-gui', default=True,
              help='Use the interactive ridge-filter mask GUI for any image '
                   'that does NOT have a pre-saved mito binary.')
# --- Ridge-filter mask pipeline (mirrors network_line_scan) ---
@click.option('--tubule-radius', default=2.0, type=float, show_default=True,
              help='Tubule radius in px for the ridge filter (top-hat disk + '
                   'meijering sigmas).')
@click.option('--sensitivity', default=1.0, type=float, show_default=True,
              help='Multiplier on the Otsu cut of the ridge response.')
@click.option('--min-object-size', default=30, type=int, show_default=True,
              help='Drop binary connected components smaller than this (px).')
@click.option('--gap-closing', default=1, type=int, show_default=True,
              help='Binary closing disk radius (px) before skeletonization.')
@click.option('--use-thickness-filter/--no-use-thickness-filter', default=False,
              help='Apply a local-thickness range filter to the mito mask '
                   '(requires the `localthickness` PyPI package).')
@click.option('--min-thickness', default=1.0, type=float, show_default=True,
              help='Min local thickness (px) when --use-thickness-filter.')
@click.option('--max-thickness', default=20.0, type=float, show_default=True,
              help='Max local thickness (px) when --use-thickness-filter.')
# --- Puncta detection (local maxima + threshold) ---
@click.option('--punct-use-gui/--no-punct-use-gui', default=True,
              help='Open an interactive GUI with live sliders for blur sigma, '
                   'min distance, threshold_abs, threshold_rel, and border. '
                   'Puncta are colored on-mito (red) vs off-mito (cyan) in '
                   'real time when a mito binary is available.')
@click.option('--punct-blur-sigma', default=1.0, type=float, show_default=True,
              help='Gaussian smoothing sigma applied before peak detection. '
                   '0 disables smoothing. Used as the GUI starting point '
                   'when --punct-use-gui.')
@click.option('--punct-min-distance', default=3, type=int, show_default=True,
              help='Minimum separation (px) between detected peaks.')
@click.option('--punct-threshold-abs', default=0.0, type=float, show_default=True,
              help='Absolute intensity threshold on the [0,1]-normalized '
                   'protein channel. 0 disables.')
@click.option('--punct-threshold-rel', default=0.2, type=float, show_default=True,
              help='Relative threshold (fraction of image max on the '
                   'normalized channel). 0 disables.')
@click.option('--punct-exclude-border', default=1, type=int, show_default=True,
              help='Pixels near the image edge to exclude from peak detection.')
@click.option('--punct-scan-radius', default=20, type=int, show_default=True,
              help='Radius (px) for the per-punctum radial intensity scan in '
                   'the mtDNA and septin channels. Writes a long-format CSV '
                   '({basename}_radial_profiles.csv) per image plus a pooled '
                   'CSV + mean ± SEM plot across all images.')
# --- Output options ---
@click.option('--save-overlay/--no-save-overlay', default=True, show_default=True,
              help='Save a per-image overlay PNG showing puncta colored by '
                   'on/off classification.')
def main(input_dir, input_pattern, output_dir, run_name,
         mito_channel, protein_channel, mtdna_channel,
         mito_binary_dir, mask_dir_input, mask_dir_output,
         use_gui, tubule_radius, sensitivity, min_object_size,
         gap_closing, use_thickness_filter, min_thickness, max_thickness,
         punct_use_gui,
         punct_blur_sigma, punct_min_distance,
         punct_threshold_abs, punct_threshold_rel, punct_exclude_border,
         punct_scan_radius,
         save_overlay):
    """Detect puncta in the protein channel and compare nearest-neighbor
    distances on the mitochondrial network vs off it."""
    out_dir = output_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)

    image_list = sorted(glob.glob(os.path.join(input_dir, input_pattern)))
    # The `mito_binary_dir` workflow writes `{basename}_mito_binary.tif` next
    # to the real multi-channel TIFFs (and that's the convention
    # network_line_scan uses too). If the user globs `*.tif`, those single-
    # channel masks get picked up and then crash on `img[mito_channel, ...]`
    # because they're 2D. Drop them here regardless of pattern.
    filtered = [p for p in image_list
                if not os.path.basename(p).lower().endswith('_mito_binary.tif')]
    skipped = len(image_list) - len(filtered)
    if skipped:
        click.echo(f"Skipping {skipped} `_mito_binary.tif` file(s) "
                   f"in the input list (those are mask outputs, not inputs).")
    image_list = filtered

    if not image_list:
        click.echo(f"No images found at {os.path.join(input_dir, input_pattern)}")
        return

    click.echo(f"Found {len(image_list)} image(s).")
    per_image_dfs = []
    per_image_radial_dfs = []
    for image_path in image_list:
        click.echo(f"Processing {os.path.basename(image_path)}")
        try:
            result = process_one_image(
                image_path,
                output_dir=out_dir,
                run_name=run_name,
                mito_channel=mito_channel,
                protein_channel=protein_channel,
                mtdna_channel=mtdna_channel,
                mito_binary_dir=mito_binary_dir,
                mask_dir_input=mask_dir_input,
                mask_dir_output=mask_dir_output,
                use_gui=use_gui,
                tubule_radius=tubule_radius,
                sensitivity=sensitivity,
                min_object_size=min_object_size,
                gap_closing=gap_closing,
                use_thickness_filter=use_thickness_filter,
                min_thickness=min_thickness,
                max_thickness=max_thickness,
                punct_use_gui=punct_use_gui,
                punct_blur_sigma=punct_blur_sigma,
                punct_min_distance=punct_min_distance,
                punct_threshold_abs=punct_threshold_abs,
                punct_threshold_rel=punct_threshold_rel,
                punct_exclude_border=punct_exclude_border,
                punct_scan_radius=punct_scan_radius,
                save_overlay=save_overlay,
            )
        except Exception as exc:
            click.echo(f"  ERROR processing {image_path}: {exc}")
            continue

        df, radial_df = result
        if df is not None and len(df):
            per_image_dfs.append(df)
        if radial_df is not None and len(radial_df):
            per_image_radial_dfs.append(radial_df)

    _save_pooled_outputs(per_image_dfs, out_dir)
    _save_radial_profile_outputs(per_image_radial_dfs, out_dir,
                                 radius=int(punct_scan_radius))
    click.echo(f"Pooled outputs written to {out_dir}")


if __name__ == '__main__':
    main()
