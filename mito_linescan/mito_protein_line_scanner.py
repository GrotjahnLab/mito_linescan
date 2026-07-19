
import sys
import os
import glob
from typing import NamedTuple

import click
import tqdm
import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt

import sknw
import networkx as nx
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import pandas as pd
#from scipy.signal import find_peaks, peak_prominences   
import scipy.interpolate as interpolate

#For local otsu thresholding
import skimage as ski
from skimage import exposure 
from skimage.morphology import disk
from skimage.filters import  rank
from skimage.util import img_as_ubyte
##################
from skimage.morphology import skeletonize
import sknw

from scipy.signal import find_peaks, peak_prominences


class MitoMaskResult(NamedTuple):
    """Return type of the ridge-filter mask builders.

    BP-2: producing a named 5-tuple means a caller that unpacks the wrong
    number of values fails loudly at the call site instead of silently
    mis-binding fields.
    """
    threshold: float
    binary: object
    skeleton: object
    graph: object
    params: dict


# Global variables for colormaps (initialized on first use)
_mito_cmap = None
_scan_cmap = None

def get_colormaps():
    """Get or create custom colormaps for mitochondria and scan visualization."""
    global _mito_cmap, _scan_cmap
    
    if _mito_cmap is not None and _scan_cmap is not None:
        return _mito_cmap, _scan_cmap
    
    # Mitochondria colormap
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.sqrt(np.linspace(0/256, 1, N))
    vals[:, 1] = np.sqrt(np.linspace(0/256, 64/256, N))
    vals[:, 2] = np.sqrt(np.linspace(0/256, 1, N))
    vals[:, 3] = np.sqrt(np.linspace(0/256, 256/256, N))
    _mito_cmap = ListedColormap(vals)

    # Scan colormap
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.sqrt(np.linspace(0/256, 64/256, N))
    vals[:, 1] = np.sqrt(np.linspace(64/256, 1, N))
    vals[:, 2] = np.sqrt(np.linspace(0/256, 64/256, N))
    vals[:, 3] = np.sqrt(np.linspace(0/256, 256/256, N))
    _scan_cmap = ListedColormap(vals)
    
    return _mito_cmap, _scan_cmap






def lasso_select_cell(mito_image, protein_image=None):
    """
    Launch a lasso selection GUI to select a cell region.

    Parameters:
    - mito_image:    2D numpy array, mitochondria channel (shown first).
    - protein_image: 2D numpy array, protein channel (optional). When
                     provided, a "Switch Channel" button lets the user toggle
                     between the two channels to draw the lasso on whichever
                     gives a clearer cell boundary.

    Returns:
    - mask: 2D boolean numpy array where True indicates selected region.
    """
    from matplotlib.widgets import LassoSelector, Slider, Button
    from matplotlib.path import Path

    channels = [mito_image]
    ch_names = ['Mito']
    if protein_image is not None:
        channels.append(protein_image)
        ch_names.append('Protein')

    state = {'ch_idx': 0}

    def current_img():
        return channels[state['ch_idx']]

    CH_COLORS = ['cyan', 'orange']   # mito → cyan, protein → orange

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.22)  # room for sliders + button

    img0 = current_img()
    vmin0 = float(np.percentile(img0, 1))
    vmax0 = float(np.percentile(img0, 99))
    im = ax.imshow(img0, cmap='gray', vmin=vmin0, vmax=vmax0)

    # mask overlay — updated after each lasso draw
    mask_rgba = np.zeros((*mito_image.shape, 4), dtype=np.float32)
    im_mask = ax.imshow(mask_rgba)

    # ---- legend in the top-left corner of the image ----
    legend_lines = []
    legend_entries = [
        ('white',  'Lasso path'),
        ('red',    'Selected region (mask)'),
    ]
    for ci, name in enumerate(ch_names):
        legend_entries.append((CH_COLORS[ci], f'Ch {ci}: {name}'))

    for i, (color, label) in enumerate(legend_entries):
        t = ax.text(
            0.01, 0.99 - i * 0.055, f'■ {label}',
            transform=ax.transAxes,
            color=color, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55, ec='none'),
        )
        legend_lines.append(t)

    # channel indicator — bold colored text showing which channel is active
    ch_indicator = ax.text(
        0.5, 0.01, '',
        transform=ax.transAxes,
        fontsize=11, fontweight='bold', ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.65, ec='none'),
    )

    def _update_channel_indicator():
        idx = state['ch_idx']
        ch_indicator.set_text(f'Viewing: {ch_names[idx]} channel')
        ch_indicator.set_color(CH_COLORS[idx])
        fig.canvas.draw_idle()

    def _title():
        ax.set_title('Lasso to select cell — close without drawing to use whole image')

    _title()
    _update_channel_indicator()

    axcolor = 'lightgoldenrodyellow'
    ax_min = fig.add_axes([0.15, 0.12, 0.65, 0.03], facecolor=axcolor)
    ax_max = fig.add_axes([0.15, 0.07, 0.65, 0.03], facecolor=axcolor)

    pmin = Slider(ax_min, 'Min', float(img0.min()), float(img0.max()), valinit=vmin0)
    pmax = Slider(ax_max, 'Max', float(img0.min()), float(img0.max()), valinit=vmax0)

    def update_contrast(_val):
        vmin, vmax = pmin.val, pmax.val
        if vmin < vmax:
            im.set_clim(vmin, vmax)
            fig.canvas.draw_idle()

    pmin.on_changed(update_contrast)
    pmax.on_changed(update_contrast)

    # "Switch Channel" button — only useful when a second channel is provided
    if protein_image is not None:
        ax_switch = fig.add_axes([0.82, 0.08, 0.15, 0.06])
        btn_switch = Button(ax_switch, 'Switch\nChannel',
                            color='lightblue', hovercolor='skyblue')

        def on_switch(_event):
            state['ch_idx'] = 1 - state['ch_idx']
            img = current_img()
            im.set_data(img)
            vmin_new = float(np.percentile(img, 1))
            vmax_new = float(np.percentile(img, 99))
            im.set_clim(vmin_new, vmax_new)
            # reset sliders to the new channel's range
            pmin.valmin = float(img.min())
            pmin.valmax = float(img.max())
            pmin.set_val(vmin_new)
            pmax.valmin = float(img.min())
            pmax.valmax = float(img.max())
            pmax.set_val(vmax_new)
            _update_channel_indicator()
            fig.canvas.draw_idle()

        btn_switch.on_clicked(on_switch)

    mask = np.zeros(mito_image.shape, dtype=bool)
    drew_lasso = False

    def onselect(verts):
        path = Path(verts)
        y, x = np.mgrid[:mito_image.shape[0], :mito_image.shape[1]]
        points = np.vstack((x.flatten(), y.flatten())).T
        mask_flat = path.contains_points(points)
        nonlocal mask, drew_lasso
        mask = mask_flat.reshape(mito_image.shape)
        drew_lasso = True
        # update red overlay so the selected region is visible
        rgba = np.zeros((*mito_image.shape, 4), dtype=np.float32)
        rgba[mask, 0] = 1.0   # red
        rgba[mask, 3] = 0.35  # semi-transparent
        im_mask.set_data(rgba)
        fig.canvas.draw_idle()

    lasso = LassoSelector(ax, onselect)  # noqa: F841 — must stay referenced

    plt.show()

    if not drew_lasso or not mask.any():
        print("No lasso drawn -- using the entire image as the cell ROI.")
        mask = np.ones(mito_image.shape, dtype=bool)
    else:
        print(f"Lasso ROI: {int(mask.sum()):,} px "
              f"({100 * mask.mean():.1f}% of image)")
    return mask




def select_threshold(image):
    """
    Display an image with an interactive slider to pick a threshold.
    Returns the chosen threshold (float).
    """
    import matplotlib.pyplot as plt

    img = np.array(image, copy=False)
    # robust display range
    vmin0 = float(np.percentile(img, 1))
    vmax0 = float(np.percentile(img, 99))

    # initial threshold: midpoint of display range
    init_thresh = float((vmin0 + vmax0) / 2.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)
    ax.set_title("Adjust threshold with the slider. Click Done when finished.")
    im = ax.imshow(img, cmap="gray", vmin=vmin0, vmax=vmax0)
    overlay = ax.imshow((img > init_thresh).astype(np.uint8), cmap=plt.cm.Reds, alpha=0.4, vmin=0, vmax=1)

    axcolor = 'lightgoldenrodyellow'
    ax_slider = fig.add_axes([0.15, 0.12, 0.7, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'Threshold', float(img.min()), float(img.max()), valinit=init_thresh)

    done = {'pressed': False}

    def update(val):
        thr = slider.val
        mask = (img > thr).astype(np.uint8)
        overlay.set_data(mask)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    ax_done = fig.add_axes([0.85, 0.02, 0.1, 0.05])
    btn = Button(ax_done, 'Done')

    def on_done(event):
        done['pressed'] = True
        plt.close(fig)

    btn.on_clicked(on_done)

    plt.show()

    # If the user closed the window manually, return the current slider value
    return float(slider.val)

def select_threshold_gui(image):
    """
    Display an image with an interactive slider to pick a threshold.
    Returns the chosen threshold (float).
    """
    import matplotlib.pyplot as plt

    img = np.array(image, copy=False)
    # robust display range
    vmin0 = float(np.percentile(img, 1))
    vmax0 = float(np.percentile(img, 99))

    # initial threshold: midpoint of display range
    init_thresh = float((vmin0 + vmax0) / 2.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)
    ax.set_title("Adjust threshold with the slider. Click Done when finished.")
    im = ax.imshow(img, cmap="gray", vmin=vmin0, vmax=vmax0)
    overlay = ax.imshow((img > init_thresh).astype(np.uint8), cmap=plt.cm.Reds, alpha=0.4, vmin=0, vmax=1)

    axcolor = 'lightgoldenrodyellow'
    ax_slider = fig.add_axes([0.15, 0.12, 0.7, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'Threshold', float(img.min()), float(img.max()), valinit=init_thresh)

    done = {'pressed': False}

    def update(val):
        thr = slider.val
        mask = (img > thr).astype(np.uint8)
        overlay.set_data(mask)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    ax_done = fig.add_axes([0.85, 0.02, 0.1, 0.05])
    btn = Button(ax_done, 'Done')

    def on_done(event):
        done['pressed'] = True
        plt.close(fig)

    btn.on_clicked(on_done)

    plt.show()

    # If the user closed the window manually, return the current slider value
    binarization_threshold = float(slider.val)
    mito_binary = img > binarization_threshold
    mito_skeleton, mito_nx = binary_to_sknw(mito_binary)
    
    # track whether user explicitly confirmed the threshold
    threshold_confirmed = False

    show_graph = True
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')

    # prepare initial network position mapping
    nodes = mito_nx.nodes()
    pos = {n: (nodes[n]['o'][1], nodes[n]['o'][0]) for n in nodes}
    node_labels = {node: node for node in mito_nx.nodes()}

    # draw the underlying images once (keeps extent/zoom consistent)
    im_mito = ax.imshow(image, cmap='gray', alpha=0.6)
    im_skel = ax.imshow(mito_skeleton, cmap=mito_cmap, alpha=1, visible=show_graph)

    # draw network elements and keep references to the artists so we can toggle visibility
    #edge_art = nx.draw_networkx_edges(mito_nx, pos, ax=ax, edge_color='yellow', alpha=0.7)
    node_art = nx.draw_networkx_nodes(mito_nx, pos, ax=ax, node_color='red', node_size=50, alpha=0.9)
    label_art = nx.draw_networkx_labels(mito_nx, pos, labels=node_labels, font_color='white', ax=ax)

    # helper that sets visibility for either a single artist or an iterable of artists
    def _set_visible(art, visible):
        if art is None:
            return
        try:
            # many NetworkX draw functions return a single Matplotlib collection,
            # but some return lists/iterables; handle both.
            for a in art:
                a.set_visible(visible)
        except TypeError:
            art.set_visible(visible)

    # apply initial visibility and keep zoom/pan behavior stable
    #_set_visible(edge_art, show_graph)
    _set_visible(node_art, show_graph)
    for t in label_art.values():
        t.set_visible(show_graph)
    ax.set_title("Mitochondria Skeleton and Network")
    ax.autoscale(enable=False)  # ensure toggling doesn't change axis limits

    # add a toggle button to show/hide the graph
    ax_button = fig.add_axes([0.85, 0.92, 0.12, 0.05])
    btn_toggle = Button(ax_button, 'Hide Graph' if show_graph else 'Show Graph')

    def on_toggle(event):
        nonlocal show_graph
        show_graph = not show_graph
        btn_toggle.label.set_text('Hide Graph' if show_graph else 'Show Graph')
        # toggle visibility of the skeleton and network artists without clearing/redrawing the axes
        im_skel.set_visible(show_graph)
        #_set_visible(edge_art, show_graph)
        _set_visible(node_art, show_graph)
        for t in label_art.values():
            t.set_visible(show_graph)
        fig.canvas.draw_idle()

    btn_toggle.on_clicked(on_toggle)


    #add another button to labeled "Confirm Threshold"
    ax_button_confirm = fig.add_axes([0.7, 0.92, 0.12, 0.05])
    btn_confirm = Button(ax_button_confirm, 'Confirm Threshold')
    def on_confirm(event):
        nonlocal threshold_confirmed
        # close the specific figure to ensure the GUI window is closed
        plt.close(fig)
        threshold_confirmed = True
        # reference the event to avoid unused-parameter warnings
        _ = event
        # reopen the threshold GUI and update the local outputs
        #binarization_threshold, mito_binary, mito_skeleton, mito_nx = select_threshold_gui(image)
    btn_confirm.on_clicked(on_confirm)
    plt.draw()

    plt.show()

    if threshold_confirmed:
        print("Threshold confirmed by user.")
        return binarization_threshold, mito_binary, mito_skeleton, mito_nx
    else:
        # If the user did not confirm, re-open the GUI and return its result
        # so the caller always receives the expected tuple instead of None.
        return select_threshold_gui(image)

def binary_to_sknw(binary_image):
    mito_skeleton = skeletonize(binary_image, method='lee')
    mito_nx = sknw.build_sknw(mito_skeleton, multi=True)
    return mito_skeleton, mito_nx


def _local_thickness_2d(binary):
    """
    Compute a 2D local thickness map (in pixel units) on a binary array
    using the `localthickness` PyPI package. Returns None if the package is
    not installed or the call fails — callers should handle that gracefully.

    Reference: https://pypi.org/project/localthickness/
    """
    try:
        import localthickness as lt  # noqa: F401
    except ImportError as exc:
        print(f"  [thickness] localthickness not installed ({exc}); "
              "skipping. Add `localthickness` to your conda env.")
        return None

    bn = np.ascontiguousarray(binary.astype(np.uint8))
    # Defensive: the package historically exposes `local_thickness` but a few
    # versions used `local_thickness_2d`. Try both.
    try:
        thk = lt.local_thickness(bn)
    except AttributeError:
        try:
            thk = lt.local_thickness_2d(bn)
        except Exception as exc:
            print(f"  [thickness] localthickness API mismatch: {exc}")
            return None
    except Exception as exc:
        print(f"  [thickness] localthickness call failed: {exc}")
        return None

    thk = np.asarray(thk, dtype=np.float32)
    if thk.shape != bn.shape:
        print(f"  [thickness] unexpected shape {thk.shape} vs {bn.shape}; "
              "skipping.")
        return None
    return thk


def select_mask_gui(
    image,
    *,
    tubule_radius=2.0,
    sensitivity=1.0,
    min_object_size=30,
    gap_closing=1,
    use_thickness_filter=False,
    min_thickness=1.0,
    max_thickness=20.0,
):
    """
    Interactive ridge-filter pipeline for selecting a mitochondrial binary
    mask + skeleton + network graph. Returns the same 4-tuple shape as
    select_threshold_gui so the rest of the pipeline is unchanged:

        (threshold_value, mito_binary, mito_skeleton, mito_nx)

    Controls (top to bottom):
      - Tubule radius (px)     : sets the structural scale of the ridge filter
                                 and the white-top-hat disk. Slow to recompute,
                                 so it only updates when you press "Recompute
                                 ridge". For ~0.17 um/px decon data with ~0.3
                                 -0.6 um mito tubules, ~2 px is right.
      - Sensitivity            : multiplier on the Otsu threshold of the ridge
                                 response. 1.0 = pure Otsu, <1 = more permissive
                                 (catches dim tubules), >1 = stricter.
      - Min size (px)          : drops connected components smaller than this,
                                 i.e. speckle / debris.
      - Gap closing (px)       : disk radius for binary closing to bridge 1-2 px
                                 breaks in tubules before skeletonization.
      - Min / Max thickness    : a *local thickness* map (computed on the
                                 binary mask via the `localthickness` PyPI
                                 package) is used to flag regions outside the
                                 [min, max] range. Critically, this exclusion
                                 is applied AFTER skeletonization so it does
                                 not introduce artefacts into the skeleton
                                 topology; we just drop skeleton pixels whose
                                 underlying local thickness is out of range
                                 and rebuild the network graph from the
                                 filtered skeleton.

    View toggles let you overlay: the binary mask, the full skeleton, the
    network nodes, the raw ridge response, and the excluded-by-thickness
    regions. Stats below the figure summarise the current mask.

    All defaults above can be supplied by the caller (CLI / config) so that
    the GUI opens at the user's preferred starting point.
    """
    from matplotlib.widgets import Slider, Button, CheckButtons
    from skimage.filters import meijering, threshold_otsu
    from skimage.morphology import (
        white_tophat, disk, remove_small_objects, remove_small_holes,
        binary_closing, skeletonize as _skel,
    )

    img = np.asarray(image, dtype=np.float32)
    # Normalize to 0..1 for filter stability (top-hat / Meijering both like
    # a well-behaved input range). We don't change the upstream pipeline.
    a = float(np.percentile(img, 1))
    b = float(np.percentile(img, 99.5))
    img_n = np.clip((img - a) / max(b - a, 1e-9), 0, 1).astype(np.float32)

    # remember defaults so "Reset" goes back to the caller-supplied values
    DEFAULTS = dict(
        tubule_r=float(tubule_radius),
        sensitivity=float(sensitivity),
        min_size=int(min_object_size),
        close_r=int(gap_closing),
        use_thickness=bool(use_thickness_filter),
        min_thick=float(min_thickness),
        max_thick=float(max_thickness),
    )

    # --------- mutable state shared by callbacks ---------
    state = {
        'tubule_r': DEFAULTS['tubule_r'],
        'sensitivity': DEFAULTS['sensitivity'],
        'min_size': DEFAULTS['min_size'],
        'close_r': DEFAULTS['close_r'],
        'use_thickness': DEFAULTS['use_thickness'],
        'min_thick': DEFAULTS['min_thick'],
        'max_thick': DEFAULTS['max_thick'],
        # cached intermediates
        'ridge': None,
        'binary': None,
        'thickness': None,    # local-thickness map (same shape as binary)
        'skel_full': None,    # skeleton built on un-filtered binary
        'excluded': None,     # binary mask of "out of [min,max] thickness"
        'skel': None,         # skel_full AND NOT excluded
        'nx': None,
        'otsu_t': None,
        'done': False,
        # view toggles
        'show_binary': True,
        'show_skel': True,
        'show_nodes': False,
        'show_ridge': False,
        'show_excluded': True,
        'show_labels': False,
    }

    def compute_ridge():
        r = state['tubule_r']
        # top-hat disk ~ 4x tubule radius removes broad background
        th_radius = max(3, int(round(4 * r)))
        th = white_tophat(img_n, disk(th_radius))
        # multi-scale ridge sigmas spanning the expected tubule radius
        sigmas = sorted({round(s, 2) for s in
                         [max(0.5, 0.5 * r), r, 1.5 * r, 2.0 * r]})
        ridge = meijering(th, sigmas=sigmas, black_ridges=False)
        state['ridge'] = ridge.astype(np.float32)
        pos = ridge[ridge > 0]
        state['otsu_t'] = float(threshold_otsu(pos)) if pos.size else 0.0

    def compute_binary():
        if state['ridge'] is None:
            return
        t = state['otsu_t'] * state['sensitivity']
        bn = state['ridge'] > t
        if state['close_r'] > 0:
            bn = binary_closing(bn, disk(int(round(state['close_r']))))
        bn = remove_small_objects(bn, min_size=int(state['min_size']))
        bn = remove_small_holes(bn, area_threshold=int(state['min_size']))
        state['binary'] = bn
        # binary changed -> invalidate thickness so it gets recomputed
        state['thickness'] = None

    def compute_thickness():
        if state['binary'] is None:
            state['thickness'] = None
            return
        state['thickness'] = _local_thickness_2d(state['binary'])

    def compute_excluded():
        """Build the 'excluded by thickness' mask (does NOT modify binary)."""
        state['excluded'] = None
        if not state['use_thickness']:
            return
        thk = state['thickness']
        bn = state['binary']
        if thk is None or bn is None:
            return
        lo, hi = state['min_thick'], state['max_thick']
        if lo > hi:
            lo, hi = hi, lo
        out_of_range = (thk < lo) | (thk > hi)
        state['excluded'] = bn & out_of_range

    def compute_skeleton():
        """Build the *unfiltered* skeleton from the current binary."""
        if state['binary'] is None:
            return
        skel_full = _skel(state['binary'], method='lee').astype(bool)
        state['skel_full'] = skel_full

    def apply_thickness_filter_and_build_graph():
        """Filter the skeleton with the excluded mask (post-skeletonization),
        then rebuild the sknw graph from the *filtered* skeleton."""
        if state['skel_full'] is None:
            return
        if state['excluded'] is not None:
            skel = state['skel_full'] & ~state['excluded']
        else:
            skel = state['skel_full']
        state['skel'] = skel
        try:
            g = sknw.build_sknw(skel.astype(np.uint8), multi=True)
        except Exception as exc:
            print(f"  [select_mask_gui] sknw.build_sknw failed: {exc}")
            g = nx.MultiGraph()
        state['nx'] = g

    def recompute_from_binary():
        """When binary changes: thickness, full skeleton, excluded, filter."""
        compute_thickness()
        compute_skeleton()
        compute_excluded()
        apply_thickness_filter_and_build_graph()

    def recompute_from_thickness():
        """When thickness range / filter toggle changes (thickness cached)."""
        compute_excluded()
        apply_thickness_filter_and_build_graph()

    def recompute_all():
        compute_ridge()
        compute_binary()
        recompute_from_binary()

    recompute_all()

    # --------- figure layout ---------
    fig = plt.figure(figsize=(11, 11))
    # leave a taller bottom strip for the extra sliders + checkboxes
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.40)
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')

    vmin = float(np.percentile(img, 1))
    vmax = float(np.percentile(img, 99))
    ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)

    # ridge map overlay (toggle)
    rmax = float(np.percentile(state['ridge'], 99.5)) + 1e-9
    im_ridge = ax.imshow(state['ridge'], cmap='inferno', alpha=0.55,
                         vmin=0, vmax=rmax, visible=state['show_ridge'])

    # binary overlay (red, alpha)
    bin_rgba = np.zeros((*img.shape, 4), dtype=np.float32)
    im_bin = ax.imshow(bin_rgba, visible=state['show_binary'])

    # excluded-by-thickness overlay (cyan dots so it visually contrasts
    # with the red binary and green skeleton)
    excl_rgba = np.zeros((*img.shape, 4), dtype=np.float32)
    im_excl = ax.imshow(excl_rgba, visible=state['show_excluded'])

    # skeleton overlay (green) -- shows the FILTERED skeleton (post-thickness)
    skel_rgba = np.zeros((*img.shape, 4), dtype=np.float32)
    im_skel = ax.imshow(skel_rgba, visible=state['show_skel'])

    # network nodes scatter
    node_scat = ax.scatter([], [], c='yellow', s=15,
                           edgecolors='black', linewidths=0.4,
                           visible=state['show_nodes'])

    ax.set_title("Mito mask GUI - tune sliders, click Done when satisfied")
    ax.set_xticks([])
    ax.set_yticks([])

    # --------- sliders ---------
    axcolor = 'lightgoldenrodyellow'
    ax_tube = fig.add_axes([0.10, 0.33, 0.55, 0.022], facecolor=axcolor)
    ax_sens = fig.add_axes([0.10, 0.29, 0.55, 0.022], facecolor=axcolor)
    ax_min = fig.add_axes([0.10, 0.25, 0.55, 0.022], facecolor=axcolor)
    ax_clos = fig.add_axes([0.10, 0.21, 0.55, 0.022], facecolor=axcolor)
    ax_tmin = fig.add_axes([0.10, 0.17, 0.55, 0.022], facecolor=axcolor)
    ax_tmax = fig.add_axes([0.10, 0.13, 0.55, 0.022], facecolor=axcolor)

    s_tube = Slider(ax_tube, 'Tubule radius (px)', 1.0, 6.0,
                    valinit=state['tubule_r'], valstep=0.1)
    s_sens = Slider(ax_sens, 'Sensitivity', 0.3, 2.0,
                    valinit=state['sensitivity'], valstep=0.02)
    s_min = Slider(ax_min, 'Min size (px)', 5, 300,
                   valinit=state['min_size'], valstep=1)
    s_clos = Slider(ax_clos, 'Gap closing (px)', 0, 3,
                    valinit=state['close_r'], valstep=1)
    s_tmin = Slider(ax_tmin, 'Min thickness (px)', 0.0, 30.0,
                    valinit=state['min_thick'], valstep=0.5)
    s_tmax = Slider(ax_tmax, 'Max thickness (px)', 0.0, 30.0,
                    valinit=state['max_thick'], valstep=0.5)

    # --------- buttons ---------
    ax_recompute = fig.add_axes([0.68, 0.33, 0.14, 0.035])
    btn_recompute = Button(ax_recompute, 'Recompute ridge',
                           color='lightblue', hovercolor='skyblue')

    ax_reset = fig.add_axes([0.68, 0.13, 0.14, 0.035])
    btn_reset = Button(ax_reset, 'Reset defaults',
                       color='lightcoral', hovercolor='salmon')

    ax_done = fig.add_axes([0.84, 0.13, 0.13, 0.035])
    btn_done = Button(ax_done, 'Done',
                      color='lightgreen', hovercolor='palegreen')

    # --------- view toggles + thickness-filter toggle ---------
    ax_check = fig.add_axes([0.68, 0.16, 0.29, 0.16])
    check = CheckButtons(
        ax_check,
        ['Binary', 'Skeleton', 'Nodes', 'Ridge', 'Excluded', 'Filter ON', 'Labels'],
        [state['show_binary'], state['show_skel'],
         state['show_nodes'], state['show_ridge'],
         state['show_excluded'], state['use_thickness'],
         state['show_labels']],
    )

    # list to track node/edge label text artists so we can clear them on toggle
    _label_artists = []

    # --------- stats text ---------
    stats_ax = fig.add_axes([0.10, 0.05, 0.86, 0.05])
    stats_ax.axis('off')
    stats_text = stats_ax.text(0.0, 0.5, '', va='center', fontsize=10,
                               family='monospace')

    def refresh_display():
        # binary overlay (red)
        rgba_b = np.zeros((*img.shape, 4), dtype=np.float32)
        if state['binary'] is not None:
            rgba_b[state['binary'], 0] = 1.0
            rgba_b[state['binary'], 3] = 0.35
        im_bin.set_data(rgba_b)

        # excluded-by-thickness overlay (cyan)
        rgba_e = np.zeros((*img.shape, 4), dtype=np.float32)
        if state['excluded'] is not None and state['excluded'].any():
            rgba_e[state['excluded'], 1] = 1.0  # G
            rgba_e[state['excluded'], 2] = 1.0  # B  -> cyan
            rgba_e[state['excluded'], 3] = 0.55
        im_excl.set_data(rgba_e)

        # skeleton overlay (green) -- filtered skeleton
        rgba_s = np.zeros((*img.shape, 4), dtype=np.float32)
        if state['skel'] is not None:
            sk = state['skel'].astype(bool)
            rgba_s[sk, 1] = 1.0
            rgba_s[sk, 3] = 1.0
        im_skel.set_data(rgba_s)

        # ridge
        if state['ridge'] is not None:
            im_ridge.set_data(state['ridge'])
            rmx = float(np.percentile(state['ridge'], 99.5)) + 1e-9
            im_ridge.set_clim(0, rmx)

        # nodes
        if state['nx'] is not None and state['nx'].number_of_nodes() > 0:
            nodes = state['nx'].nodes()
            pts = np.array([[nodes[n]['o'][1], nodes[n]['o'][0]]
                            for n in nodes])
            node_scat.set_offsets(pts)
        else:
            node_scat.set_offsets(np.empty((0, 2)))

        # visibility
        im_bin.set_visible(state['show_binary'])
        im_skel.set_visible(state['show_skel'])
        node_scat.set_visible(state['show_nodes'])
        im_ridge.set_visible(state['show_ridge'])
        im_excl.set_visible(state['show_excluded'] and state['use_thickness'])

        # stats
        bin_pct = (100 * state['binary'].mean()
                   if state['binary'] is not None else 0.0)
        skel_px = (int(state['skel'].astype(bool).sum())
                   if state['skel'] is not None else 0)
        skel_full_px = (int(state['skel_full'].astype(bool).sum())
                        if state['skel_full'] is not None else 0)
        dropped = skel_full_px - skel_px
        n_nodes = (state['nx'].number_of_nodes()
                   if state['nx'] is not None else 0)
        n_edges = (state['nx'].number_of_edges()
                   if state['nx'] is not None else 0)

        long_paths = 0
        if state['nx'] is not None:
            for u, v, k in state['nx'].edges(keys=True):
                pts = state['nx'][u][v][k].get('pts')
                if pts is not None and len(pts) >= 30:
                    long_paths += 1

        otsu = state['otsu_t'] or 0.0
        thr = otsu * state['sensitivity']
        if state['thickness'] is not None and state['binary'] is not None \
                and state['binary'].any():
            tvals = state['thickness'][state['binary']]
            thick_summary = (f"thickness: p50={np.median(tvals):.1f} "
                             f"p95={np.percentile(tvals, 95):.1f}px")
        else:
            thick_summary = "thickness: n/a"
        filt_state = "ON" if state['use_thickness'] else "off"
        stats_text.set_text(
            f"binary: {bin_pct:5.2f}%   "
            f"skel(filt): {skel_px:>6,}   "
            f"dropped: {dropped:>5,}   "
            f"nodes: {n_nodes:>4}   "
            f"edges: {n_edges:>4}   "
            f"paths>=30: {long_paths:>3}   "
            f"thr: {thr:.3f}   "
            f"{thick_summary}   "
            f"filter[{state['min_thick']:.1f}-{state['max_thick']:.1f}]: {filt_state}"
        )
        # node / edge labels
        for art in _label_artists:
            art.remove()
        _label_artists.clear()
        if state['show_labels'] and state['nx'] is not None:
            g = state['nx']
            nodes_d = g.nodes()
            h, w = img.shape[:2]
            # node IDs — skip any node whose position is outside the image
            for n in nodes_d:
                y, x = nodes_d[n]['o']
                if not (0 <= x < w and 0 <= y < h):
                    continue
                txt = ax.text(
                    x, y, str(n),
                    color='yellow', fontsize=9, ha='center', va='center',
                    clip_on=True,
                    bbox=dict(boxstyle='round,pad=0.1', fc='black',
                              alpha=0.6, ec='none'),
                )
                txt.set_clip_box(ax.bbox)
                _label_artists.append(txt)

        fig.canvas.draw_idle()

    refresh_display()

    # ---- slider callbacks ----
    def on_sens(_val):
        state['sensitivity'] = float(s_sens.val)
        compute_binary()
        recompute_from_binary()
        refresh_display()

    def on_min(_val):
        state['min_size'] = int(s_min.val)
        compute_binary()
        recompute_from_binary()
        refresh_display()

    def on_clos(_val):
        state['close_r'] = int(s_clos.val)
        compute_binary()
        recompute_from_binary()
        refresh_display()

    def on_tube(_val):
        # don't recompute on every drag (the ridge filter is slow);
        # just remember the new value and wait for the button.
        state['tubule_r'] = float(s_tube.val)

    def on_tmin(_val):
        state['min_thick'] = float(s_tmin.val)
        recompute_from_thickness()
        refresh_display()

    def on_tmax(_val):
        state['max_thick'] = float(s_tmax.val)
        recompute_from_thickness()
        refresh_display()

    s_sens.on_changed(on_sens)
    s_min.on_changed(on_min)
    s_clos.on_changed(on_clos)
    s_tube.on_changed(on_tube)
    s_tmin.on_changed(on_tmin)
    s_tmax.on_changed(on_tmax)

    # ---- button callbacks ----
    def on_recompute(_event):
        state['tubule_r'] = float(s_tube.val)
        recompute_all()
        refresh_display()
    btn_recompute.on_clicked(on_recompute)

    def on_reset(_event):
        # set sliders -> callbacks update state
        s_tube.set_val(DEFAULTS['tubule_r'])
        s_sens.set_val(DEFAULTS['sensitivity'])
        s_min.set_val(DEFAULTS['min_size'])
        s_clos.set_val(DEFAULTS['close_r'])
        s_tmin.set_val(DEFAULTS['min_thick'])
        s_tmax.set_val(DEFAULTS['max_thick'])
        # also reset thickness-filter toggle through the CheckButtons
        if state['use_thickness'] != DEFAULTS['use_thickness']:
            try:
                # idx 5 is 'Filter ON' in the CheckButtons list
                check.set_active(5)
            except Exception:
                state['use_thickness'] = DEFAULTS['use_thickness']
        state['tubule_r'] = DEFAULTS['tubule_r']
        state['sensitivity'] = DEFAULTS['sensitivity']
        state['min_size'] = DEFAULTS['min_size']
        state['close_r'] = DEFAULTS['close_r']
        state['min_thick'] = DEFAULTS['min_thick']
        state['max_thick'] = DEFAULTS['max_thick']
        state['use_thickness'] = DEFAULTS['use_thickness']
        recompute_all()
        refresh_display()
    btn_reset.on_clicked(on_reset)

    def on_done(_event):
        state['done'] = True
        plt.close(fig)
    btn_done.on_clicked(on_done)

    # ---- check buttons ----
    def on_check(label):
        if label == 'Binary':
            state['show_binary'] = not state['show_binary']
            refresh_display()
        elif label == 'Skeleton':
            state['show_skel'] = not state['show_skel']
            refresh_display()
        elif label == 'Nodes':
            state['show_nodes'] = not state['show_nodes']
            refresh_display()
        elif label == 'Ridge':
            state['show_ridge'] = not state['show_ridge']
            refresh_display()
        elif label == 'Excluded':
            state['show_excluded'] = not state['show_excluded']
            refresh_display()
        elif label == 'Filter ON':
            state['use_thickness'] = not state['use_thickness']
            # thickness may not be computed yet if it was always off
            if state['use_thickness'] and state['thickness'] is None:
                compute_thickness()
            recompute_from_thickness()
            refresh_display()
        elif label == 'Labels':
            state['show_labels'] = not state['show_labels']
            refresh_display()
    check.on_clicked(on_check)

    plt.show()

    # If the user closed the window without clicking Done, still return the
    # latest computed mask (consistent with select_threshold_gui's behaviour).
    threshold_value = (state['otsu_t'] or 0.0) * state['sensitivity']
    # The returned mito_binary is the *final* user-confirmed binary:
    #   binary AND NOT excluded   when the thickness filter is on
    #   binary                    otherwise
    # This matches the filtered skeleton/graph that we also return — the
    # caller gets a self-consistent (binary, skeleton, graph) triple.
    binary_unfiltered = (state['binary']
                         if state['binary'] is not None
                         else np.zeros_like(img, dtype=bool))
    if state.get('excluded') is not None:
        binary_final = binary_unfiltered & ~state['excluded']
    else:
        binary_final = binary_unfiltered
    skel = (state['skel']
            if state['skel'] is not None
            else np.zeros_like(img, dtype=bool))
    graph = state['nx'] if state['nx'] is not None else nx.MultiGraph()
    final_params = {
        'tubule_radius':      float(state['tubule_r']),
        'sensitivity':        float(state['sensitivity']),
        'min_object_size':    int(state['min_size']),
        'gap_closing':        int(state['close_r']),
        'use_thickness_filter': bool(state['use_thickness']),
        'min_thickness':      float(state['min_thick']),
        'max_thickness':      float(state['max_thick']),
        'ridge_threshold':    float(threshold_value),
    }
    return MitoMaskResult(float(threshold_value), binary_final, skel, graph, final_params)


def compute_mito_mask_noninteractive(
    image,
    *,
    tubule_radius=2.0,
    sensitivity=1.0,
    min_object_size=30,
    gap_closing=1,
    use_thickness_filter=False,
    min_thickness=1.0,
    max_thickness=20.0,
):
    """
    Non-interactive (--no-gui) version of the ridge-filter mask pipeline.
    Same logic as select_mask_gui, same return shape:

        (threshold_value, mito_binary, mito_skeleton_filtered, mito_nx)

    The thickness filter (when enabled) is applied AFTER skeletonization, so
    it never affects the topology produced by skeletonize/sknw — it only
    removes skeleton pixels whose underlying local thickness is outside the
    [min_thickness, max_thickness] range, and the network graph is rebuilt
    from the filtered skeleton.
    """
    from skimage.filters import meijering, threshold_otsu
    from skimage.morphology import (
        white_tophat, disk, remove_small_objects, remove_small_holes,
        binary_closing, skeletonize as _skel,
    )

    img = np.asarray(image, dtype=np.float32)
    a = float(np.percentile(img, 1))
    b = float(np.percentile(img, 99.5))
    img_n = np.clip((img - a) / max(b - a, 1e-9), 0, 1).astype(np.float32)

    r = float(tubule_radius)
    th_radius = max(3, int(round(4 * r)))
    th = white_tophat(img_n, disk(th_radius))
    sigmas = sorted({round(s, 2) for s in
                     [max(0.5, 0.5 * r), r, 1.5 * r, 2.0 * r]})
    ridge = meijering(th, sigmas=sigmas, black_ridges=False)

    pos = ridge[ridge > 0]
    otsu_t = float(threshold_otsu(pos)) if pos.size else 0.0
    thr = otsu_t * float(sensitivity)

    bn = ridge > thr
    if gap_closing > 0:
        bn = binary_closing(bn, disk(int(round(gap_closing))))
    bn = remove_small_objects(bn, min_size=int(min_object_size))
    bn = remove_small_holes(bn, area_threshold=int(min_object_size))

    skel_full = _skel(bn, method='lee').astype(bool)

    excluded = None
    if use_thickness_filter:
        thk = _local_thickness_2d(bn)
        if thk is not None:
            lo, hi = float(min_thickness), float(max_thickness)
            if lo > hi:
                lo, hi = hi, lo
            excluded = bn & ((thk < lo) | (thk > hi))

    if excluded is not None:
        skel = skel_full & ~excluded
        bn_final = bn & ~excluded
    else:
        skel = skel_full
        bn_final = bn

    try:
        graph = sknw.build_sknw(skel.astype(np.uint8), multi=True)
    except Exception as exc:
        print(f"  [compute_mito_mask_noninteractive] sknw failed: {exc}")
        graph = nx.MultiGraph()

    # Return the *final* (thickness-filtered) binary so it's self-consistent
    # with the filtered skeleton/graph.
    final_params = {
        'tubule_radius':        float(tubule_radius),
        'sensitivity':          float(sensitivity),
        'min_object_size':      int(min_object_size),
        'gap_closing':          int(gap_closing),
        'use_thickness_filter': bool(use_thickness_filter),
        'min_thickness':        float(min_thickness),
        'max_thickness':        float(max_thickness),
        'ridge_threshold':      float(thr),
    }
    return MitoMaskResult(float(thr), bn_final, skel, graph, final_params)


def process_images(
    input_dir,
    input_pattern,
    mask_dir_output,
    mask_dir_input,
    run_name,
    mito_ch,
    protein_ch,
    use_threshold_gui,
    scan_width,
    path_sampling,
    min_path_length,
    tubule_radius=2.0,
    sensitivity=1.0,
    min_object_size=30,
    gap_closing=1,
    use_thickness_filter=False,
    min_thickness=1.0,
    max_thickness=20.0,
    binary_mask_dir_output='',
    mask_ch=None,
):
    """Main processing function for analyzing mitochondrial networks."""

    # Create output directory if needed
    if not os.path.exists(mask_dir_output):
        os.makedirs(mask_dir_output)
    # Optional output directory for the *final* mito binary mask (the user-
    # confirmed binary with thickness-excluded regions removed). When empty,
    # the binary mask is not written.
    if binary_mask_dir_output and not os.path.exists(binary_mask_dir_output):
        os.makedirs(binary_mask_dir_output)
    
    # Get colormaps
    mito_cmap, scan_cmap = get_colormaps()
    
    # Find all images
    image_list = glob.glob(os.path.join(input_dir, input_pattern))
    if not image_list:
        click.echo(f"No images found matching pattern: {os.path.join(input_dir, input_pattern)}")
        return
    
    click.echo(f"Found {len(image_list)} images to process")
    
    for image in tqdm.tqdm(image_list):
        # Strip only the real TIFF (and optional .ome) extension. Filenames
        # like "20250924_SV40mef_0.1PFA_..._52.1494_decon_NaN.ome.tif" contain
        # dots inside the name; a naive split('.')[0] collapses different
        # files to the same prefix and causes them to overwrite each other.
        raw_name = os.path.basename(image)
        name_lower = raw_name.lower()
        if name_lower.endswith('.ome.tif'):
            basename = raw_name[:-len('.ome.tif')]
        elif name_lower.endswith('.ome.tiff'):
            basename = raw_name[:-len('.ome.tiff')]
        elif name_lower.endswith('.tiff'):
            basename = raw_name[:-len('.tiff')]
        elif name_lower.endswith('.tif'):
            basename = raw_name[:-len('.tif')]
        else:
            basename = os.path.splitext(raw_name)[0]
        output_dir = os.path.join(input_dir, basename + run_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img = tf.imread(image)
        mito_img = img[mito_ch, :, :]
        protein_img = img[protein_ch, :, :]
        
        # Handle mask loading or creation
        # Priority: (1) mask channel in image → (2) saved mask directory → (3) draw with lasso
        if mask_ch is not None:
            # Use a channel from the image as the cell mask (threshold at Otsu)
            cell_ch_img = img[mask_ch, :, :]
            from skimage.filters import threshold_otsu
            thr = threshold_otsu(cell_ch_img)
            mask = cell_ch_img > thr
            click.echo(f"  Cell mask from channel {mask_ch} (Otsu thr={thr:.1f}): "
                       f"{int(mask.sum()):,} px ({100*mask.mean():.1f}%)")
        elif mask_dir_input:
            mask_path = os.path.join(mask_dir_input, os.path.basename(image))
            if os.path.exists(mask_path):
                mask = tf.imread(mask_path).astype(bool)
                click.echo(f"  Loaded cell mask from {mask_path}")
            else:
                click.echo(f"  No saved mask found at {mask_path} — drawing lasso")
                mask = lasso_select_cell(mito_img, protein_img)
                tf.imwrite(os.path.join(mask_dir_output, os.path.basename(image)), mask)
        else:
            mask = lasso_select_cell(mito_img, protein_img)
            plt.imshow(mask.astype(int))
            plt.show()
            tf.imwrite(os.path.join(mask_dir_output, os.path.basename(image)), mask)

        # Prepare mitochondrial image
        mito_img = (mito_img - mito_img.min()) / (mito_img.max() - mito_img.min()) * 2 - 1
        mito_img_eq = exposure.equalize_hist(mito_img, nbins=256, mask=(mito_img > -0.9))
        plt.show()
        mito_img_eq = mito_img_eq * mask

        # Get threshold and binary image via the new ridge-filter GUI.
        # (The old single-slider GUI is still available as select_threshold_gui
        # if you want to revert: just swap select_mask_gui for it below.)
        pipeline_kwargs = dict(
            tubule_radius=tubule_radius,
            sensitivity=sensitivity,
            min_object_size=min_object_size,
            gap_closing=gap_closing,
            use_thickness_filter=use_thickness_filter,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
        )
        if use_threshold_gui:
            binarization_threshold, mito_binary, mito_skeleton, mito_nx, gui_params = \
                select_mask_gui(mito_img_eq, **pipeline_kwargs)
            click.echo(f"Selected ridge threshold: {binarization_threshold:.3f}")
        else:
            # Non-interactive: run the same pipeline using config/CLI defaults.
            binarization_threshold, mito_binary, mito_skeleton, mito_nx, gui_params = \
                compute_mito_mask_noninteractive(mito_img_eq, **pipeline_kwargs)
            click.echo(f"Computed ridge threshold (no-gui): {binarization_threshold:.3f}")

        # Save run parameters to YAML for reproducibility
        import yaml
        params_record = {
            'image': os.path.basename(image),
            'run_name': run_name,
            'channels': {
                'mito_channel':    mito_ch,
                'protein_channel': protein_ch,
                **(({'mask_channel': mask_ch}) if mask_ch is not None else {}),
            },
            'cell_mask': {
                'source': ('channel' if mask_ch is not None
                           else 'directory' if mask_dir_input else 'lasso'),
            },
            'scan': {
                'scan_width':      scan_width,
                'path_sampling':   path_sampling,
                'min_path_length': min_path_length,
            },
            'ridge_filter': gui_params,
        }
        params_path = os.path.join(
            output_dir, f"{basename}{run_name}_parameters.yml"
        )
        with open(params_path, 'w') as _pf:
            yaml.dump(params_record, _pf, default_flow_style=False, sort_keys=False)
        click.echo(f"  Parameters saved → {params_path}")

        # Optionally save the final mito binary mask (post-thickness-filter)
        # so the user can reuse it in downstream workflows. We save uint8 0/255
        # which is the most portable form for a binary mask.
        if binary_mask_dir_output:
            mito_binary_uint8 = (np.asarray(mito_binary, dtype=bool)
                                 .astype(np.uint8) * 255)
            binary_out_path = os.path.join(
                binary_mask_dir_output, f"{basename}_mito_binary.tif"
            )
            tf.imwrite(binary_out_path, mito_binary_uint8,
                       photometric='minisblack')
            click.echo(f"Wrote final mito binary mask: {binary_out_path}")

        nodes = mito_nx.nodes()
        pos = np.array([[nodes[i]['o'][1], nodes[i]['o'][0]] for i in nodes])
        node_labels = {node: node for node in mito_nx.nodes()}

        mito_scan = img[mito_ch, :, :]
        protein_scan = img[protein_ch, :, :]

        for u, v in mito_nx.edges():
            for i in range(len(mito_nx[u][v])):
                # Name files after the actual node IDs so they match the GUI labels.
                # For the common case (single edge between u and v) use "u_v";
                # for parallel edges (multi-graph) append the edge key: "u_v_k".
                edge_tag = f"{u}_{v}" if i == 0 else f"{u}_{v}_{i}"
                path = mito_nx[u][v][i]['pts']
                path_mito = path
                
                if len(path) < min_path_length:
                    continue
                    
                # Fit a spline to the path
                path_x = path[:, 1]
                path_y = path[:, 0]
                tck, uu = interpolate.splprep([path_x, path_y], s=50)
                x_i, y_i = interpolate.splev(np.linspace(0, 1, path_sampling*len(path)), tck)
                dx, dy = interpolate.splev(np.linspace(0, 1, path_sampling*len(path)), tck, der=1)

                path = np.column_stack((y_i, x_i))
                click.echo(f"Path length: {len(path_mito)}")
                
                mito_intensities = []
                scan_intensities = []
                path_length = []
                normal_x_plot = []
                normal_y_plot = []

                # Process each point in the path
                for p_ind in range(len(path_mito)):
                    point = np.array(path_mito[p_ind])
                    idx = np.argmin(np.linalg.norm(path - point, axis=1))

                    # Get the normal vector
                    normal = np.array([-dy[idx], dx[idx]])
                    normal = normal / np.linalg.norm(normal)   
                    
                    normal_x = []
                    normal_y = []
                    path_dist = 0
                    
                    # Calculate total path length up to this point
                    for ii in range(idx):
                        path_dist += np.linalg.norm(path[ii] - path[ii-1]) if ii > 0 else 0
                
                    # Find points along the normal vector
                    for dt in range(-scan_width, scan_width):
                        x = int(point[0] + dt * normal[1])
                        y = int(point[1] + dt * normal[0])
                        normal_x.append(x)
                        normal_y.append(y)
                        if p_ind % 5 == 0:
                            normal_x_plot.append(x)
                            normal_y_plot.append(y)

                    # Remove duplicates 
                    normal_x = np.array(normal_x)
                    normal_y = np.array(normal_y)
                    points = np.stack((normal_x, normal_y), axis=1)
                    unique_points = np.unique(points, axis=0)
                    normal_x = unique_points[:, 0]
                    normal_y = unique_points[:, 1]

                    # Calculate intensities along the normal, skip if the point is out of bounds
                    mito_intensity = 0
                    scan_intensity = 0
                    for j in range(len(normal_x)):
                        if (normal_x[j] < 0 or normal_x[j] >= protein_scan.shape[0] or 
                            normal_y[j] < 0 or normal_y[j] >= protein_scan.shape[1]):
                            continue
                        mito_intensity += mito_scan[normal_x[j], normal_y[j]]
                        scan_intensity += protein_scan[normal_x[j], normal_y[j]]

                    mito_intensities.append(mito_intensity / len(normal_x))
                    scan_intensities.append(scan_intensity / len(normal_x))
                    path_length.append(path_dist)

                # Plot intensities along the path
                # Create visualization and save results
                fig, ax = plt.subplots(1, 3, figsize=(25, 5), width_ratios=[1, 1, 3])

                # Plot mitochondria channel
                ax[0].imshow(mito_scan, cmap='gray', alpha=1)
                ax[0].scatter(path_mito[:, 1], path_mito[:, 0], c=cm.winter(np.array(path_length)/np.max(path_length)))
                ax[0].plot(path_x, path_y, color='blue', linewidth=1)
                ax[0].scatter(normal_y_plot, normal_x_plot, color='red', s=1)
                ax[0].set_title(f"Mito {edge_tag} - Path length: {len(path_mito)}")
                ax[0].set_facecolor('black')
                ax[0].set_xlim(np.min(path_x)-20, np.max(path_x)+20)
                ax[0].set_ylim(np.min(path_y)-20, np.max(path_y)+20)

                # Plot scan channel
                ax[1].imshow(protein_scan, cmap='gray', alpha=1)
                ax[1].scatter(path_mito[:, 1], path_mito[:, 0], c=cm.winter(np.array(path_length)/np.max(path_length)))
                ax[1].plot(path_x, path_y, color='blue', linewidth=1)
                ax[1].scatter(normal_y_plot, normal_x_plot, color='red', s=1)
                ax[1].set_title(f"Scan {edge_tag} - Path length: {len(path_mito)}")
                ax[1].set_facecolor('black')
                ax[1].set_xlim(np.min(path_x)-20, np.max(path_x)+20)
                ax[1].set_ylim(np.min(path_y)-20, np.max(path_y)+20)

                # Plot intensity profiles
                mito_intensities = np.array(mito_intensities)/np.max(mito_intensities)
                scan_intensities = np.array(scan_intensities)/np.max(scan_intensities)
                ax[2].plot(path_length, mito_intensities, color='blue', label='Mito')
                ax[2].plot(path_length, scan_intensities, color='orange', label='Scan')
                ax[2].scatter(path_length, np.zeros(len(path_length)) + 0.9*np.min([np.min(mito_intensities), np.min(scan_intensities)]),
                             c=cm.winter(np.array(path_length)/np.max(path_length)), label="Path")

                # Find peaks
                peaks, _ = find_peaks(scan_intensities, height=0)
                proms = peak_prominences(scan_intensities, peaks, wlen=10)[0]
                path_length_peaks = [path_length[i_peak] for i_peak in peaks]
                scan_intensity_peaks = [scan_intensities[i_peak] for i_peak in peaks]
                contour_heights = np.array(scan_intensity_peaks) - proms
                
                ax[2].vlines(x=path_length_peaks, ymin=contour_heights, ymax=scan_intensity_peaks)
                ax[2].plot(path_length_peaks, scan_intensity_peaks, "rx", label="Peaks")
                ax[2].set_title("Mito and Scan Intensities")
                ax[2].set_xlabel("Distance along path")
                ax[2].set_ylabel("Intensity (AU)")
                ax[2].legend()
                
                plt.savefig(f"{output_dir}/{basename}_mito_{edge_tag}_intensities.png")
                plt.close()

                # Save data to CSV
                data = {'Distance': path_length, 'Mito_Intensity': mito_intensities, 'Scan_Intensity': scan_intensities}
                df = pd.DataFrame(data)
                df.to_csv(f"{output_dir}/{basename}_mito_{edge_tag}.csv", index=False)


@click.command()
@click.option('--input-dir', default='20251021_decon_data/tiff', help='Input directory containing TIFF images')
@click.option('--input-pattern', default='snap*.tiff', help='Pattern to match input TIFF files')
@click.option('--mask-dir-output', default='20251021_decon_data/tiff/masks', help='Output directory for masks')
@click.option('--mask-dir-input', default='20251021_decon_data/tiff/masks/', help='Input directory for existing masks')
@click.option('--run-name', default='run1', help='Run name suffix for output directories')
@click.option('--mito-channel', default=0, type=int, help='0-based index for mitochondria channel')
@click.option('--protein-channel', default=2, type=int, help='0-based index for protein channel')
@click.option('--mask-channel', default=None, type=int,
              help='0-based index of a channel to use as the cell mask (Otsu threshold). '
                   'Takes priority over --mask-dir-input and lasso drawing.')
@click.option('--scan-width', default=4, type=int, help='Pixels on each side of the path for scanning')
@click.option('--path-sampling', default=5, type=int, help='Number of subpixel samples along the normal')
@click.option('--min-path-length', default=30, type=int, help='Minimum path length to process')
# --- ridge-filter mask pipeline (also live-tunable in the GUI) ---
@click.option('--tubule-radius', default=2.0, type=float,
              help='Tubule radius in px (drives top-hat disk and ridge sigmas). '
                   'Default 2.0 is right for ~0.17 um/px decon data.')
@click.option('--sensitivity', default=1.0, type=float,
              help='Multiplier on the Otsu cut of the ridge response. '
                   '1.0 = pure Otsu, <1 catches dim tubules, >1 is stricter.')
@click.option('--min-object-size', default=30, type=int,
              help='Drop binary connected components smaller than this many px.')
@click.option('--gap-closing', default=1, type=int,
              help='Binary closing disk radius (px) to bridge small breaks.')
# --- thickness filter (applied AFTER skeletonization) ---
@click.option('--use-thickness-filter/--no-use-thickness-filter', default=False,
              help='Apply a local-thickness range filter to the skeleton. '
                   'Requires the `localthickness` PyPI package.')
@click.option('--min-thickness', default=1.0, type=float,
              help='Minimum allowed local thickness (px) for skeleton pixels. '
                   'Pixels with thickness below this are dropped from the '
                   'skeleton (not from the binary).')
@click.option('--max-thickness', default=20.0, type=float,
              help='Maximum allowed local thickness (px) for skeleton pixels. '
                   'Pixels with thickness above this (e.g. bright clumps) '
                   'are dropped from the skeleton (not from the binary).')
# --- Final-binary export ---
@click.option('--binary-mask-dir-output', default='', type=str,
              help='Directory to write the final mito binary mask '
                   '(post-thickness-filter) as `{basename}_mito_binary.tif` '
                   '(uint8, 0/255). Leave empty to skip saving.')
def main(input_dir, input_pattern, mask_dir_output, mask_dir_input, run_name,
         mito_channel, protein_channel, mask_channel, scan_width, path_sampling,
         min_path_length, tubule_radius, sensitivity, min_object_size,
         gap_closing, use_thickness_filter, min_thickness, max_thickness,
         binary_mask_dir_output):
    """
    Analyze mitochondrial networks and protein distribution in fluorescence microscopy images.

    This tool processes multi-channel TIFF images to identify mitochondrial cristae structure
    and quantify protein localization along the mitochondrial network.
    """
    click.echo("Starting Mitochondrial Protein Scanner")
    click.echo(f"Input directory: {input_dir}")
    click.echo(f"Pattern: {input_pattern}")

    process_images(
        input_dir=input_dir,
        input_pattern=input_pattern,
        mask_dir_output=mask_dir_output,
        mask_dir_input=mask_dir_input,
        run_name=run_name,
        mito_ch=mito_channel,
        protein_ch=protein_channel,
        # DEBT-1/BP-1: the --use-gui/--no-gui flag was removed because
        # config_to_args emitted --no-use-gui, which never matched. The
        # network_line_scan CLI is headless-only; the interactive GUI path is
        # still reachable via select_mask_gui in the library API.
        use_threshold_gui=False,
        scan_width=scan_width,
        path_sampling=path_sampling,
        min_path_length=min_path_length,
        tubule_radius=tubule_radius,
        sensitivity=sensitivity,
        min_object_size=min_object_size,
        gap_closing=gap_closing,
        use_thickness_filter=use_thickness_filter,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        binary_mask_dir_output=binary_mask_dir_output,
        mask_ch=mask_channel,
    )
    
    click.echo("Processing complete!")


if __name__ == '__main__':
    main()