
import sys
import os
import glob

import click
import tqdm
import tiffile as tf
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






def lasso_select_cell(image):
    """
    Launch a lasso selection GUI to select a cell region in the given image.

    Parameters:
    - image: 2D numpy array representing the image.

    Returns:
    - mask: 2D boolean numpy array where True indicates selected region.
    """
    from matplotlib.widgets import LassoSelector, Slider
    from matplotlib.path import Path
    from matplotlib.widgets import Slider, Button
    import numpy as np

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.18)  # make room for sliders

    # initial contrast limits using robust percentiles
    vmin0 = float(np.percentile(image, 1))
    vmax0 = float(np.percentile(image, 99))
    im = ax.imshow(image, cmap='gray', vmin=vmin0, vmax=vmax0)
    ax.set_title('Lasso to select region. Adjust contrast with sliders below.')

    # Slider axes (normalized figure coords)
    axcolor = 'lightgoldenrodyellow'
    ax_min = fig.add_axes([0.15, 0.08, 0.7, 0.03], facecolor=axcolor)
    ax_max = fig.add_axes([0.15, 0.04, 0.7, 0.03], facecolor=axcolor)

    pmin = Slider(ax_min, 'Min', float(image.min()), float(image.max()), valinit=vmin0)
    pmax = Slider(ax_max, 'Max', float(image.min()), float(image.max()), valinit=vmax0)

    def update(val):
        vmin = pmin.val
        vmax = pmax.val
        if vmin >= vmax:
            # prevent inverted contrast range
            return
        im.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    pmin.on_changed(update)
    pmax.on_changed(update)

    mask = np.zeros(image.shape, dtype=bool)

    def onselect(verts):
        path = Path(verts)
        y, x = np.mgrid[:image.shape[0], :image.shape[1]]
        points = np.vstack((x.flatten(), y.flatten())).T
        mask_flat = path.contains_points(points)
        nonlocal mask
        mask = mask_flat.reshape(image.shape)
        # keep figure open until user closes it

    lasso = LassoSelector(ax, onselect)

    plt.show()
    print(mask.astype(int))
    return mask


def local_otsu_threshold(image, selem_radius=otsu_r, adjust_r=True):
    """Compute local Otsu threshold for the given image."""
    selem = disk(selem_radius)
    #adjust the image range to -1 to 1
    image = (image - image.min()) / (image.max() - image.min()) * 2 - 1
    image_256 = (image - image.min()) / (image.max() - image.min()) * 255

    local_otsu = rank.otsu(img_as_ubyte(image), selem)
    #if the adjust_r start a plt.imshow and let the user decide the radius
    # initial contrast limits
    vmin0 = float(np.percentile(image, 1))
    vmax0 = float(np.percentile(image, 99))

    # clamp initial radius and compute a sensible max
    selem_radius = max(1, int(selem_radius))
    max_rad = max(1, min(image.shape) // 10)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.28)

    im = ax.imshow(image, cmap='gray', vmin=vmin0, vmax=vmax0,alpha=1)
    overlay = ax.imshow((image_256 > local_otsu).astype(np.uint8),
                cmap=ListedColormap(['none', 'red']), alpha=0.5, vmin=0, vmax=1)
    ax.set_title('Adjust Otsu radius and contrast. Click Done when finished.')

    ax_rad = fig.add_axes([0.15, 0.18, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    ax_vmin = fig.add_axes([0.15, 0.12, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    ax_vmax = fig.add_axes([0.15, 0.06, 0.7, 0.03], facecolor='lightgoldenrodyellow')

    s_radius = Slider(ax_rad, 'Radius', 1, max_rad, valinit=selem_radius, valstep=1)
    s_vmin = Slider(ax_vmin, 'Min', float(image.min()), float(image.max()), valinit=vmin0)
    s_vmax = Slider(ax_vmax, 'Max', float(image.min()), float(image.max()), valinit=vmax0)

    done = {'pressed': False}

    def recompute_and_update(_=None):
        # recompute local Otsu with integer radius
        r = max(1, int(s_radius.val))
        nonlocal local_otsu
        local_otsu = rank.otsu(img_as_ubyte(image), disk(r))
        # update overlay (image > local_otsu)
        mask = (image_256 > local_otsu).astype(np.uint8)
        overlay.set_data(mask)
        # update contrast
        vmin = s_vmin.val
        vmax = s_vmax.val
        if vmin < vmax:
            im.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    s_radius.on_changed(recompute_and_update)
    s_vmin.on_changed(recompute_and_update)
    s_vmax.on_changed(recompute_and_update)

    ax_done = fig.add_axes([0.85, 0.01, 0.1, 0.04])
    btn_done = Button(ax_done, 'Done')

    def on_done(event):
        done['pressed'] = True
        plt.close(fig)

    btn_done.on_clicked(on_done)

    # initial draw
    recompute_and_update()
    plt.show()
    return local_otsu

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
    min_path_length
):
    """Main processing function for analyzing mitochondrial networks."""
    
    # Create output directory if needed
    if not os.path.exists(mask_dir_output):
        os.makedirs(mask_dir_output)
    
    # Get colormaps
    mito_cmap, scan_cmap = get_colormaps()
    
    # Find all images
    image_list = glob.glob(os.path.join(input_dir, input_pattern))
    if not image_list:
        click.echo(f"No images found matching pattern: {os.path.join(input_dir, input_pattern)}")
        return
    
    click.echo(f"Found {len(image_list)} images to process")
    
    for image in tqdm.tqdm(image_list):
        basename = os.path.basename(image).split('.')[0]
        output_dir = os.path.join(input_dir, basename + run_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img = tf.imread(image)
        mito_img = img[mito_ch, :, :]
        protein_img = img[protein_ch, :, :]
        
        # Handle mask loading or creation
        if mask_dir_input:
            mask_path = os.path.join(mask_dir_input, os.path.basename(image))
            if os.path.exists(mask_path):
                mask = tf.imread(mask_path).astype(bool)
            else:
                mask = lasso_select_cell(mito_img)
                tf.imwrite(os.path.join(mask_dir_output, os.path.basename(image)), mask)
        else:
            mask = lasso_select_cell(mito_img)
            plt.imshow(mask.astype(int))
            plt.show()
            #write out masked image
            tf.imwrite(os.path.join(mask_dir_output, os.path.basename(image)), mask)

        # Prepare mitochondrial image
        mito_img = (mito_img - mito_img.min()) / (mito_img.max() - mito_img.min()) * 2 - 1
        mito_img_eq = exposure.equalize_hist(mito_img, nbins=256, mask=(mito_img > -0.9))
        plt.show()
        mito_img_eq = mito_img_eq * mask

        # Get threshold and binary image
        if use_threshold_gui:
            binarization_threshold, mito_binary, mito_skeleton, mito_nx = select_threshold_gui(mito_img_eq)
            click.echo(f"Selected binarization threshold: {binarization_threshold:.3f}")

        nodes = mito_nx.nodes()
        pos = np.array([[nodes[i]['o'][1], nodes[i]['o'][0]] for i in nodes])
        node_labels = {node: node for node in mito_nx.nodes()}

        mito_scan = img[mito_ch, :, :]
        protein_scan = img[protein_ch, :, :]
        mito_i = 0
        
        for u, v in mito_nx.edges():
            for i in range(len(mito_nx[u][v])):
                mito_i = mito_i + 1
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
                ax[0].set_title(f"Mito {mito_i} - Path length: {len(path_mito)}")
                ax[0].set_facecolor('black')
                ax[0].set_xlim(np.min(path_x)-20, np.max(path_x)+20)
                ax[0].set_ylim(np.min(path_y)-20, np.max(path_y)+20)

                # Plot scan channel
                ax[1].imshow(protein_scan, cmap='gray', alpha=1)
                ax[1].scatter(path_mito[:, 1], path_mito[:, 0], c=cm.winter(np.array(path_length)/np.max(path_length)))
                ax[1].plot(path_x, path_y, color='blue', linewidth=1)
                ax[1].scatter(normal_y_plot, normal_x_plot, color='red', s=1)
                ax[1].set_title(f"Scan {mito_i} - Path length: {len(path_mito)}")
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
                
                plt.savefig(f"{output_dir}/{basename}_mito_{mito_i}_intensities.png")
                plt.close()

                # Save data to CSV
                data = {'Distance': path_length, 'Mito_Intensity': mito_intensities, 'Scan_Intensity': scan_intensities}
                df = pd.DataFrame(data)
                df.to_csv(f"{output_dir}/{basename}_mito_{mito_i}.csv", index=False)


@click.command()
@click.option('--input-dir', default='20251021_decon_data/tiff', help='Input directory containing TIFF images')
@click.option('--input-pattern', default='snap*.tiff', help='Pattern to match input TIFF files')
@click.option('--mask-dir-output', default='20251021_decon_data/tiff/masks', help='Output directory for masks')
@click.option('--mask-dir-input', default='20251021_decon_data/tiff/masks/', help='Input directory for existing masks')
@click.option('--run-name', default='run1', help='Run name suffix for output directories')
@click.option('--mito-channel', default=0, type=int, help='0-based index for mitochondria channel')
@click.option('--protein-channel', default=2, type=int, help='0-based index for protein channel')
@click.option('--use-gui/--no-gui', default=True, help='Use interactive GUI for threshold selection')
@click.option('--scan-width', default=4, type=int, help='Pixels on each side of the path for scanning')
@click.option('--path-sampling', default=5, type=int, help='Number of subpixel samples along the normal')
@click.option('--min-path-length', default=30, type=int, help='Minimum path length to process')
def main(input_dir, input_pattern, mask_dir_output, mask_dir_input, run_name, 
         mito_channel, protein_channel, use_gui, scan_width, path_sampling, min_path_length):
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
        use_threshold_gui=use_gui,
        scan_width=scan_width,
        path_sampling=path_sampling,
        min_path_length=min_path_length
    )
    
    click.echo("Processing complete!")


if __name__ == '__main__':
    main()