#!/usr/bin/env python3

"""
This script reads mitos and scan structures from a file and separates mitos using connected components.
Processes microscopy images to analyze mitochondrial and scan structures.
"""

import os
import click
import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import LassoSelector, Slider
from matplotlib.path import Path
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import pandas as pd
from .utils import create_colormaps


def draw_mitochondria(mito_image, scan_image):
    '''    
    Draw mitochondria region using lasso selector and return binary mask. Switch channels if they are flipped.
    '''
    y_pixels = mito_image.shape[0]
    x_pixels = mito_image.shape[1]


    fig = plt.figure()

    ax_z = fig.add_subplot(111)
    mito_cmap, scan_cmap = create_colormaps()
    ax_z.set_facecolor("black")
    ax_z.imshow(mito_image, cmap=mito_cmap, alpha = 0.7)
    ax_z.imshow(scan_image, cmap=scan_cmap, alpha = 0.3)
    ax_z.set_title("Draw mitochondria region (Z view)")

    # colour legend
    _legend_entries = [
        ('#ff80ff', 'Mito channel (magenta)'),
        ('#80ff80', 'Protein channel (green)'),
        ('red',     'Lasso path'),
    ]
    for i, (color, label) in enumerate(_legend_entries):
        ax_z.text(
            0.01, 0.99 - i * 0.07, f'■  {label}',
            transform=ax_z.transAxes,
            color=color, fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6, ec='none'),
        )

    # initialize the mask so it exists in the enclosing scope
    inside_mask_2d = np.zeros((y_pixels, x_pixels), dtype=bool)

    # initialize line artist for displaying lasso points
    lasso_line, = ax_z.plot([], [], 'r-', linewidth=2, label='Lasso path')

    def on_select(verts):
        nonlocal inside_mask_2d
        if len(verts) < 3:
            return
        try:
            from skimage.draw import polygon
            verts_array = np.array(verts)
            # verts are (x, y) — polygon() wants (row, col) = (y, x)
            rr, cc = polygon(verts_array[:, 1], verts_array[:, 0],
                             shape=(y_pixels, x_pixels))
            inside_mask_2d = np.zeros((y_pixels, x_pixels), dtype=bool)
            inside_mask_2d[rr, cc] = True
            lasso_line.set_data(verts_array[:, 0], verts_array[:, 1])
        except Exception as e:
            print(f"Lasso selection error (ignored): {e}")

    lasso = LassoSelector(ax_z, on_select)
    lasso.set_active(True)

    def on_press(event):
        if event.inaxes == ax_z:
            lasso_line.set_data([], [])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_press)

    switch_channels = False
    def toggle_channels(event):
        nonlocal switch_channels
        switch_channels = not switch_channels
        if switch_channels:
            ax_z.images[0].set_data(scan_image)
            ax_z.images[1].set_data(mito_image)
            ax_z.set_title("Draw mitochondria region (Z view) - channels switched")
        else:
            ax_z.images[0].set_data(mito_image)
            ax_z.images[1].set_data(scan_image)
            ax_z.set_title("Draw mitochondria region (Z view)")
        fig.canvas.draw_idle()

    switch_button_ax = fig.add_axes([0.63, 0.01, 0.17, 0.05])
    switch_button = plt.Button(switch_button_ax, 'Switch Channels')
    switch_button.on_clicked(toggle_channels)

    done_button_ax = fig.add_axes([0.82, 0.01, 0.1, 0.05])
    done_button = plt.Button(done_button_ax, 'Done',
                             color='lightgreen', hovercolor='palegreen')
    # Use a timer to close the figure one event-loop tick after the button
    # click — closing synchronously inside a callback deadlocks some backends.
    done_button.on_clicked(
        lambda _: fig.canvas.manager.destroy() if hasattr(fig.canvas, 'manager') else plt.close(fig)
    )

    plt.show(block=True)

    inside_mask_2d = inside_mask_2d.astype(mito_image.dtype)
    return inside_mask_2d, switch_channels

@click.command()
@click.option('--input-directory', help='Input Directory', required=True)
@click.option('--manual-mask-directory', default='', help='Output directory for manually drawn masks (optional, default is same as input)', required=False)
@click.option('--mito-channel', default=0, help='Channel index for mitochondria (0-based)', required=False)
@click.option('--target-channel', default=1, help='Channel index for mask (0-based)', required=False)
@click.option('--scan-width', default=7, help='Width of scan lines in pixels', required=False)
@click.option('--sampling-radius', default=3, help='Radius for weighted average sampling in pixels', required=False)
@click.option('--outliers-csv', default='', help='Path to outliers CSV file from plot_peak_distance_analysis (optional, only process outlier images)', required=False)
def main(input_directory, manual_mask_directory, target_channel, mito_channel, scan_width=5, sampling_radius=3, outliers_csv=''):
    input_image_dir = input_directory
    
    #if manual_mask_directory is not provided, save the output in the same input directory
    if manual_mask_directory and not os.path.exists(manual_mask_directory):
        os.makedirs(manual_mask_directory)
    #if the manual_mask_directory does not exist, create it
    if not os.path.exists(input_image_dir):
        os.makedirs(input_image_dir)
    image_list = [f for f in os.listdir(input_image_dir) if f.endswith('.tif')]
    if not image_list:
        print(f"No TIFF files found in directory: {input_image_dir}")
        return
    
    # If outliers CSV is provided, filter to only process outlier images
    if outliers_csv and os.path.exists(outliers_csv):
        print(f"Loading outliers from: {outliers_csv}")
        outliers_df = pd.read_csv(outliers_csv)
        outlier_image_names = set(outliers_df['image_name'].unique())
        print(f"Found {len(outlier_image_names)} unique outlier images")
        
        # Filter image list to only include outlier images
        # Match by comparing the base filename (without extension) to outlier image names
        original_count = len(image_list)
        filtered_images = []
        for tiff_file in image_list:
            # Get base filename without extension
            base_name = os.path.splitext(tiff_file)[0]
            # Remove _mito_mask suffix if present
            if base_name.endswith('_mito_mask'):
                base_name = base_name[:-len('_mito_mask')]
            
            # Check if this matches any outlier image name (partial match allowed)
            for outlier_name in outlier_image_names:
                if outlier_name in base_name or base_name in outlier_name:
                    filtered_images.append(tiff_file)
                    break
        
        image_list = filtered_images
        print(f"Processing {len(image_list)} of {original_count} images (outliers only)")
        print(f"Outlier images: {outlier_image_names}")
        if not image_list:
            print("No matching outlier images found in the input directory")
            return
    
    # Initialize state for apply-all options
    overwrite_all = False
    skip_all = False
    
    for input_image in image_list:
        
        basename = os.path.basename(input_image)
        basename = basename[:basename.find(".tif")]

        output_image_path = os.path.join(manual_mask_directory, f"{basename}_mito_mask.tif") if manual_mask_directory else os.path.join(input_image_dir, f"{basename}_mito_mask.tif")
        #if the output file already exists, prompt the user to overwrite or skip
        response = 'y'  # default to overwrite
        if os.path.exists(output_image_path):
            # Check if user already selected apply-all options
            if skip_all:
                print(f"Skipping (skip all): {output_image_path}")
                continue
            elif overwrite_all:
                print(f"Overwriting (overwrite all): {output_image_path}")
                os.remove(output_image_path)
            else:
                while True:
                    response = input(f"Output file already exists: {output_image_path}\n(y/n/a=overwrite all/s=skip all): ").strip().lower()
                    if response == 'n':
                        print("Skipping...")
                        break
                    elif response == 'y':
                        os.remove(output_image_path)
                        break
                    elif response == 'a':
                        overwrite_all = True
                        os.remove(output_image_path)
                        print("Overwriting all mode enabled.")
                        break
                    elif response == 's':
                        skip_all = True
                        print("Skip all mode enabled.")
                        break
                    else:
                        print("Please enter y / n / a (overwrite all) / s (skip all).")
                if response in ('n', 's'):
                    continue
        mito_channel = 1
        target_channel = 0

        image = tf.imread(os.path.join(input_image_dir, input_image))
        print(f"Input file: {input_image}, shape: {image.shape}, dtype: {image.dtype}")
        
        mito_image = image[mito_channel, :, :]
        target_image = image[target_channel, :, :]
        mask_image, switch_channels = draw_mitochondria(mito_image, target_image)
        print(f'  Mask drawn for {basename} — saving...', flush=True)

        if switch_channels:
            mito_image, target_image = target_image, mito_image

        # Convert mask from boolean to match image dtype
        if np.issubdtype(image.dtype, np.floating):
            mask_image_converted = mask_image.astype(image.dtype)
        else:
            max_val = np.iinfo(image.dtype).max
            mask_image_converted = mask_image.astype(image.dtype)
            mask_image_converted *= max_val

        output_image = np.stack(
            [mask_image_converted, mito_image, target_image], axis=0
        )
        tf.imwrite(output_image_path, output_image,
                   photometric='minisblack', compression=None)
        print(f'  Saved → {output_image_path}', flush=True)


if __name__ == "__main__":
    main()
