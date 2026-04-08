#!/usr/bin/env python3

"""
This script reads mitos and scan structures from a file and separates mitos using connected components.
Processes microscopy images to analyze mitochondrial and scan structures.
"""

import os
import click
import tiffile as tf
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import LassoSelector, Slider
from matplotlib.path import Path
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import pandas as pd

def create_colormaps():
    """Create custom colormaps for mitochondria and scan visualization."""
    # Mitochondria colormap
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.sqrt(np.linspace(0/256, 1, N))
    vals[:, 1] = np.sqrt(np.linspace(0/256, 64/256, N))
    vals[:, 2] = np.sqrt(np.linspace(0/256, 1, N))
    vals[:, 3] = np.sqrt(np.linspace(0/256, 256/256, N))
    mito_cmap = ListedColormap(vals)

    # Scan colormap
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.sqrt(np.linspace(0/256, 64/256, N))
    vals[:, 1] = np.sqrt(np.linspace(64/256, 1, N))
    vals[:, 2] = np.sqrt(np.linspace(0/256, 64/256, N))
    vals[:, 3] = np.sqrt(np.linspace(0/256, 256/256, N))
    scan_cmap = ListedColormap(vals)
    
    return mito_cmap, scan_cmap


def draw_mitochondria(mito_image, scan_image):
        # Prepare indices assuming mito_image shape is (Z, Y, X)

    y_pixels = mito_image.shape[0]
    x_pixels = mito_image.shape[1]


    # Compute figure size so pixels are displayed with correct proportions
    # Keep a fixed display height (in inches) and scale widths by image aspect ratios
    # Z view (axis 0) slices have shape (Y, X)
    # Y view (axis 1) slices have shape (Z, X)
    display_height = 6.0
    fig = plt.figure()

    ax_z = fig.add_subplot(111)
    mito_cmap, scan_cmap = create_colormaps()
    ax_z.set_facecolor("black")
    ax_z.imshow(mito_image, cmap=mito_cmap, alpha = 0.7)
    ax_z.imshow(scan_image, cmap=scan_cmap, alpha = 0.3)
    ax_z.set_title("Draw mitochondria region (Z view)")

    # initialize the mask so it exists in the enclosing scope
    inside_mask_2d = np.zeros((y_pixels, x_pixels), dtype=bool)

    # lasso selector for user draw mitochondria region
    def on_select(verts):
        nonlocal inside_mask_2d
        lasso_path = Path(verts)
        X, Y = np.meshgrid(np.arange(x_pixels), np.arange(y_pixels), indexing='xy')
        coords = np.vstack((X.ravel(), Y.ravel())).T  # shape (num_points, 2)

        # Determine which points are inside the lasso path
        inside = lasso_path.contains_points(coords)
        inside_mask_2d = inside.reshape((y_pixels, x_pixels))  # shape (Y, X)
        fig.canvas.draw_idle()

    lasso = LassoSelector(ax_z, on_select)
    lasso.set_active(True)
    #add a botton to switch mito and scan channels if necessary
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
    switch_button_ax = fig.add_axes([0.8, 0.01, 0.15, 0.05])  # x, y, width, height
    switch_button = plt.Button(switch_button_ax, 'Switch Channels')
    switch_button.on_clicked(toggle_channels)
    plt.show()
    #convert the boolean mask to the same dtype as the input image (e.g., uint8 or float)
    inside_mask_2d = inside_mask_2d.astype(mito_image.dtype)
    return inside_mask_2d, switch_channels


def create_colormaps():
    """Create custom colormaps for mitochondria and scan visualization."""
    # Mitochondria colormap
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.sqrt(np.linspace(0/256, 1, N))
    vals[:, 1] = np.sqrt(np.linspace(0/256, 64/256, N))
    vals[:, 2] = np.sqrt(np.linspace(0/256, 1, N))
    vals[:, 3] = np.sqrt(np.linspace(0/256, 256/256, N))
    mito_cmap = ListedColormap(vals)

    # Scan colormap
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.sqrt(np.linspace(0/256, 64/256, N))
    vals[:, 1] = np.sqrt(np.linspace(64/256, 1, N))
    vals[:, 2] = np.sqrt(np.linspace(0/256, 64/256, N))
    vals[:, 3] = np.sqrt(np.linspace(0/256, 256/256, N))
    scan_cmap = ListedColormap(vals)
    
    return mito_cmap, scan_cmap

@click.command()
@click.option('--i', help='Input Directory', required=True)
@click.option('--o', default='', help='Output directory (optional, default is same as input)', required=False)
def main(i, o):
    input_image = i
    scan_width = 7
    
    image_list = [f for f in os.listdir(input_image) if f.endswith('.tif')]
    if not image_list:
        print(f"No TIFF files found in directory: {input_image}")
        return
    for input_image in image_list:
        
        basename = os.path.basename(input_image)
        basename = basename[:basename.find(".tif")]

        output_image_path = os.path.join(o, f"{basename}_mito_mask.tif") if o else os.path.join(i, f"{basename}_mito_mask.tif")
        mito_channel = 1
        target_channel = 0

        image = tf.imread(os.path.join(i, input_image))
        print(f"Input file: {input_image}, shape: {image.shape}, dtype: {image.dtype}")
        
        mito_image = image[mito_channel, :, :]
        target_image = image[target_channel, :, :]
        mask_image, switch_channels = draw_mitochondria(mito_image, target_image)

        if switch_channels:
            mito_image, target_image = target_image, mito_image

        # Convert mask from boolean to match image dtype
        if np.issubdtype(image.dtype, np.floating):
            mask_image_converted = mask_image.astype(image.dtype)
        else:
            max_val = np.iinfo(image.dtype).max
            mask_image_converted = (mask_image.astype(image.dtype) * max_val)
        
        # Create 3D output image using pre-allocated array (C-contiguous)
        output_image = np.empty((3, image.shape[1], image.shape[2]), dtype=image.dtype)
        output_image[target_channel, :, :] = target_image
        output_image[mito_channel, :, :] = mito_image
        output_image[2, :, :] = mask_image_converted
        
        print(f"Output shape: target={target_image.shape}, mito={mito_image.shape}, mask={mask_image_converted.shape}")
        print(f"Mask min/max: {mask_image_converted.min()}/{mask_image_converted.max()}")
        
        # Save as multi-page TIFF with 3 separate grayscale pages
        # Save first page
        tf.imwrite(output_image_path, mask_image_converted, photometric='minisblack')
        # Append remaining pages
        tf.imwrite(output_image_path, mito_image, photometric='minisblack', append=True)
        tf.imwrite(output_image_path, target_image, photometric='minisblack', append=True)


if __name__ == "__main__":
    main()
