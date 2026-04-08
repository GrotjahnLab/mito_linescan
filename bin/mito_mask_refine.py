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
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.ndimage import gaussian_filter
from mito_protein_omm_localization import weighted_average_scan
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


def refine_mask_edges(mask_image, mito_image, scan_width=3):
    """
    Refine mask edges by scanning perpendicular to edges to find mito signal centroid.
    
    Parameters:
    -----------
    mask_image : ndarray
        Binary mask (inside=True, outside=False)
    mito_image : ndarray
        Mitochondria intensity image
    scan_width : int
        Number of pixels to scan in each direction perpendicular to edge
    
    Returns:
    --------
    refined_mask : ndarray
        Refined binary mask with edges centered on mito signal
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt, convolve
    import numpy as np
    
    refined_mask = mask_image.astype(bool).copy()
    mask_edges = np.logical_xor(ndimage.binary_erosion(mask_image), mask_image)
    refine_coords = []
    
    # Get edge pixel coordinates
    edge_coords = np.where(mask_edges)
    
    if len(edge_coords[0]) == 0:
        return refined_mask.astype(mask_image.dtype)
    
    # Normalize mito image for scanning
    mito_normalized = mito_image.astype(float)
    mito_normalized = (mito_normalized - mito_normalized.min()) / (mito_normalized.max() - mito_normalized.min() + 1e-8)
    

    # Compute gradient to get normal direction at each edge point
    # Apply Gaussian filter to mask before computing gradients for smoother results
    mask_for_gradient = gaussian_filter(mask_image.astype(float), sigma=1.5)
    gy, gx = np.gradient(mask_for_gradient)
    
    # Smooth the gradient fields for cleaner normal directions
    gy = gaussian_filter(gy, sigma=1.0)
    gx = gaussian_filter(gx, sigma=1.0)
    
    # Normalize gradients
    grad_mag = np.sqrt(gx**2 + gy**2) + 1e-8
    nx = gx / grad_mag  # normal x
    ny = gy / grad_mag  # normal y
    

    
    # For each edge pixel
    for y, x in zip(edge_coords[0], edge_coords[1]):
        if not (0 <= y < mask_image.shape[0] and 0 <= x < mask_image.shape[1]):
            continue
        
        # Get normal direction at this edge pixel
        normal_x = nx[y, x]
        normal_y = ny[y, x]
        
        # Scan along the normal direction (both inward and outward)
        scan_intensities = []
        scan_distances = []
        
        for dist in range(-scan_width, scan_width + 1):
            # Position along normal
            scan_y = int(y + normal_y * dist)
            scan_x = int(x + normal_x * dist)
            
            # Check bounds
            if 0 <= scan_y < mito_image.shape[0] and 0 <= scan_x < mito_image.shape[1]:
                intensity = mito_normalized[scan_y, scan_x]
                scan_intensities.append(weighted_average_scan(mito_normalized, scan_y, scan_x, 3))
                scan_distances.append(dist)
        
        # Find centroid of intensity along the normal
        if len(scan_intensities) > 0:
            scan_intensities = np.array(scan_intensities)
            scan_distances = np.array(scan_distances)
            
            # Weighted average to find centroid
            if scan_intensities.sum() > 0:
                centroid_dist = np.average(scan_distances, weights=scan_intensities)
            else:
                centroid_dist = 0
            
            # Calculate refined position
            refined_y = y + normal_y * centroid_dist
            refined_x = x + normal_x * centroid_dist
            
            # Round to nearest pixel
            refined_y = int(np.round(refined_y))
            refined_x = int(np.round(refined_x))
            
            # Check bounds and store refined coordinates
            if 0 <= refined_y < mask_image.shape[0] and 0 <= refined_x < mask_image.shape[1]:
                refine_coords.append((refined_y, refined_x))
            
    #make a smoothed refined mask by filling in the refined coordinates and then applying a closing operation to fill in gaps
    for y, x in refine_coords:
        refined_mask[y, x] = True
    # Apply binary closing to fill in gaps
    refined_mask = binary_dilation(refined_mask, iterations=2)
    #fill the holes in the mask
    refined_mask = ndimage.binary_fill_holes(refined_mask)
    #erode the mask slightly to get back to original size after dilation
    refined_mask = ndimage.binary_erosion(refined_mask, iterations=2)
    return refined_mask.astype(mask_image.dtype)

@click.command()
@click.option('--i', help='Input Image Directory', required=True)
@click.option('--mask_channel', help='Mask channel index (optional, default=0)', default=0, required=False)
@click.option('--mito_channel', help='Mitochondria channel index (optional, default=1)', default=1, required=False)
@click.option('--target_channel', help='Scan channel index (optional, default=2)', default=2, required=False)
@click.option('--o', default='', help='Output directory (optional, default is same as input)', required=False)
def main(i, o, mask_channel, mito_channel, target_channel):
    scan_width = 3
    # Create output directory if it doesn't exist
    if o and not os.path.exists(o):
        os.makedirs(o)
    
    
    image_list = [f for f in os.listdir(i) if f.endswith('.tif')]
    if not image_list:
        print(f"No TIFF files found in directory: {i}")
        return
    for input_image in image_list:
        input_image_path = os.path.join(i, input_image)
        basename = os.path.basename(input_image)
        basename = basename[:basename.find(".tif")]

        output_image_path = os.path.join(o, f"{basename}_mito_mask.tif") if o else os.path.join(i, f"{basename}_mito_mask.tif")
        
            #read the image and extract channels (assuming the image is in (channels, height, width) format)
        with tf.TiffFile(input_image_path) as tif:
            # Read all pages into a 3D array
            image = np.array([page.asarray() for page in tif.pages])

        
        mito_image = image[mito_channel, :, :]
        target_image = image[target_channel, :, :]
        mask_image = image[mask_channel, :, :]
        #first smooth the mask so it has less sharp edges for better lasso selection

        mask_image_smoothed = gaussian_filter(mask_image, sigma=3)
        #binarize the mask again after smoothing
        mask_image_smoothed = (mask_image_smoothed > 0.5).astype(mask_image.dtype)
        
        # Refine mask edges by scanning perpendicular to find mito signal centroid
        refined_mask = refine_mask_edges(mask_image_smoothed, mito_image, scan_width=5)

        # Save first page (refined mask)
        tf.imwrite(output_image_path, refined_mask, photometric='minisblack')
        # Append remaining pages
        tf.imwrite(output_image_path, mito_image, photometric='minisblack', append=True)
        tf.imwrite(output_image_path, target_image, photometric='minisblack', append=True)

        refined_edge = np.logical_xor(ndimage.binary_erosion(refined_mask), refined_mask)
        #save a PNG overlapping the new refined mask on the mito image for visualization
        mito_cmap, scans_cmap = create_colormaps()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        # Set both figure and axis background to black
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Display images
        ax.imshow(mito_image, cmap=mito_cmap)
        ax.imshow(target_image, cmap=scans_cmap, alpha=0.5)
        ax.imshow(refined_edge, cmap=mito_cmap)
        ax.axis('off')
        
        # Save the figure properly
        output_png_path = os.path.join(o, f"{basename}_mito_mask_overlay.png")
        fig.savefig(output_png_path, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig)


if __name__ == "__main__":
    main()
