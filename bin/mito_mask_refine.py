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
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
from scipy.ndimage import gaussian_filter
from .utils import weighted_average_scan, create_colormaps

def refine_mask_edges(mask_image, mito_image, scan_width=5, smooth_window=10):
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
    smooth_window : int
        Size of the  smoothing window for gradients
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
                scan_intensities.append(weighted_average_scan(mito_normalized, scan_y, scan_x, radius=3))
                scan_distances.append(dist)
        
        # Find the peak intensity along the scan 
        if len(scan_intensities) > 0:
            scan_intensities = np.array(scan_intensities)
            scan_distances = np.array(scan_distances)
            
            # Find the peak intensity along the scan and its corresponding distance
            if scan_intensities.sum() > 0:
                peak_dist = np.argmax(scan_intensities)
            else:
                peak_dist = 0
            
            # Calculate refined position
            refined_y = y + normal_y * scan_distances[peak_dist]
            refined_x = x + normal_x * scan_distances[peak_dist]
            
            # Round to nearest pixel
            refined_y = int(np.round(refined_y))
            refined_x = int(np.round(refined_x))
            
            # Check if the movement is more than 3 pixels
            movement = np.sqrt((refined_y - y)**2 + (refined_x - x)**2)
            if movement > 3:
                # Keep original position if movement is too large
                refined_y = y
                refined_x = x
            
            # Check bounds and store refined coordinates
            if 0 <= refined_y < mask_image.shape[0] and 0 <= refined_x < mask_image.shape[1]:
                refine_coords.append((refined_y, refined_x))
    #Fit a single spline to all refined coordinates and oversample by 10x
    refine_coords_oversampled = []
    if len(refine_coords) >= 3:
        refine_coords = np.array(refine_coords)
        from scipy.interpolate import splprep, splev
        try:
            # Fit a single spline to all coordinates s= number of points to smooth the spline, adjust as needed
            tck, u = splprep([refine_coords[:, 0], refine_coords[:, 1]], s=int(len(refine_coords)))
            # Generate points along the spline with 10x oversampling
            num_points = len(refine_coords) * 10
            u_new = np.linspace(0, 1, num_points)
            spline_coords = splev(u_new, tck)
            # Store oversampled coordinates as float tuples
            refine_coords_oversampled = list(zip(spline_coords[0], spline_coords[1]))
        except Exception as e:
            print(f"Warning: Spline fitting failed: {e}")
            # Fall back to original coordinates
            refine_coords_oversampled = [(float(c[0]), float(c[1])) for c in refine_coords]     

    #make a smoothed refined mask by filling in the refined coordinates
    if len(refine_coords_oversampled) > 0:
        # Create a new mask from the refined edge coordinates
        refined_mask = np.zeros_like(mask_image, dtype=bool)
        
        # Draw the refined boundary using oversampled float coordinates
        for coord in refine_coords_oversampled:
            y = int(np.round(coord[0]))
            x = int(np.round(coord[1]))
            if 0 <= y < refined_mask.shape[0] and 0 <= x < refined_mask.shape[1]:
                refined_mask[y, x] = True
        
        # Dilate the boundary slightly to create a filled region
        #refined_mask = binary_dilation(refined_mask, iterations=1)
        
        # Fill the holes in the mask
        refined_mask = ndimage.binary_fill_holes(refined_mask)
    
    return refined_mask.astype(mask_image.dtype)

@click.command()
@click.option('--input-directory', help='Input Image Directory', required=True)
@click.option('--mask-channel', help='Mask channel index (optional, default=0)', default=0, required=False)
@click.option('--mito-channel', help='Mitochondria channel index (optional, default=1)', default=1, required=False)
@click.option('--target-channel', help='Scan channel index (optional, default=2)', default=2, required=False)
@click.option('--refined-mask-directory', default='', help='Output directory for refined masks (optional, default is same as input)', required=False)
def main(input_directory, refined_mask_directory, mask_channel, mito_channel, target_channel):
    # Create output directory if it doesn't exist
    if refined_mask_directory and not os.path.exists(refined_mask_directory):
        os.makedirs(refined_mask_directory)
    
    
    image_list = [f for f in os.listdir(input_directory) if f.endswith('.tif')]
    if not image_list:
        print(f"No TIFF files found in directory: {input_directory}")
        return
    for input_image in image_list:
        input_image_path = os.path.join(input_directory, input_image)
        basename = os.path.basename(input_image)
        basename = basename[:basename.find(".tif")]

        output_image_path = os.path.join(refined_mask_directory, f"{basename}_mito_mask.tif") if refined_mask_directory else os.path.join(input_directory, f"{basename}_mito_mask.tif")
        
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
        output_png_dir = os.path.join(refined_mask_directory, "png_previews")
        if not os.path.exists(output_png_dir):
            os.makedirs(output_png_dir)
        output_png_path = os.path.join(refined_mask_directory, f"png_previews/{basename}_mito_mask_overlay.png")
        fig.savefig(output_png_path, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig)


if __name__ == "__main__":
    main()
