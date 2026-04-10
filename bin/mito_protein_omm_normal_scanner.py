#!/usr/bin/env python3

"""
This script reads mitos and scan structures from a file and separates mitos using connected components.
Processes microscopy images to analyze mitochondrial and scan structures.
"""

import sys
import os
import glob

import click
import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.morphology import skeletonize
import sknw
import networkx as nx
from matplotlib.widgets import LassoSelector, Slider
from matplotlib.path import Path
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import pandas as pd
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from scipy.signal import find_peaks, peak_prominences
from skimage import exposure
import scipy.interpolate as interpolate
import pickle
import json
from .utils import weighted_average_scan, create_colormaps



def interactive_mask_erosion(mito_image, mask_image):
    """Interactive mask erosion selection using matplotlib slider.
    
    Args:
        mito_image: Mitochondria image to display as background
        mask_image: Binary mask image to erode
        
    Returns:
        erosion_value: Final erosion value selected (1-20)
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Adjust subplot to make space for the slider

    initial_erosion = 1
    erosion_value = initial_erosion

    # Normalize mito_image for display
    #mito_display = mito_image / np.max(mito_image[:]) if np.max(mito_image[:]) > 0 else mito_image
    
    # Plot the mitochondria image
    ax.imshow(mito_image, cmap='gray')
    
    # Create initial eroded mask
    eroded_mask = binary_erosion(mask_image, structure=disk(initial_erosion)).astype(np.uint8)
    
    # Overlay the mask with transparency
    mask_display = ax.imshow(eroded_mask, cmap='Reds', alpha=0.5)
    
    ax.set_title(f'Erosion: {initial_erosion}')
    
    # Create axes for the slider
    ax_slider = plt.axes([0.2, 0.08, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    # Create the Slider widget (integer values from 1 to 20)
    slider = Slider(ax_slider, 'Erosion', 1, 20, valinit=initial_erosion, valstep=1)

    def update_plot(val):
        nonlocal erosion_value
        erosion_value = int(slider.val)
        
        # Erode the mask with the new value
        eroded_mask = binary_erosion(mask_image, structure=disk(erosion_value)).astype(np.uint8)
        
        # Update the mask overlay
        mask_display.set_data(eroded_mask)
        ax.set_title(f'Erosion: {erosion_value}')
        fig.canvas.draw_idle()

    # Register the update function with the slider's on_changed event
    slider.on_changed(update_plot)
    plt.show()
    
    return erosion_value


@click.command()
@click.option('--input-directory', type=click.Path(exists=True))
@click.option('--output-directory', default='test/', help='Output directory')
@click.option('--mito-channel', default=1, help='Mitochondria channel index (0-based)')
@click.option('--scan-channel', default=0, help='Scan channel index (0-based)')
@click.option('--mask-channel', default=2, help='Mask channel index (0-based)')
@click.option('--scan-width', default=5, help='Width of scan lines in pixels')
@click.option('--sampling-radius', default=3, help='Radius for weighted average sampling in pixels')
@click.option('--mito-thickness-threshold', default=5, help='Ignore areas where mitochondria are thinner than this threshold (in pixels)')
def main(input_directory, output_directory, mito_channel, scan_channel, mask_channel, scan_width, sampling_radius, mito_thickness_threshold):
    '''
    Main function to process images and analyze mitochondrial localization.
    Example usage: python mito_protein_omm_normal_scanner.py --input-directory /path/to/images/ --output-directory /path/to/output/ --mito-channel 1 --scan-channel 0 --mask-channel 2 --scan-width 7 --sampling-radius 3 --mito-thickness-threshold 1
    '''
    input_image_list = glob.glob(os.path.join(input_directory, '*.tif'))
    if not input_image_list:
        print(f"No TIFF files found in directory: {input_directory}")
        sys.exit(1)
    
    #create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_dir = output_directory

    scan_width = scan_width
    for input_image in input_image_list:
        print(f"Processing file: {input_image}")
        #mito_cmap, scan_cmap = create_colormaps()
        basename = os.path.basename(input_image)
        
        mito_ch = mito_channel
        target_channel = scan_channel
        mask_ch = mask_channel

        #read the image and extract channels (assuming the image is in (channels, height, width) format)
        with tf.TiffFile(input_image) as tif:
            # Read all pages into a 3D array
            image = np.array([page.asarray() for page in tif.pages])
        
        print(f"Loaded image with shape: {image.shape} and dtype: {image.dtype}")
        
        # Ensure image is 3D (pages, height, width)
        if image.ndim == 2:
            raise ValueError(f"Image is 2D with shape {image.shape}. Expected 3D (pages, height, width). Check if this is a multi-page TIFF.")
        
        mito_image = image[mito_ch, :, :]
        target_image = image[target_channel, :, :]
        mask_image = image[mask_ch, :, :]

        distance = mask_image.copy()
        distance = distance_transform_edt(distance.astype(np.uint8)).astype(np.float32)

        # get skeleton (medial axis)
        binary = mask_image.copy()
        binary = binary.astype(np.float32)/255
        skeleton = skeletonize(binary).astype(np.float32)

        # apply skeleton to select center line of distance 
        thickness = distance * skeleton

        #remove the parts that are thinner than a certain threshold from the original mask
        thickness_threshold = mito_thickness_threshold
        thickness_mask = (thickness < thickness_threshold) & (thickness > 0.0)
        #dilate the thickness mask to cover slightly larger areas
        thickness_mask = binary_dilation(thickness_mask, structure=disk(thickness_threshold+2), iterations=1).astype(np.uint8)
        #invert the thickness mask to get the areas that are thicker than the threshold
        thickness_mask = 1 - thickness_mask

        #plt.imshow(thickness_mask, cmap='gray')
        #plt.title("Thickness Mask")
        #plt.show()



        # get the outer edges of the eroded mask
        edge_mask = mask_image - binary_erosion(mask_image, structure=disk(2), iterations=1).astype(mask_image.dtype)

        #Now dilate the original mask to make sure we cover all mito areas
        dilated_mask = binary_dilation(mask_image, structure=disk(7), iterations=1).astype(mask_image.dtype)
        mito_image = mito_image * dilated_mask
        target_image = target_image * dilated_mask
        #plt.imshow(edge_mask, cmap='gray')
        #plt.title("Eroded Mask Edges")
        #plt.show()

        # Display initial overlay
        fig, ax = plt.subplots()


        #make sure the images are normalized to [-1, 1]
        print("normalizing images for display")
        mito_image = (mito_image - np.min(mito_image)) / (np.max(mito_image) - np.min(mito_image))
        target_image = (target_image - np.min(target_image)) / (np.max(target_image) - np.min(target_image))

        # Apply threshold and preprocessing
        mito_binary = edge_mask > 0
        # Remove edges from the binary mask
        mito_binary[:5, :] = 0
        mito_binary[-5:, :] = 0
        mito_binary[:, :5] = 0
        mito_binary[:, -5:] = 0

        # Skeletonize the binary mask
        mito_skeleton = skeletonize(mito_binary, method='lee')  # Use 'lee' method for better results

        # Build network from skeleton
        mito_nx = sknw.build_sknw(mito_skeleton, multi=True)

        # Display skeleton and network
        fig = plt.figure(figsize=(10, 10))
        mito_cmap, scan_cmap = create_colormaps()
        plt.imshow(mito_image, cmap=mito_cmap, alpha=1)
        plt.imshow(target_image, cmap=scan_cmap, alpha=0.5)
        plt.imshow(mito_skeleton, cmap=mito_cmap, alpha=1)

        nodes = mito_nx.nodes()
        pos = np.array([[nodes[i]['o'][1], nodes[i]['o'][0]] for i in nodes])
        node_labels = {node: node for node in mito_nx.nodes()}
        nx.draw(mito_nx, pos, alpha=0.5, width=0, labels=node_labels, node_size=300, 
                node_color='pink', font_color="whitesmoke")
        fig.set_facecolor('black')
        plt.title(f"Skeleton and Network for {basename}")
        plt.axis('off')
        plt.savefig(f"{output_dir}/{basename}_skeleton_network.png")
        plt.close()

        # Process each mitochondrial path
        mito_i = 0
        for u, v in mito_nx.edges():
            for i in range(len(mito_nx[u][v])):
                mito_i = mito_i + 1
                path = mito_nx[u][v][i]['pts']
                path_mito = path
                
                if len(path) < 30:
                    continue

                #remove the points that are not in the thickness mask
                mask_indices = []
                for j in range(len(path)):
                    if thickness_mask[int(path[j, 0]), int(path[j, 1])] == 0:
                        mask_indices.append(j)
                path = np.delete(path, mask_indices, axis=0)
                path_mito = path
                
                if len(path) < 30:
                    continue

                # Fit a spline to the path 
                path_x = path[:, 1]
                path_y = path[:, 0]
                tck, uu = interpolate.splprep([path_x, path_y], s=50)
                x_i, y_i = interpolate.splev(np.linspace(0, 1, 200), tck)
                dx, dy = interpolate.splev(np.linspace(0, 1, 200), tck, der=1)
                


                path = np.column_stack((y_i, x_i))
                print(f"Path length: {len(path_mito)}")
                
                detailed_data = []
                normal_x_plot = []
                normal_y_plot = []

                # Process each point in the path
                for p_ind in range(len(path_mito)):
                    #check if this point is in the thickness mask
                    


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

                    # Remove duplicates and out-of-bounds points
                    normal_x = np.array(normal_x)
                    normal_y = np.array(normal_y)
                    points = np.stack((normal_x, normal_y), axis=1)
                    unique_points = np.unique(points, axis=0)
                    normal_x = unique_points[:, 0]
                    normal_y = unique_points[:, 1]

                    # Collect intensities along the normal (raw arrays, not averaged)
                    mito_intensity_array = []
                    scan_intensity_array = []
                    mask_intensity_array = []
                    valid_normal_x = []
                    valid_normal_y = []
                    
                    for j in range(len(normal_x)):
                        if (normal_x[j] < 0 or normal_x[j] >= target_image.shape[0] or 
                            normal_y[j] < 0 or normal_y[j] >= target_image.shape[1]):
                            continue
                        
                        # Use weighted average scan instead of single pixel
                        mito_intensity_array.append(float(weighted_average_scan(mito_image, normal_x[j], normal_y[j], sampling_radius)))
                        scan_intensity_array.append(float(weighted_average_scan(target_image, normal_x[j], normal_y[j], sampling_radius)))
                        mask_intensity_array.append(float(weighted_average_scan(mask_image, normal_x[j], normal_y[j], 1)))
                        valid_normal_x.append(int(normal_x[j]))
                        valid_normal_y.append(int(normal_y[j]))
                    
                    # Store detailed data for this skeleton point
                    detailed_data.append({
                        'skeleton_point': [float(point[0]), float(point[1])],
                        'path_distance': float(path_dist),
                        'normal_vector': [float(normal[0]), float(normal[1])],
                        'normal_line_points': list(zip(valid_normal_x, valid_normal_y)),
                        'mito_intensities': mito_intensity_array,
                        'scan_intensities': scan_intensity_array,
                        'mask_intensities': mask_intensity_array
                    })

                # Create visualization and save results
                # Extract averaged data for plotting
                mito_intensities = np.array([np.mean(d['mito_intensities']) if d['mito_intensities'] else 0 for d in detailed_data])
                scan_intensities = np.array([np.mean(d['scan_intensities']) if d['scan_intensities'] else 0 for d in detailed_data])
                path_length = np.array([d['path_distance'] for d in detailed_data])
                
                fig, ax = plt.subplots(1, 3, figsize=(25, 5), width_ratios=[1, 1, 3])
                
                
                ax[0].imshow(mito_image, cmap='gray', alpha=1)
                ax[0].scatter(path_mito[:, 1], path_mito[:, 0], c=cm.winter(np.array(path_length)/np.max(path_length)))
                ax[0].plot(path_x, path_y, color='blue', linewidth=1)
                ax[0].scatter(normal_y_plot, normal_x_plot, color='red', s=1)
                ax[0].set_title(f"Mito {mito_i} - Path length: {len(path_mito)}")
                ax[0].set_facecolor('black')
                ax[0].set_xlim(np.min(path_x)-20, np.max(path_x)+20)
                ax[0].set_ylim(np.min(path_y)-20, np.max(path_y)+20)

                # Plot scan channel
                ax[1].imshow(target_image, cmap='gray', alpha=1)
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

                # Save detailed data structure to pickle file (preserves all data)
                with open(f"{output_dir}/{basename}_mito_{mito_i}_detailed.pkl", 'wb') as f:
                    pickle.dump({
                        'mito_id': mito_i,
                        'image_name': basename,
                        'detailed_data': detailed_data
                    }, f)
                
                # Also save a more readable JSON version 
                json_data = {
                    'mito_id': mito_i,
                    'image_name': basename,
                    'detailed_data': detailed_data
                }
                with open(f"{output_dir}/{basename}_mito_{mito_i}_detailed.json", 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # Save a summary CSV with averaged values for quick inspection
                summary_data = {
                    'Distance': path_length, 
                    'Mito_Intensity_Mean': mito_intensities, 
                    'Scan_Intensity_Mean': scan_intensities
                }
                df = pd.DataFrame(summary_data)
                scan_basename = basename
                df.to_csv(f"{output_dir}/{scan_basename}_mito_{mito_i}_summary.csv", index=False)
                

if __name__ == "__main__":
    main()