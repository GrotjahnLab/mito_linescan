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
        edge_mask = mask_image - binary_erosion(mask_image, structure=disk(1), iterations=1).astype(mask_image.dtype)


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
                #mask_indices = []
                #for j in range(len(path)):
                #    x_idx = int(path[j, 0])
                #    y_idx = int(path[j, 1])
                    # Check bounds before accessing thickness_mask
                #    if x_idx < 0 or x_idx >= thickness_mask.shape[1] or y_idx < 0 or y_idx >= thickness_mask.shape[0]:
                #        mask_indices.append(j)
                #    elif thickness_mask[y_idx, x_idx] == 0:
                #        mask_indices.append(j)
                #path = np.delete(path, mask_indices, axis=0)
                path_mito = path
                
                if len(path) < 30:
                    continue

                # Fit a spline to the path 
                path_y = path[:, 1]
                path_x = path[:, 0]
                tck, uu = interpolate.splprep([path_y, path_x], s=1)
                y_i, x_i = interpolate.splev(np.linspace(0, 1, 200), tck)


                path_interp = np.column_stack((y_i, x_i))
                print(f"Path length: {len(path_mito)}")
                
                detailed_data = []
                normal_x_plot = []
                normal_y_plot = []

                # Process each point in the path
                for p_ind in range(len(path_mito)):
                    
                    point = np.array(path_mito[p_ind])
                    
                    # Get local tangent by fitting a local spline around this point
                    # Use points within a window around the current point
                    window_size = min(12, len(path_mito) // 2)
                    start_idx = max(0, p_ind - window_size)
                    end_idx = min(len(path_mito), p_ind + window_size + 1)
                    
                    local_path = path_mito[start_idx:end_idx]
                    
                    if len(local_path) >= 3:
                        try:
                            local_y = local_path[:, 1]
                            local_x = local_path[:, 0]
                            local_tck, _ = interpolate.splprep([local_y, local_x], s=2, k=min(3, len(local_path)-1))
                            
                            # Evaluate derivative at the center of window (normalized parameter)
                            center_idx = p_ind - start_idx
                            local_param = center_idx / max(1, end_idx - start_idx - 1)
                            dy_local, dx_local = interpolate.splev(local_param, local_tck, der=1)
                            
                            # Compute normal vector by rotating tangent 90 degrees
                            normal = np.array([-dy_local, dx_local])
                            norm_mag = np.linalg.norm(normal)
                            if norm_mag > 0:
                                normal = normal / norm_mag
                            else:
                                normal = np.array([0, 1])
                        except Exception as e:
                            print(f"Error fitting local spline at point {p_ind}: {e}")
                            # Fallback: use finite difference
                            if p_ind > 0 and p_ind < len(path_mito) - 1:
                                tangent = path_mito[p_ind+1] - path_mito[p_ind-1]
                                normal = np.array([-tangent[1], tangent[0]])
                                norm_mag = np.linalg.norm(normal)
                                if norm_mag > 0:
                                    normal = normal / norm_mag
                            else:
                                normal = np.array([0, 1])
                    else:
                        # Fallback: use finite difference
                        if p_ind > 0 and p_ind < len(path_mito) - 1:
                            tangent = path_mito[p_ind+1] - path_mito[p_ind-1]
                            normal = np.array([-tangent[1], tangent[0]])
                            norm_mag = np.linalg.norm(normal)
                            if norm_mag > 0:
                                normal = normal / norm_mag
                        else:
                            normal = np.array([0, 1])
                    
                    # Determine scan direction based on mask values on either side of the path
                    test_point_1 = (int(point[1] + 3*normal[1]), int(point[0] + 10*normal[0]))
                    test_point_2 = (int(point[1] - 3*normal[1]), int(point[0] - 10*normal[0]))
                    if weighted_average_scan(mask_image, test_point_1[1], test_point_1[0], 1) == 1:
                        normal = -normal
                        print(f"Flipping normal at point {p_ind} because test point 1 is inside mask")
                    elif weighted_average_scan(mask_image, test_point_2[1], test_point_2[0], 1) == 1:
                        normal = normal
                        print(f"Keeping normal at point {p_ind} because test point 2 is inside mask")
                    
                    normal_y = []
                    normal_x = []
                    path_dist = 0
                    
                    # Calculate total path length up to this point
                    for ii in range(p_ind):
                        path_dist += np.linalg.norm(path_mito[ii] - path_mito[ii-1]) if ii > 0 else 0
                
                    # Find points along the normal vector
                    for dt in range(-scan_width, scan_width):
                        #get y,x with distance dt along the normal direction
                        y = int(point[1] + dt*normal[1])
                        x = int(point[0] + dt*normal[0])
                        #print(f"Normal point at distance {dt}: ({y}, {x}, {normal[0]}, {normal[1]})")
                        normal_y.append(y)
                        normal_x.append(x)
                        if p_ind % 5 == 0:
                            normal_y_plot.append(y)
                            normal_x_plot.append(x)

                    # Remove duplicates and out-of-bounds points
                    normal_y = np.array(normal_y)
                    normal_x = np.array(normal_x)
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
                    normal_distance_array = []
                    
                    for j in range(len(normal_x)):
                        if (normal_x[j] < 0 or normal_x[j] >= target_image.shape[0] or 
                            normal_y[j] < 0 or normal_y[j] >= target_image.shape[1]):
                            continue
                        
                        # Calculate distance along the normal direction (should be -scan_width to scan_width range)
                        dist_from_center = np.linalg.norm([normal_y[j] - point[1], normal_x[j] - point[0]])
                        #print([normal_y[j], normal_x[j]],[point[1], point[0]], dist_from_center)
                        # Use weighted average scan instead of single pixel
                        mito_intensity_array.append(float(weighted_average_scan(mito_image, normal_x[j], normal_y[j], sampling_radius)))
                        scan_intensity_array.append(float(weighted_average_scan(target_image, normal_x[j], normal_y[j], sampling_radius)))
                        mask_intensity_array.append(float(weighted_average_scan(mask_image, normal_x[j], normal_y[j], sampling_radius)))
                        valid_normal_x.append(int(normal_x[j]))
                        valid_normal_y.append(int(normal_y[j]))
                        # Store the dot product with normal to get signed distance along normal, if the distance is zero, sign is also zero
                        if np.linalg.norm([normal_x[j] - point[0], normal_y[j] - point[1]]) == 0:
                            sign_dist = 0.0
                        else:
                            sign_dist = np.dot([normal_x[j] - point[0], normal_y[j] - point[1]], normal) / np.linalg.norm([normal_x[j] - point[0], normal_y[j] - point[1]])

                        
                        normal_distance_array.append(float(sign_dist * np.linalg.norm([normal_x[j] - point[0], normal_y[j] - point[1]])))

                        #sort the normal points and intensities by the distance along the normal by sorting the normal_distance_array and applying the same sorting to the intensity arrays
                    sorted_indices = np.argsort(normal_distance_array)
                    normal_distance_array = np.array(normal_distance_array)[sorted_indices]
                    mito_intensity_array = np.array(mito_intensity_array)[sorted_indices]
                    scan_intensity_array = np.array(scan_intensity_array)[sorted_indices]
                    mask_intensity_array = np.array(mask_intensity_array)[sorted_indices]
                    valid_normal_x = np.array(valid_normal_x)[sorted_indices]
                    valid_normal_y = np.array(valid_normal_y)[sorted_indices]

                    #show normal points on the image for debugging
                    if False:
                        fig, ax = plt.subplots(1,2, figsize=(12, 6))
                        ax[0].imshow(mito_image, cmap=mito_cmap, alpha=1)
                        ax[0].imshow(target_image, cmap=scan_cmap, alpha=0.5)
                        ax[0].imshow(mask_image, cmap='gray', alpha=0.2)
                        ax[0].plot(path_y, path_x, color='blue', linewidth=1)
                        
                        ax[0].scatter(path_mito[:, 1], path_mito[:, 0])
                        
                        ax[0].scatter(valid_normal_y, valid_normal_x, c=normal_distance_array, cmap='coolwarm', s=20)
                        ax[0].set_title(f"Normal points for skeleton point {p_ind} (distance along normal shown in color)")
                        ax[1].plot(normal_distance_array, mito_intensity_array, color='blue', label='Mito Intensity')
                        ax[1].plot(normal_distance_array, scan_intensity_array, color='green', label='Scan Intensity')
                        ax[1].plot(normal_distance_array, mask_intensity_array, color='orange', label='Mask Intensity')
                        ax[1].set_title(f"Intensity profiles along normal for skeleton point {p_ind}")
                        ax[1].set_xlabel("Distance along normal (pixels)")
                        ax[1].set_ylabel("Intensity")
                        plt.show()

                    print(np.array(normal_distance_array))
                    
                    # Store detailed data for this skeleton point
                    detailed_data.append({
                        'skeleton_point': [float(point[0]), float(point[1])],
                        'path_distance': float(path_dist),
                        'normal_vector': [float(normal[0]), float(normal[1])],
                        'normal_line_points': [[int(x), int(y)] for x, y in zip(valid_normal_x, valid_normal_y)],
                        'mito_intensities': [float(x) for x in mito_intensity_array],
                        'scan_intensities': [float(x) for x in scan_intensity_array],
                        'mask_intensities': [float(x) for x in mask_intensity_array],
                        'normal_distances': [float(x) for x in normal_distance_array]
                    })

                # Create visualization and save results
                # Extract averaged data for plotting
                mito_intensities = np.array([np.mean(d['mito_intensities']) if len(d['mito_intensities']) > 0 else 0 for d in detailed_data])
                scan_intensities = np.array([np.mean(d['scan_intensities']) if len(d['scan_intensities']) > 0 else 0 for d in detailed_data])
                path_length = np.array([d['path_distance'] for d in detailed_data])
                
                fig, ax = plt.subplots(1, 3, figsize=(25, 5), width_ratios=[1, 1, 1])
                
                
                ax[0].imshow(mito_image, cmap='gray', alpha=1)
                ax[0].scatter(path_mito[:, 1], path_mito[:, 0], c=cm.winter(np.array(path_length)/np.max(path_length)))
                ax[0].plot(path_y, path_x, color='blue', linewidth=1)
                ax[0].scatter(normal_y_plot, normal_x_plot, color='red', s=1)
                ax[0].set_title(f"Mito {mito_i} - Path length: {len(path_mito)}")
                ax[0].set_facecolor('black')
                ax[0].set_xlim(np.min(path_y)-20, np.max(path_y)+20)
                ax[0].set_ylim(np.min(path_x)-20, np.max(path_x)+20)

                # Plot scan channel
                ax[1].imshow(target_image, cmap='gray', alpha=1)
                ax[1].scatter(path_mito[:, 1], path_mito[:, 0], c=cm.winter(np.array(path_length)/np.max(path_length)))
                ax[1].plot(path_y, path_x, color='blue', linewidth=1)
                ax[1].scatter(normal_y_plot, normal_x_plot, color='red', s=1)
                ax[1].set_title(f"Scan {mito_i} - Path length: {len(path_mito)}")
                ax[1].set_facecolor('black')
                ax[1].set_xlim(np.min(path_y)-20, np.max(path_y)+20)
                ax[1].set_ylim(np.min(path_x)-20, np.max(path_x)+20)

                # Plot mask channel
                ax[2].imshow(mask_image, cmap='gray', alpha=1)
                ax[2].scatter(path_mito[:, 1], path_mito[:, 0], c=cm.winter(np.array(path_length)/np.max(path_length)))
                ax[2].plot(path_y, path_x, color='blue', linewidth=1)
                ax[2].scatter(normal_y_plot, normal_x_plot, color='red', s=1)
                ax[2].set_title(f"Mask {mito_i} - Path length: {len(path_mito)}")
                ax[2].set_facecolor('black')
                ax[2].set_xlim(np.min(path_y)-20, np.max(path_y)+20)
                ax[2].set_ylim(np.min(path_x)-20, np.max(path_x)+20)
                
                
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