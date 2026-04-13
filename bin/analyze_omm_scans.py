#!/usr/bin/env python3

"""
Script to read pickle files from mito_septin_localization.py and visualize intensity profiles.
Creates 3 plots showing mito, septin, and mask intensities overlapped with transparency.
"""

import pickle
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import click
from scipy.optimize import curve_fit
import csv
from scipy.signal import find_peaks, peak_prominences


# Smooth profiles by averaging every 5 neighbors
def smooth_profile(profile, window=5):
    """Apply moving average smoothing to a profile."""
    if len(profile) < window:
        return profile
    smoothed = np.convolve(profile, np.ones(window)/window, mode='valid')
    return smoothed



def visualize_intensity_profiles(pkl_file, output_dir=None, intensity_threshold=0.3, prominence_threshold=0.1):
    """
    Read a pickle file and create intensity profile plots.
    
    Args:
        pkl_file: Path to the pickle file to read
        output_dir: Directory to save plots (defaults to same as pkl file)
        intensity_threshold: Minimum intensity threshold for peaks
        prominence_threshold: Minimum prominence threshold for peaks
        
    Returns:
        Dictionary with image_name, mito_id, and list of COM values
    """
    
    # Load the pickle file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    detailed_data = data['detailed_data']
    mito_id = data['mito_id']
    image_name = data['image_name']
    
    if output_dir is None:
        output_dir = os.path.dirname(pkl_file)
    
    # Extract intensity data
    mito_intensities_all = []
    target_intensities_all = []
    mask_intensities_all = []
    distances_all = []
    distances_rev_all = []
    normal_line_points_all = []
    
    # Collect all intensity profiles and calculate distances
    for point_data in detailed_data:
        mito_int = point_data['mito_intensities']
        target_int = point_data['scan_intensities']
        mask_int = point_data['mask_intensities']
        skeleton_point = point_data['skeleton_point']
        normal_line_points = point_data['normal_line_points']
        
        # Only include if we have data
        if len(mito_int) > 0:
            # Calculate distances from the first point to each normal line point
            distances = []
            for (nx, ny) in normal_line_points:
                dist = np.sqrt((nx - normal_line_points[0][0])**2 + (ny - normal_line_points[0][1])**2)
                distances.append(dist)
            # Calculate the distances from the last point to each normal line point
            distances_rev = []
            for (nx, ny) in normal_line_points:
                dist = np.sqrt((nx - normal_line_points[-1][0])**2 + (ny - normal_line_points[-1][1])**2)
                distances_rev.append(dist)
            
            mito_intensities_all.append(mito_int)
            target_intensities_all.append(target_int)
            mask_intensities_all.append(mask_int)
            distances_all.append(distances)
            distances_rev_all.append(distances_rev)
            normal_line_points_all.append(normal_line_points)
    
    # Orient profiles based on mask intensities
    # If first half average > second half average, reverse all three
    for i in range(len(mask_intensities_all)):
        mask_int = np.array(mask_intensities_all[i])
        midpoint = len(mask_int) // 2
        first_half_avg = np.mean(mask_int[:midpoint])
        second_half_avg = np.mean(mask_int[midpoint:])
        
        if first_half_avg > second_half_avg:
            #mito_intensities_all[i] = mito_intensities_all[i][::-1]
            #target_intensities_all[i] = target_intensities_all[i][::-1]
            #mask_intensities_all[i] = mask_intensities_all[i][::-1]
            distances_all[i] = distances_rev_all[i]
    

    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Intensity Profiles - Mito {mito_id} ({image_name})', fontsize=16)
    
    # Plot mito intensities
    for i, profile in enumerate(mito_intensities_all):
        axes[0].plot(distances_all[i], profile, color='blue', alpha=0.1, linewidth=0.5)
    axes[0].set_title('Mito Intensities')
    axes[0].set_xlabel('Distance from skeleton (pixels)')
    axes[0].set_ylabel('Intensity')
    axes[0].set_facecolor('white')
    axes[0].grid(True, alpha=0.3)
    
    # Plot target intensities
    for i, profile in enumerate(target_intensities_all):
        axes[1].plot(distances_all[i], profile, color='green', alpha=0.1, linewidth=0.5)
    axes[1].set_title('Target Intensities')
    axes[1].set_xlabel('Distance from skeleton (pixels)')
    axes[1].set_ylabel('Intensity')
    axes[1].set_facecolor('white')
    axes[1].grid(True, alpha=0.3)
    
    # Plot mask intensities
    for i, profile in enumerate(mask_intensities_all):
        axes[2].plot(distances_all[i], profile, color='red', alpha=0.1, linewidth=0.5)
    axes[2].set_title('Mask Intensities')
    axes[2].set_xlabel('Distance from skeleton (pixels)')
    axes[2].set_ylabel('Intensity')
    axes[2].set_facecolor('white')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f'{image_name}_mito_{mito_id}_intensity_profiles.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()
    
    #average all the aligned lines and plot them and save
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'Average Aligned Intensity Profiles (mito > 0.2 at x=0) - Mito {mito_id} ({image_name})', fontsize=14)
    
    threshold = 0.05
    aligned_mito_profiles = []
    aligned_target_profiles = []
    target_com_values = []
    mito_peak_values = []
    mito_com_points = []
    target_com_points = []
    mito_gaussian_fit_params = []
    mito_gaussian_points = []

    for i, profile in enumerate(mito_intensities_all):
        mito_profile = np.array(profile)
        
        
        # find peaks and their prominence
        peaks, _ = find_peaks(mito_profile)
        prominences = peak_prominences(mito_profile, peaks)[0]
        # get the highest peak and check if its  intensity and prominence are above the thresholds
        if len(peaks) == 0:
            continue
        highest_peak_idx = np.argmax(prominences)
        highest_peak = peaks[highest_peak_idx]
        if mito_profile[highest_peak] < intensity_threshold or prominences[highest_peak_idx] < prominence_threshold:
            continue
        
        mito_peak = highest_peak
        # Shift distances so Gaussian center is at 0
        shifted_distances = np.array(distances_all[i]) - mito_peak
        #calculate the center of mass of scan intensities on shifted distances, to see what side of the mito the septin is on
        target_profile = np.array(target_intensities_all[i])
        target_com = np.sum(np.array(distances_all[i]) * target_profile) / np.sum(target_profile)
        
        # Find the actual image points corresponding to mito_com and target_com
        mito_peak_dist = distances_all[i][mito_peak]



        # For COM: find which distance corresponds to the COM value
        # com is relative to shifted_distances, need to convert back to original distance
        # Find closest normal_line_point to this distance
        closest_idx = np.argmin(np.abs(np.array(distances_all[i]) - target_com))
        target_com_point = normal_line_points_all[i][closest_idx]
        
        mito_peak_values.append(mito_peak_dist)
        mito_com_points.append(mito_peak_dist)
        target_com_values.append(target_com)
        target_com_points.append(target_com_point)
        print(f"Profile {i}: Mito COM at {mito_peak_dist:.2f} -> {mito_peak_dist}, Target COM at {target_com:.2f} -> {target_com_point}")

        aligned_mito_profiles.append(np.interp(np.linspace(shifted_distances[0], shifted_distances[-1], 100), shifted_distances, mito_profile))
        aligned_target_profiles.append(np.interp(np.linspace(shifted_distances[0], shifted_distances[-1], 100), shifted_distances, target_profile))
    
    if aligned_mito_profiles:
        avg_mito_profile = np.mean(aligned_mito_profiles, axis=0)
        avg_target_profile = np.mean(aligned_target_profiles, axis=0)
        avg_distances = np.linspace(shifted_distances[0], shifted_distances[-1], 100)
        
        ax.plot(avg_distances, avg_mito_profile, color='blue', linewidth=2, label='Average Mito')
        ax.plot(avg_distances, avg_target_profile, color='green', linewidth=2, label='Average Target')
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Distance from threshold crossing (pixels)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save the average aligned plot
    output_file_avg_aligned = os.path.join(output_dir, f'{image_name}_mito_{mito_id}_intensity_profiles_avg_aligned.png')
    plt.savefig(output_file_avg_aligned, dpi=150, bbox_inches='tight')
    print(f"Saved average aligned plot to {output_file_avg_aligned}")
    plt.close()
    
    # Return results for CSV writing
    return {
        'image_name': image_name,
        'mito_id': mito_id,
        'mito_peak_values': mito_peak_values,
        'target_com_values': target_com_values,
        'mito_com_points': mito_com_points,
        'target_com_points': target_com_points,
        'mito_gaussian_fit_params': mito_gaussian_fit_params

    }


def process_directory(pkl_dir, intensity_threshold=0.3, prominence_threshold=0.1):
    """
    Process all pickle files in a directory.
    
    Args:
        pkl_dir: Directory containing pickle files
        intensity_threshold: Minimum intensity threshold for peaks
        prominence_threshold: Minimum prominence threshold for peaks
    """
    pkl_files = glob.glob(os.path.join(pkl_dir, '*_detailed.pkl'))
    
    if not pkl_files:
        print(f"No pickle files found in {pkl_dir}")
        return
    
    print(f"Found {len(pkl_files)} pickle files")
    print(f"Using intensity_threshold={intensity_threshold}, prominence_threshold={prominence_threshold}")
    
    # Collect results for CSV
    results = []
    
    for pkl_file in pkl_files:
        print(f"\nProcessing {os.path.basename(pkl_file)}...")
        try:
            result = visualize_intensity_profiles(pkl_file, pkl_dir, intensity_threshold, prominence_threshold)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
    
    # Write results to CSV
    if results:
        csv_file = os.path.join(pkl_dir, 'target_com_values.csv')
        
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['image_name', 'mito_com', 'target_com', 'mito_com_point', 'target_com_point', 'mito_gaussian_fit_params']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                image_name = result['image_name']
                # Write one row per scan
                for mito_com, target_com, mito_com_point, target_com_point, mito_gaussian_fit in zip(result['mito_peak_values'], result['target_com_values'], result['mito_com_points'], result['target_com_points'], result['mito_gaussian_fit_params']):
                    row = {
                        'image_name': image_name,
                        'mito_com': f'{mito_com:.4f}',
                        'target_com': f'{target_com:.4f}',
                        'mito_com_point': f'{mito_com_point}',
                        'target_com_point': f'{target_com_point}',
                        'mito_gaussian_fit_params': f'{mito_gaussian_fit}'
                    }
                    writer.writerow(row)
        
        print(f"\nSaved scan data to {csv_file}")


@click.command()
@click.option('--directory', type=click.Path(exists=True), required=True, help='Directory containing pickle files')
@click.option('--peak-threshold', type=float, default=0.3, help='Minimum intensity threshold for peaks (default: 0.3)')
@click.option('--peak-prominence', type=float, default=0.1, help='Minimum prominence threshold for peaks (default: 0.1)')
def main(directory, peak_threshold, peak_prominence):
    """
    Analyze OMM scan pickle files and generate intensity profile plots and CSV reports.
    
    Example:
        python analyze_omm_scans.py --directory test/ --peak-threshold 0.1 --peak-prominence 0.02
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        sys.exit(1)
    
    process_directory(directory, peak_threshold, peak_prominence)
    print("\nDone!")


if __name__ == "__main__":
    main()
