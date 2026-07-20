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
    smoothed = np.convolve(profile, np.ones(window)/window, mode='same')
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

    
    # Collect all intensity profiles and calculate distances
    mito_interp_profiles = []
    target_interp_profiles = []
    mask_interp_profiles = []
    valid_peaks_data = []
    distances_shifted = []

    for idx, point_data in enumerate(detailed_data):
        mito_int = np.array(point_data['mito_intensities'])
        target_int = np.array(point_data['scan_intensities'])
        mask_int = np.array(point_data['mask_intensities'])
        skeleton_point = point_data['skeleton_point']
        normal_line_points = point_data['normal_line_points']
        distances = np.array(point_data['normal_distances'])

        # SCI-1: never smooth the coordinate (distance) axis — it is the `xp`
        # grid for np.interp and must stay raw and strictly monotonic. Smooth
        # only the intensity profiles.
        mito_int = smooth_profile(mito_int, window=3)
        target_int = smooth_profile(target_int, window=3)
        mask_int = smooth_profile(mask_int, window=3)
        
        # Create a common distance grid for interpolation
        distance_grid = np.linspace(np.min(distances), np.max(distances), num=200)        
        mito_interp = np.interp(distance_grid, distances, mito_int)
        target_interp = np.interp(distance_grid, distances, target_int)
        mask_interp = np.interp(distance_grid, distances, mask_int)
        
        # Smooth interpolated profiles
        mito_interp = smooth_profile(mito_interp, window=7)
        target_interp = smooth_profile(target_interp, window=7)
        mask_interp = smooth_profile(mask_interp, window=7)
        
        # Find mito peaks
        mito_peaks, _ = find_peaks(mito_interp, height=intensity_threshold, prominence=prominence_threshold)
        if len(mito_peaks) == 0:
            print(f"No mito peaks found for {image_name} mito {mito_id} at point index {idx}")
            continue
        mito_prominences = peak_prominences(mito_interp, mito_peaks)[0]
        highest_mito_peak_idx = int(np.argmax(mito_interp[mito_peaks]))
        highest_mito_peak = mito_peaks[highest_mito_peak_idx]
        
        # Filter based on target profile meeting peak and prominence thresholds
        target_peaks, _ = find_peaks(target_interp, height=intensity_threshold, prominence=prominence_threshold)
        if len(target_peaks) == 0:
            print(f"No target peaks found for {image_name} mito {mito_id} at point index {idx}")
            continue
        target_prominences = peak_prominences(target_interp, target_peaks)[0]
        highest_target_peak_idx = int(np.argmax(target_interp[target_peaks]))
        highest_target_peak = target_peaks[highest_target_peak_idx]
        
        
        # Check if mask profile is larger on left side of mito peak (scan done properly)
        mask_left = mask_interp[:highest_mito_peak]
        mask_right = mask_interp[highest_mito_peak:]
        if np.mean(mask_left) <= np.mean(mask_right):
            continue
        

        # All criteria met - collect data for CSV
        target_peak_distance = distance_grid[highest_target_peak]
        mito_peak_distance = distance_grid[highest_mito_peak]
        
        # Map back to original normal_line_points indices
        closest_mito_idx = int(np.argmin(np.abs(distances - mito_peak_distance)))
        closest_target_idx = int(np.argmin(np.abs(distances - target_peak_distance)))
        
        mito_peak_image_point = normal_line_points[closest_mito_idx]
        target_peak_image_point = normal_line_points[closest_target_idx]
        # collect data for CSV: peaks value and distances, prominences, and image points
        valid_peaks_data.append({
            'image_name': image_name,
            'mito_id': mito_id,
            'mito_peak_distance': mito_peak_distance,
            'mito_peak_intensity': mito_interp[highest_mito_peak],
            'mito_peak_prominence': mito_prominences[highest_mito_peak_idx],
            'target_peak_distance': target_peak_distance,
            'target_peak_intensity': target_interp[highest_target_peak],
            'target_peak_prominence': target_prominences[highest_target_peak_idx],
            'mito_peak_image_point': mito_peak_image_point,
            'target_peak_image_point': target_peak_image_point
        })

        mito_interp_profiles.append(mito_interp)
        target_interp_profiles.append(target_interp)
        mask_interp_profiles.append(mask_interp)
        distances_shifted.append(distance_grid - distance_grid[highest_mito_peak])

    # Convert lists to arrays outside the loop
    if mito_interp_profiles:
        mito_interp_profiles = np.array(mito_interp_profiles)
        target_interp_profiles = np.array(target_interp_profiles)
        mask_interp_profiles = np.array(mask_interp_profiles)

        distances_shifted = np.array(distances_shifted)
        mito_interp_profiles = mito_interp_profiles / np.max(mito_interp_profiles, axis=1, keepdims=True)
        target_interp_profiles = target_interp_profiles / np.max(target_interp_profiles, axis=1, keepdims=True)
        mask_interp_profiles = mask_interp_profiles / np.max(mask_interp_profiles, axis=1, keepdims=True)        
        # Calculate mean and standard deviation
        mito_mean = np.mean(mito_interp_profiles, axis=0)
        mito_std = np.std(mito_interp_profiles, axis=0)
        target_mean = np.mean(target_interp_profiles, axis=0)
        target_std = np.std(target_interp_profiles, axis=0)
        mask_mean = np.mean(mask_interp_profiles, axis=0)
        mask_std = np.std(mask_interp_profiles, axis=0)

        # Common x-axis for mean curves: average of the per-profile shifted grids
        mean_distances_shifted = np.mean(distances_shifted, axis=0)

        # Create cumulative plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot individual profiles with low opacity
        for i,profile in enumerate(mito_interp_profiles):
            ax.plot(distances_shifted[i], profile, color='blue', alpha=0.1, linewidth=0.5)
        for i,profile in enumerate(target_interp_profiles):
            ax.plot(distances_shifted[i], profile, color='green', alpha=0.1, linewidth=0.5)
        for i,profile in enumerate(mask_interp_profiles):
            ax.plot(distances_shifted[i], profile, color='orange', alpha=0.1, linewidth=0.5)
        # Plot mean with shaded error area
        ax.plot(mean_distances_shifted, mito_mean, color='blue', linewidth=2, label='Mito Mean')
        ax.fill_between(mean_distances_shifted, mito_mean - mito_std, mito_mean + mito_std,
                        color='blue', alpha=0.2, label='Mito ± 1 SD')

        ax.plot(mean_distances_shifted, target_mean, color='green', linewidth=2, label='Target Mean')
        ax.fill_between(mean_distances_shifted, target_mean - target_std, target_mean + target_std,
                        color='green', alpha=0.2, label='Target ± 1 SD')

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Distance along normal (pixels)', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.set_title(f'{image_name} - Mito {mito_id} - Cumulative Profiles', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        output_file_cumulative = os.path.join(output_dir, f'{image_name}_mito_{mito_id}_intensity_profiles_cumulative.png')
        plt.savefig(output_file_cumulative, dpi=150, bbox_inches='tight')
        print(f"Saved cumulative profile plot to {output_file_cumulative}")
        plt.close()
    
    # Return results for CSV writing
    return {
        'image_name': image_name,
        'mito_id': mito_id,
        'valid_peaks_data': valid_peaks_data
    }


def process_directory(pkl_dir, output_dir=None, intensity_threshold=0.3, prominence_threshold=0.1):
    """
    Process all pickle files in a directory.
    
    Args:
        pkl_dir: Directory containing pickle files
        output_dir: Directory to save plots and CSV (defaults to pkl_dir)
        intensity_threshold: Minimum intensity threshold for peaks
        prominence_threshold: Minimum prominence threshold for peaks
    """
    if output_dir is None:
        output_dir = pkl_dir
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
            result = visualize_intensity_profiles(pkl_file, output_dir, intensity_threshold, prominence_threshold)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
    
    # Write results to CSV
    if results:
        # Write valid peaks data to CSV
        valid_peaks_list = []
        for result in results:
            if 'valid_peaks_data' in result and result['valid_peaks_data']:
                valid_peaks_list.extend(result['valid_peaks_data'])
        
        if valid_peaks_list:
            csv_file_peaks = os.path.join(output_dir, 'scan_data.csv')
            with open(csv_file_peaks, 'w', newline='') as f:
                fieldnames = ['image_name', 'mito_id', 'mito_peak_distance', 'mito_peak_intensity', 'mito_peak_prominence', 'target_peak_distance', 'target_peak_intensity', 'target_peak_prominence', 'mito_peak_image_point', 'target_peak_image_point']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for peak_data in valid_peaks_list:
                    row = {
                        'image_name': peak_data['image_name'],
                        'mito_id': peak_data['mito_id'],
                        'mito_peak_distance': f'{peak_data["mito_peak_distance"]:.4f}',
                        'mito_peak_intensity': f'{peak_data["mito_peak_intensity"]:.4f}',
                        'mito_peak_prominence': f'{peak_data["mito_peak_prominence"]:.4f}',
                        'target_peak_distance': f'{peak_data["target_peak_distance"]:.4f}',
                        'target_peak_intensity': f'{peak_data["target_peak_intensity"]:.4f}',
                        'target_peak_prominence': f'{peak_data["target_peak_prominence"]:.4f}',
                        'mito_peak_image_point': f'{peak_data["mito_peak_image_point"]}',
                        'target_peak_image_point': f'{peak_data["target_peak_image_point"]}'
                    }
                    writer.writerow(row)
            
            print(f"\nSaved scan data to {csv_file_peaks}")


@click.command()
@click.option('--input-directory', type=click.Path(exists=True), required=True, help='Directory containing pickle files')
@click.option('--output-directory', type=click.Path(), default=None, help='Output directory for plots and CSV (defaults to input directory)')
@click.option('--peak-threshold', type=float, default=0.3, help='Minimum intensity threshold for peaks (default: 0.3)')
@click.option('--peak-prominence', type=float, default=0.1, help='Minimum prominence threshold for peaks (default: 0.1)')
def main(input_directory, output_directory, peak_threshold, peak_prominence):
    """
    Analyze OMM scan pickle files and generate intensity profile plots and CSV reports.
    
    Example:
        python analyze_omm_scans.py --input-directory test/ --output-directory results/ --peak-threshold 0.1 --peak-prominence 0.02
    """
    if not os.path.exists(input_directory):
        print(f"Directory {input_directory} does not exist")
        sys.exit(1)
    
    process_directory(input_directory, output_directory, peak_threshold, peak_prominence)
    print("\nDone!")


if __name__ == "__main__":
    main()
