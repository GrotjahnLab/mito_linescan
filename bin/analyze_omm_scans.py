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
    
    # Extract intensity data
    mito_intensities_all = []
    target_intensities_all = []
    mask_intensities_all = []
    distances_all = []
    distances_rev_all = []
    normal_line_points_all = []
    
    # Collect all intensity profiles and calculate distances
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlabel("Distance along normal (pixels)")
    ax.set_ylabel("Intensity")
    for idx, point_data in enumerate(detailed_data):
        mito_int = np.array(point_data['mito_intensities'])
        target_int = np.array(point_data['scan_intensities'])
        mask_int = np.array(point_data['mask_intensities'])
        skeleton_point = point_data['skeleton_point']
        normal_line_points = point_data['normal_line_points']
        distances = np.array(point_data['normal_distances'])

        #smooth all the profiles
        distances= smooth_profile(distances, window=3)
        mito_int = smooth_profile(mito_int, window=3)
        target_int = smooth_profile(target_int, window=3)
        mask_int = smooth_profile(mask_int, window=3)

        peaks, _ = find_peaks(mito_int)
        prominences = peak_prominences(mito_int, peaks)[0]
        #shift the distances so that the highest peak is at 0
        if len(peaks) > 0:
            highest_peak_idx = int(np.argmax(prominences))
            highest_peak = int(peaks[highest_peak_idx])
            distances = distances - distances[highest_peak]
        ax.plot(distances, mito_int, color='blue', label='Mito Intensity' if idx == 0 else '')
        ax.plot(distances, target_int, color='green', label='Scan Intensity' if idx == 0 else '')
        ax.plot(distances, mask_int, color='orange', label='Mask Intensity' if idx == 0 else '')
        ax.scatter(distances[peaks], mito_int[peaks], color='red', label='Peaks' if idx == 0 else '')
        #for i, peak in enumerate(peaks):
        #   ax.annotate(f'{mito_int[peak]:.2f}\n({distances[peak]:.2f})\nprom: {prominences[i]:.2f}', (distances[peak], mito_int[peak]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        #ax.set_title(f"Intensity profiles along normal for skeleton point {skeleton_point}")    
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{image_name}_intensity_profiles.png"), dpi=150, bbox_inches='tight')
    plt.close()


    #Now that the shifted distnces are there, we can make a cumulative plot of all the profiles, for that we go through -0.8 to 0.8 of the range of distances and calculate the average intensity of all profiles at each distance, then plot that with a shaded area for the standard deviation. We can also plot the individual profiles with low opacity in the background to show the variability.
    all_distances = np.concatenate([point_data['normal_distances'] for point_data in detailed_data])
    min_distance = np.min(all_distances)
    max_distance = np.max(all_distances)
    distance_range = max_distance - min_distance
    # Use -0.8 to 0.8 of the range
    normalized_min = -0.8 * distance_range / 2
    normalized_max = 0.8 * distance_range / 2
    distance_grid = np.linspace(normalized_min, normalized_max, 100)
    
    mito_interp_profiles = []
    target_interp_profiles = []
    mask_interp_profiles = []
    
    # Collect data for valid peaks that pass all criteria
    valid_peaks_data = []
    
    for point_data in detailed_data:
        mito_int = np.array(point_data['mito_intensities'])
        target_int = np.array(point_data['scan_intensities'])
        mask_int = np.array(point_data['mask_intensities'])
        distances = np.array(point_data['normal_distances'])
        normal_line_points = point_data['normal_line_points']
        
        if len(distances) < 2:
            continue
        
        peaks, _ = find_peaks(mito_int)
        prominences = peak_prominences(mito_int, peaks)[0]
        #shift the distances so that the highest peak is at 0
        if len(peaks) > 0:
            highest_peak_idx = int(np.argmax(prominences))
            highest_peak = int(peaks[highest_peak_idx])
            distances = distances - distances[highest_peak]
        try:
            mito_interp = np.interp(distance_grid, distances, mito_int)
            target_interp = np.interp(distance_grid, distances, target_int)
            mask_interp = np.interp(distance_grid, distances, mask_int)
        except Exception as e:
            print(f"Error interpolating profile: {e}")
            continue
        
        # Smooth interpolated profiles
        mito_interp = smooth_profile(mito_interp, window=3)
        target_interp = smooth_profile(target_interp, window=3)
        mask_interp = smooth_profile(mask_interp, window=3)
        
        # Find mito peaks
        mito_peaks, _ = find_peaks(mito_interp)
        if len(mito_peaks) == 0:
            continue
        mito_prominences = peak_prominences(mito_interp, mito_peaks)[0]
        highest_mito_peak_idx = int(np.argmax(mito_prominences))
        highest_mito_peak = mito_peaks[highest_mito_peak_idx]
        
        # Filter based on target profile meeting peak and prominence thresholds
        target_peaks, _ = find_peaks(target_interp)
        if len(target_peaks) == 0:
            continue
        target_prominences = peak_prominences(target_interp, target_peaks)[0]
        highest_target_peak_idx = int(np.argmax(target_prominences))
        highest_target_peak = target_peaks[highest_target_peak_idx]
        
        # Check if both mito and target peaks meet thresholds
        if (mito_interp[highest_mito_peak] < intensity_threshold or 
            mito_prominences[highest_mito_peak_idx] < prominence_threshold or
            target_interp[highest_target_peak] < intensity_threshold or 
            target_prominences[highest_target_peak_idx] < prominence_threshold):
            continue
        
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
        
        valid_peaks_data.append({
            'image_name': image_name,
            'mito_id': mito_id,
            'target_peak_distance': target_peak_distance,
            'mito_peak_image_point': mito_peak_image_point,
            'target_peak_image_point': target_peak_image_point
        })

        mito_interp_profiles.append(mito_interp)
        target_interp_profiles.append(target_interp)
        mask_interp_profiles.append(mask_interp)

    # Convert lists to arrays outside the loop
    if mito_interp_profiles:
        mito_interp_profiles = np.array(mito_interp_profiles)
        target_interp_profiles = np.array(target_interp_profiles)
        mask_interp_profiles = np.array(mask_interp_profiles)

        
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
        
        # Create cumulative plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot individual profiles with low opacity
        for profile in mito_interp_profiles:
            ax.plot(distance_grid, profile, color='blue', alpha=0.1, linewidth=0.5)
        for profile in target_interp_profiles:
            ax.plot(distance_grid, profile, color='green', alpha=0.1, linewidth=0.5)
        for profile in mask_interp_profiles:
            ax.plot(distance_grid, profile, color='orange', alpha=0.1, linewidth=0.5)
        # Plot mean with shaded error area
        ax.plot(distance_grid, mito_mean, color='blue', linewidth=2, label='Mito Mean')
        ax.fill_between(distance_grid, mito_mean - mito_std, mito_mean + mito_std, 
                        color='blue', alpha=0.2, label='Mito ± 1 SD')
        
        ax.plot(distance_grid, target_mean, color='green', linewidth=2, label='Target Mean')
        ax.fill_between(distance_grid, target_mean - target_std, target_mean + target_std, 
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


    if False:
        print(point_data['normal_distances'])
        # Only include if we have data
        if len(mito_int) > 0:
            mito_intensities_all.append(mito_int)
            target_intensities_all.append(target_int)
            mask_intensities_all.append(mask_int)
            distances_all.append(distances)
            normal_line_points_all.append(normal_line_points)

    
    aligned_mito_profiles = []
    aligned_target_profiles = []
    aligned_mask_profiles = []
    aligned_distances_list = []
    target_com_values = []
    mito_peak_values = []
    mito_com_points = []
    target_com_points = []
    mito_gaussian_fit_params = []
    shifted_distances_all = []
    

    #for each profile, find the highest peak and calculate its distance, then shift distances so that this peak is at 0. Also calculate the center of mass of the target profile and its distance, and save all these values for plotting and CSV output. Only include profiles where the highest peak is above the intensity and prominence thresholds.
    for i, profile in enumerate(mito_intensities_all):
        mito_profile = np.array(profile)
        
        
        # find peaks and their prominence
        peaks, _ = find_peaks(mito_profile)
        prominences = peak_prominences(mito_profile, peaks)[0]
        # get the highest peak and check if its  intensity and prominence are above the thresholds
        if len(peaks) == 0:
            continue
        highest_peak_idx = int(np.argmax(prominences))
        highest_peak = int(peaks[highest_peak_idx])
        if mito_profile[highest_peak] < intensity_threshold or prominences[highest_peak_idx] < prominence_threshold:
            continue
        
        mito_peak = highest_peak
        # Shift distances so Gaussian center is at 0
        mito_peak_dist = distances_all[i][mito_peak]
        shifted_distances = np.array(distances_all[i]) - mito_peak_dist
        #calculate the center of mass of scan intensities on shifted distances, to see what side of the mito the septin is on
        target_profile = np.array(target_intensities_all[i])
        target_com = np.sum(shifted_distances * target_profile) / np.sum(target_profile)
        
        # Find the actual image points corresponding to mito_com and target_com
        
        # Find closest normal_line_point to this distance
        closest_idx = int(np.argmin(np.abs(shifted_distances - target_com)))
        target_com_point = normal_line_points_all[i][closest_idx]
        
        mito_peak_values.append(mito_peak_dist)
        mito_com_points.append(mito_peak_dist)
        target_com_values.append(target_com)
        target_com_points.append(target_com_point)
        shifted_distances_all.append(shifted_distances) 
        #print(f"Profile {i}: Mito Peak at {mito_peak_dist:.2f} -> {mito_peak_dist}, Target COM at {target_com:.2f} -> {target_com_point}")

        # Create aligned distance array matching the interpolated profiles
        #aligned_distances = np.linspace(shifted_distances[0], shifted_distances[-1], 100)
        aligned_distances_list.append(shifted_distances)
        
        #aligned_mito_profiles.append(np.interp(aligned_distances, shifted_distances, mito_profile))
        #aligned_target_profiles.append(np.interp(aligned_distances, shifted_distances, target_profile))
        #aligned_mask_profiles.append(np.interp(aligned_distances, shifted_distances, mask_intensities_all[i]))
    
    #Plot individual aligned profiles with their distances
    if not aligned_mito_profiles:
        print(f"No valid profiles found for {image_name} mito {mito_id}")
        return None
    plt.figure(figsize=(6, 4))
    for i in range(len(aligned_mito_profiles)):

        print(f"Plotting profile {i+1}/{len(aligned_mito_profiles)} for {image_name} mito {mito_id}")
        plt.plot(distances_all[i], mito_intensities_all[i], color='blue', alpha=0.5, label='Mito' if i == 0 else "")
        plt.plot(distances_all[i], target_intensities_all[i], color='green', alpha=0.5, label='Target' if i == 0 else "")
        plt.plot(distances_all[i], mask_intensities_all[i], color=np.random.rand(3), alpha=0.5, label='Mask' if i == 0 else "")
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    plt.xlabel('Distance from threshold crossing (pixels)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title(f'{image_name} - Mito {mito_id} - Profile {i}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    output_file_individual = os.path.join(output_dir, f'{image_name}_mito_{mito_id}_intensity_profile_{i}.png')
    plt.savefig(output_file_individual, dpi=150, bbox_inches='tight')
    print(f"Saved individual profile plot to {output_file_individual}")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    if aligned_mito_profiles:
        avg_mito_profile = np.mean(aligned_mito_profiles, axis=0)
        avg_target_profile = np.mean(aligned_target_profiles, axis=0)
        # Use the first profile's distance grid as reference for averaging
        avg_distances = aligned_distances_list[0]
        
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
        'mito_gaussian_fit_params': mito_gaussian_fit_params,
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
        csv_file = os.path.join(output_dir, 'target_com_values.csv')
        
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
        
        # Write valid peaks data to separate CSV
        valid_peaks_list = []
        for result in results:
            if 'valid_peaks_data' in result and result['valid_peaks_data']:
                valid_peaks_list.extend(result['valid_peaks_data'])
        
        if valid_peaks_list:
            csv_file_peaks = os.path.join(output_dir, 'valid_peaks_data.csv')
            with open(csv_file_peaks, 'w', newline='') as f:
                fieldnames = ['image_name', 'mito_id', 'target_peak_distance', 'mito_peak_image_point', 'target_peak_image_point']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for peak_data in valid_peaks_list:
                    row = {
                        'image_name': peak_data['image_name'],
                        'mito_id': peak_data['mito_id'],
                        'target_peak_distance': f'{peak_data["target_peak_distance"]:.4f}',
                        'mito_peak_image_point': str(peak_data['mito_peak_image_point']),
                        'target_peak_image_point': str(peak_data['target_peak_image_point'])
                    }
                    writer.writerow(row)
            print(f"Saved {len(valid_peaks_list)} valid peaks to {csv_file_peaks}")


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
