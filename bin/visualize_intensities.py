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
from scipy.optimize import curve_fit
import csv

# Smooth profiles by averaging every 5 neighbors
def smooth_profile(profile, window=5):
    """Apply moving average smoothing to a profile."""
    if len(profile) < window:
        return profile
    smoothed = np.convolve(profile, np.ones(window)/window, mode='valid')
    return smoothed

def gaussian(x, a, x0, sigma):
    """Gaussian function for curve fitting."""
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def visualize_intensity_profiles(pkl_file, output_dir=None):
    """
    Read a pickle file and create intensity profile plots.
    
    Args:
        pkl_file: Path to the pickle file to read
        output_dir: Directory to save plots (defaults to same as pkl file)
        
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
    septin_intensities_all = []
    mask_intensities_all = []
    distances_all = []
    distances_rev_all = []
    normal_line_points_all = []
    
    # Collect all intensity profiles and calculate distances
    for point_data in detailed_data:
        mito_int = point_data['mito_intensities']
        septin_int = point_data['scan_intensities']
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
            septin_intensities_all.append(septin_int)
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
            #septin_intensities_all[i] = septin_intensities_all[i][::-1]
            #mask_intensities_all[i] = mask_intensities_all[i][::-1]
            distances_all[i] = distances_rev_all[i]
    

    
    # Apply smoothing to all profiles and adjust distances
    for i in range(len(mito_intensities_all)):
        window = 3
        if len(mito_intensities_all[i]) >= window:
            mito_intensities_all[i] = smooth_profile(mito_intensities_all[i], window)
            septin_intensities_all[i] = smooth_profile(septin_intensities_all[i], window)
            mask_intensities_all[i] = smooth_profile(mask_intensities_all[i], window)
            # Adjust distances to match smoothed length (smoothing reduces length by window-1)
            distances_all[i] = distances_all[i][window//2:len(distances_all[i])-window//2]
    
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
    
    # Plot septin intensities
    for i, profile in enumerate(septin_intensities_all):
        axes[1].plot(distances_all[i], profile, color='green', alpha=0.1, linewidth=0.5)
    axes[1].set_title('Septin Intensities')
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
    
    # Also create a combined plot with all three overlapped
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'Combined Intensity Profiles - Mito {mito_id} ({image_name})', fontsize=14)
    
    for i, profile in enumerate(mito_intensities_all):
            ax.plot(distances_all[i], profile, color='blue', alpha=0.1, linewidth=0.5, label='Mito' if i == 0 else '')
    
    for i, profile in enumerate(septin_intensities_all):
            ax.plot(distances_all[i], profile, color='green', alpha=0.1, linewidth=0.5, label='Septin' if i == 0 else '')
    
    for i, profile in enumerate(mask_intensities_all):
            ax.plot(distances_all[i], profile, color='red', alpha=0.1, linewidth=0.5, label='Mask' if i == 0 else '')
    
    ax.set_xlabel('Distance from skeleton (pixels)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save the combined plot
    output_file_combined = os.path.join(output_dir, f'{image_name}_mito_{mito_id}_intensity_profiles_combined.png')
    plt.savefig(output_file_combined, dpi=150, bbox_inches='tight')
    print(f"Saved combined plot to {output_file_combined}")
    plt.close()
    
    # Create aligned plot where first mito > 0.2 is at x=0
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'Aligned Intensity Profiles (mito > 0.2 at x=0) - Mito {mito_id} ({image_name})', fontsize=14)
    
    threshold = 0.05
    for i, profile in enumerate(mito_intensities_all):
        mito_profile = np.array(profile)
        
        # Find first index where mito intensity exceeds threshold
        above_threshold = np.where(mito_profile > threshold)[0]
        
        if len(above_threshold) > 0:
            shift_idx = above_threshold[0]
            # Shift distances so this point is at 0
            shifted_distances = np.array(distances_all[i]) - distances_all[i][shift_idx]
            
            # Plot mito and septin channels with shifted distances
            ax.plot(shifted_distances, mito_profile, color='blue', alpha=0.1, linewidth=0.5, label='Mito' if i == 0 else '')
            ax.plot(shifted_distances, septin_intensities_all[i], color='green', alpha=0.1, linewidth=0.5, label='Septin' if i == 0 else '')
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Distance from threshold crossing (pixels)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save the aligned plot
    output_file_aligned = os.path.join(output_dir, f'{image_name}_mito_{mito_id}_intensity_profiles_aligned.png')
    plt.savefig(output_file_aligned, dpi=150, bbox_inches='tight')
    print(f"Saved aligned plot to {output_file_aligned}")
    plt.close()
    #average all the aligned lines and plot them and save
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'Average Aligned Intensity Profiles (mito > 0.2 at x=0) - Mito {mito_id} ({image_name})', fontsize=14)
    
    threshold = 0.05
    aligned_mito_profiles = []
    aligned_septin_profiles = []
    com_values = []
    rmse_values = []
    gaussian_centers = []
    gaussian_points = []
    com_points = []
    
    for i, profile in enumerate(mito_intensities_all):
        mito_profile = np.array(profile)
        
        # Fit a Gaussian to the mito profile and shift center to zero
        
        # Define Gaussian function
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
        
        try:
            # Fit Gaussian to mito profile
            x_data = np.arange(len(mito_profile))
            popt, pcov = curve_fit(gaussian, x_data, mito_profile, p0=[np.max(mito_profile), np.argmax(mito_profile), 5])
            center_idx = popt[1]
            
            # Calculate fit error (residuals and R-squared)
            fitted_profile = gaussian(x_data, *popt)
            residuals = mito_profile - fitted_profile
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((mito_profile - np.mean(mito_profile))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Shift distances so Gaussian center is at 0
            shifted_distances = np.array(distances_all[i]) - distances_all[i][int(center_idx)]
            
            #calculate the center of mass of scan intensities on shifted distances, to see what side of the mito the septin is on
            septin_profile = np.array(septin_intensities_all[i])
            com = np.sum(shifted_distances * septin_profile) / np.sum(septin_profile)
            
            # Find the actual image points corresponding to Gaussian center and COM
            # Gaussian center is at center_idx in the original distances
            gaussian_dist = distances_all[i][int(center_idx)]
            # Find closest normal_line_point to this distance
            gaussian_point_idx = int(np.round(center_idx)) if int(center_idx) < len(normal_line_points_all[i]) else len(normal_line_points_all[i]) - 1
            gaussian_point = normal_line_points_all[i][gaussian_point_idx]
            
            # For COM: find which distance corresponds to the COM value
            # com is relative to shifted_distances, need to convert back to original distance
            com_dist = gaussian_dist + com
            # Find closest normal_line_point to this distance
            closest_idx = np.argmin(np.abs(np.array(distances_all[i]) - com_dist))
            com_point = normal_line_points_all[i][closest_idx]
            
            com_values.append(com)
            rmse_values.append(rmse)
            gaussian_centers.append(center_idx)
            gaussian_points.append(gaussian_point)
            com_points.append(com_point)
            print(f"Profile {i}: Gaussian center at {center_idx:.2f} -> {gaussian_point}, Septin COM at {com:.2f} -> {com_point}, R²={r_squared:.4f}, RMSE={rmse:.4f}")

            aligned_mito_profiles.append(np.interp(np.linspace(shifted_distances[0], shifted_distances[-1], 100), shifted_distances, mito_profile))
            aligned_septin_profiles.append(np.interp(np.linspace(shifted_distances[0], shifted_distances[-1], 100), shifted_distances, septin_intensities_all[i]))
        except:
            # Skip profiles where Gaussian fit fails
            pass
    
    if aligned_mito_profiles:
        avg_mito_profile = np.mean(aligned_mito_profiles, axis=0)
        avg_septin_profile = np.mean(aligned_septin_profiles, axis=0)
        avg_distances = np.linspace(shifted_distances[0], shifted_distances[-1], 100)
        
        ax.plot(avg_distances, avg_mito_profile, color='blue', linewidth=2, label='Average Mito')
        ax.plot(avg_distances, avg_septin_profile, color='green', linewidth=2, label='Average Septin')
    
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
        'com_values': com_values,
        'rmse_values': rmse_values,
        'gaussian_centers': gaussian_centers,
        'gaussian_points': gaussian_points,
        'com_points': com_points
    }


def process_directory(pkl_dir):
    """
    Process all pickle files in a directory.
    
    Args:
        pkl_dir: Directory containing pickle files
    """
    pkl_files = glob.glob(os.path.join(pkl_dir, '*_detailed.pkl'))
    
    if not pkl_files:
        print(f"No pickle files found in {pkl_dir}")
        return
    
    print(f"Found {len(pkl_files)} pickle files")
    
    # Collect results for CSV
    results = []
    
    for pkl_file in pkl_files:
        print(f"\nProcessing {os.path.basename(pkl_file)}...")
        try:
            result = visualize_intensity_profiles(pkl_file, pkl_dir)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
    
    # Write results to CSV
    if results:
        csv_file = os.path.join(pkl_dir, 'septin_com_values.csv')
        
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['image_name', 'gaussian_rmse', 'gaussian_center', 'scan_com', 'gaussian_point', 'com_point']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                image_name = result['image_name']
                # Write one row per scan
                for rmse, center, com, gauss_pt, com_pt in zip(result['rmse_values'], result['gaussian_centers'], result['com_values'], result['gaussian_points'], result['com_points']):
                    row = {
                        'image_name': image_name,
                        'gaussian_rmse': f'{rmse:.4f}',
                        'gaussian_center': f'{center:.4f}',
                        'scan_com': f'{com:.4f}',
                        'gaussian_point': f'{gauss_pt}',
                        'com_point': f'{com_pt}'
                    }
                    writer.writerow(row)
        
        print(f"\nSaved scan data to {csv_file}")


if __name__ == "__main__":

    
    if len(sys.argv) < 2:
        # Default to test/ directory if no argument provided
        pkl_dir = "test/"
    else:
        pkl_dir = sys.argv[1]
    
    if not os.path.exists(pkl_dir):
        print(f"Directory {pkl_dir} does not exist")
        sys.exit(1)
    
    process_directory(pkl_dir)
    print("\nDone!")
