#!/usr/bin/env python3

"""
Script to read scan_data.csv and create a whisker plot of the distance difference between mito and target peaks,
grouped by the first 5 characters of the image name.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import click
import os


def identify_outliers(df):
    """
    Identify outliers in peak_distance_diff for each group using the IQR method.
    
    Args:
        df: DataFrame with 'group', 'peak_distance_diff', and 'image_name' columns
        
    Returns:
        List of tuples: (group, image_name, peak_distance_diff)
    """
    outliers = []
    
    groups = sorted(df['group'].unique())
    for group in groups:
        group_data = df[df['group'] == group]
        values = group_data['peak_distance_diff']
        
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        group_outliers = group_data[(group_data['peak_distance_diff'] < lower_bound) | 
                                     (group_data['peak_distance_diff'] > upper_bound)]
        
        for _, row in group_outliers.iterrows():
            outliers.append({
                'group': group,
                'image_name': row['image_name'],
                'peak_distance_diff': row['peak_distance_diff'],
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
    
    return outliers


def analyze_peak_distances(csv_file, output_dir=None, group_by_chars=5, min_mito_intensity=None, max_mito_intensity=None, 
                          min_target_intensity=None, max_target_intensity=None,
                          min_mito_prominence=None, max_mito_prominence=None,
                          min_target_prominence=None, max_target_prominence=None):
    """
    Read scan_data.csv and create a violin plot of peak distance differences.
    
    Args:
        csv_file: Path to the scan_data.csv file
        output_dir: Directory to save the plot (defaults to same directory as csv file)
        group_by_chars: Number of starting characters of image_name to use for grouping (default: 5)
        min_mito_intensity: Minimum mito peak intensity threshold
        max_mito_intensity: Maximum mito peak intensity threshold
        min_target_intensity: Minimum target peak intensity threshold
        max_target_intensity: Maximum target peak intensity threshold
        min_mito_prominence: Minimum mito peak prominence threshold
        max_mito_prominence: Maximum mito peak prominence threshold
        min_target_prominence: Minimum target peak prominence threshold
        max_target_prominence: Maximum target peak prominence threshold
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    original_count = len(df)
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Apply thresholds
    print("Applying thresholds:")
    if min_mito_intensity is not None:
        print(f"  Mito intensity >= {min_mito_intensity}")
        df = df[df['mito_peak_intensity'] >= min_mito_intensity]
    if max_mito_intensity is not None:
        print(f"  Mito intensity <= {max_mito_intensity}")
        df = df[df['mito_peak_intensity'] <= max_mito_intensity]
    if min_target_intensity is not None:
        print(f"  Target intensity >= {min_target_intensity}")
        df = df[df['target_peak_intensity'] >= min_target_intensity]
    if max_target_intensity is not None:
        print(f"  Target intensity <= {max_target_intensity}")
        df = df[df['target_peak_intensity'] <= max_target_intensity]
    if min_mito_prominence is not None:
        print(f"  Mito prominence >= {min_mito_prominence}")
        df = df[df['mito_peak_prominence'] >= min_mito_prominence]
    if max_mito_prominence is not None:
        print(f"  Mito prominence <= {max_mito_prominence}")
        df = df[df['mito_peak_prominence'] <= max_mito_prominence]
    if min_target_prominence is not None:
        print(f"  Target prominence >= {min_target_prominence}")
        df = df[df['target_peak_prominence'] >= min_target_prominence]
    if max_target_prominence is not None:
        print(f"  Target prominence <= {max_target_prominence}")
        df = df[df['target_peak_prominence'] <= max_target_prominence]
    
    filtered_count = len(df)
    print(f"\nRecords: {original_count} original -> {filtered_count} after filtering")
    
    if filtered_count == 0:
        print("No data remaining after applying thresholds!")
        return
    
    # Extract the specified number of characters from image_name as group
    df['group'] = df['image_name'].str[:group_by_chars]
    
    # Calculate the difference between mito_peak_distance and target_peak_distance
    df['peak_distance_diff'] = df['mito_peak_distance'] - df['target_peak_distance']
    
    print(f"Loaded {len(df)} records from {csv_file}")
    print(f"Groups: {df['group'].unique()}")
    print(f"\nSummary statistics:")
    print(df.groupby('group')['peak_distance_diff'].describe())
    
    # Identify and print outliers
    outliers = identify_outliers(df)
    if outliers:
        print(f"\n\nOUTLIERS DETECTED ({len(outliers)} total):")
        print("-" * 80)
        for outlier in outliers:
            print(f"Group: {outlier['group']:5s} | Image: {outlier['image_name']:40s} | "
                  f"Value: {outlier['peak_distance_diff']:8.2f} | "
                  f"Bounds: [{outlier['lower_bound']:.2f}, {outlier['upper_bound']:.2f}]")
        
        # Save outliers to CSV file
        outliers_df = pd.DataFrame(outliers)
        outliers_file = os.path.join(output_dir, 'peak_distance_outliers.csv')
        outliers_df.to_csv(outliers_file, index=False)
        print(f"\nOutliers saved to {outliers_file}")
    else:
        print("\nNo outliers detected.")

    
    # Create figure with box plot and violin plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    groups = sorted(df['group'].unique())
    data_to_plot = [df[df['group'] == group]['peak_distance_diff'].values for group in groups]
    
    # Left plot: Box plot
    bp = ax1.boxplot(data_to_plot, tick_labels=groups, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='OMM')
    ax1.set_xlabel('Image Group (First 5 characters)', fontsize=12)
    ax1.set_ylabel('Peak Distance Difference (pixels)', fontsize=12)
    ax1.set_title('Box Plot', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=11)
    
    # Right plot: Violin plot
    parts = ax2.violinplot(data_to_plot, positions=range(len(groups)), showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='OMM')
    ax2.set_xticks(range(len(groups)))
    ax2.set_xticklabels(groups)
    ax2.set_xlabel('Image Group (First 5 characters)', fontsize=12)
    ax2.set_ylabel('Peak Distance Difference (pixels)', fontsize=12)
    ax2.set_title('Violin Plot', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=11)
    
    plt.suptitle('Distribution of (Mito Peak Distance - Target Peak Distance) by Image Group', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'peak_distance_difference_whisker_plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved box plot and violin plot to {output_file}")
    plt.close()


@click.command()
@click.option('--csv-file', type=click.Path(exists=True), required=True, help='Path to scan_data.csv file')
@click.option('--output-directory', type=click.Path(), default=None, help='Output directory for the plot (defaults to same as CSV file)')
@click.option('--group-by-chars', type=int, default=5, help='Number of starting characters of image_name to use for grouping (default: 5)')
@click.option('--min-mito-intensity', type=float, default=0.3, help='Minimum mito peak intensity')
@click.option('--max-mito-intensity', type=float, default=None, help='Maximum mito peak intensity')
@click.option('--min-target-intensity', type=float, default=0.2, help='Minimum target peak intensity')
@click.option('--max-target-intensity', type=float, default=None, help='Maximum target peak intensity')
@click.option('--min-mito-prominence', type=float, default=0.08, help='Minimum mito peak prominence')
@click.option('--max-mito-prominence', type=float, default=None, help='Maximum mito peak prominence')
@click.option('--min-target-prominence', type=float, default=0.05, help='Minimum target peak prominence')
@click.option('--max-target-prominence', type=float, default=None, help='Maximum target peak prominence')
def main(csv_file, output_directory, group_by_chars, min_mito_intensity, max_mito_intensity, 
         min_target_intensity, max_target_intensity, min_mito_prominence, max_mito_prominence,
         min_target_prominence, max_target_prominence):
    """
    Analyze peak distance differences from scan_data.csv and create a violin plot.
    
    Example:
        python plot_peak_distance_analysis.py --csv-file data_masked_omm_analysis/scan_data.csv
        python plot_peak_distance_analysis.py --csv-file data_masked_omm_analysis/scan_data.csv --group-by-chars 3
        python plot_peak_distance_analysis.py --csv-file data_masked_omm_analysis/scan_data.csv --min-mito-intensity 0.1 --min-target-intensity 0.1
    """
    analyze_peak_distances(csv_file, output_directory, group_by_chars, min_mito_intensity, max_mito_intensity,
                          min_target_intensity, max_target_intensity, min_mito_prominence, max_mito_prominence,
                          min_target_prominence, max_target_prominence)
    print("\nDone!")


if __name__ == "__main__":
    main()
