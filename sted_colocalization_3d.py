import numpy as np
import tifffile
import mrcfile
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
import matplotlib.patches as mpatches
import csv
import click

@click.command()
@click.option('--input', 'input_dir', type=click.Path(exists=True), required=True, 
              help='Input directory containing TIF files')
@click.option('--output', 'output_dir', type=click.Path(), required=True, 
              help='Output directory for results')
@click.option('--mtdna-threshold', 'mtdna_threshold', type=float, default=99, 
              help='mtDNA threshold percentile (default: 99)')
@click.option('--mtdna-dilation', 'mtdna_dilation', type=int, default=3, 
              help='mtDNA dilation iterations (default: 3)')
def main(input_dir, output_dir, mtdna_threshold, mtdna_dilation):
    """
    Analyze 3D STED colocalization data with customizable parameters.
    
    Example:
        python sted_colocalization_3d.py --input ./mtDNA_SEPT9_3DSTED --output ./results --mtdna-threshold 99 --mtdna-dilation 3
    """
    # Convert to Path objects
    input_directory = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all TIF files
    tif_files = list(input_directory.glob("*.tif")) + list(input_directory.glob("*.tiff"))
    tif_files.sort()

    print(f"Found {len(tif_files)} TIF files in {input_directory}")
    print(f"mtDNA threshold: {mtdna_threshold}th percentile")
    print(f"mtDNA dilation iterations: {mtdna_dilation}")
    print(f"Output directory: {output_path}")

    # Results storage
    results = []

    # Process each TIF file
    for tif_file in tif_files:
        print(f"\n{'='*80}")
        print(f"Processing: {tif_file.name}")
        print(f"{'='*80}")
        
        # Read the TIF file
        print(f"Reading TIF file: {tif_file}")
        with tifffile.TiffFile(tif_file) as tif:
            # Get image data
            data = tif.asarray()
            print(f"Image shape: {data.shape}")
            print(f"Image dtype: {data.dtype}")

        # Assuming the structure is (Z, Channels, Height, Width) or (Z*Channels, Height, Width)
        # We need to determine the actual structure
        if len(data.shape) == 4:
            # Already in (Z, Channels, Height, Width) format
            z_slices, num_channels, height, width = data.shape
            print(f"Data format: (Z={z_slices}, Channels={num_channels}, Height={height}, Width={width})")
            
            # Separate channels
            channel_data = [data[:, ch, :, :] for ch in range(num_channels)]

        elif len(data.shape) == 3:
            # Likely (Z*Channels, Height, Width) - need to reshape
            total_slices, height, width = data.shape
            
            # Assuming 3 channels and equal distribution
            num_channels = 3
            if total_slices % num_channels != 0:
                raise ValueError(f"Total slices ({total_slices}) not divisible by {num_channels} channels")
            
            z_slices = total_slices // num_channels
            print(f"Data format: ({total_slices} images, Height={height}, Width={width})")
            print(f"Reshaping to: (Z={z_slices}, Channels={num_channels}, Height={height}, Width={width})")
            
            # Reshape to separate channels
            data_reshaped = data.reshape(z_slices, num_channels, height, width)
            channel_data = [data_reshaped[:, ch, :, :] for ch in range(num_channels)]

        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # Extract individual channels
        mtdna_ch = channel_data[0]  # Channel 0: mtDNA
        mito_ch = channel_data[1]   # Channel 1: mito
        septin_ch = channel_data[2] # Channel 2: septin

        print(f"\nChannel shapes (before thresholding):")
        print(f"  mtDNA: {mtdna_ch.shape}")
        print(f"  mito: {mito_ch.shape}")
        print(f"  septin: {septin_ch.shape}")

        # Threshold each channel by its average intensity (keep only brighter parts)
        print(f"\nThresholding channels by average intensity:")

        mtdna_avg = np.mean(mtdna_ch)
        mtdna_mask_thresh = mtdna_ch > mtdna_avg
        mtdna_ch_thresholded = mtdna_ch.copy()
        mtdna_ch_thresholded[~mtdna_mask_thresh] = 0
        print(f"  mtDNA: avg={mtdna_avg:.2f}, {np.sum(mtdna_mask_thresh)} voxels above threshold")

        mito_avg = np.mean(mito_ch)
        mito_mask_thresh = mito_ch > mito_avg
        mito_ch_thresholded = mito_ch.copy()
        mito_ch_thresholded[~mito_mask_thresh] = 0
        print(f"  mito: avg={mito_avg:.2f}, {np.sum(mito_mask_thresh)} voxels above threshold")

        septin_avg = np.mean(septin_ch)
        septin_mask_thresh = septin_ch > septin_avg
        septin_ch_thresholded = septin_ch.copy()
        septin_ch_thresholded[~septin_mask_thresh] = 0
        print(f"  septin: avg={septin_avg:.2f}, {np.sum(septin_mask_thresh)} voxels above threshold")

        # Use thresholded channels for analysis
        mtdna_ch = mtdna_ch_thresholded
        mito_ch = mito_ch_thresholded
        septin_ch = septin_ch_thresholded

        print(f"\nChannel shapes (after thresholding):")
        print(f"  mtDNA: {mtdna_ch.shape}, non-zero voxels: {np.sum(mtdna_ch > 0)}")
        print(f"  mito: {mito_ch.shape}, non-zero voxels: {np.sum(mito_ch > 0)}")
        print(f"  septin: {septin_ch.shape}, non-zero voxels: {np.sum(septin_ch > 0)}")

        # Save each channel as MRC file
        channel_names = ["septin", "mito", "mtDNA"]

        for ch_idx, (ch_name, ch_data) in enumerate(zip(channel_names, channel_data)):
            output_file = output_path / f"{tif_file.stem}_{ch_name}.mrc"
            
            print(f"\nSaving {ch_name} to: {output_file}")
            print(f"  Shape: {ch_data.shape}")
            print(f"  Data type: {ch_data.dtype}")
            print(f"  Min/Max values: {ch_data.min()}/{ch_data.max()}")
            
            with mrcfile.new(str(output_file), overwrite=True) as mrc:
                mrc.set_data(ch_data.astype(np.float32))

        print("\n" + "="*60)
        print("ANALYZING REGIONAL SEPTIN DENSITY (3D)")
        print("="*60)

        # Work directly with 3D data (no max projections)
        # Threshold to create 3D binary masks
        # Adjust thresholds based on 3D intensity distribution
        mito_threshold = np.percentile(mito_ch[mito_ch > 0], 30) if np.sum(mito_ch > 0) > 0 else 0
        mito_mask_3d = mito_ch > mito_threshold
        # Dilate mito mask to ensure we capture nearby regions
        mito_mask_3d = ndimage.binary_dilation(mito_mask_3d, iterations=3)

        # Threshold mtDNA using the provided parameter
        mtdna_threshold_val = np.percentile(mtdna_ch[mtdna_ch > 0], mtdna_threshold) if np.sum(mtdna_ch > 0) > 0 else 0
        septin_threshold = np.percentile(septin_ch[septin_ch > 0], 95) if np.sum(septin_ch > 0) > 0 else 0

        mtdna_mask_3d = mtdna_ch > mtdna_threshold_val
        septin_mask_3d = septin_ch > septin_threshold

        print(f"mtDNA threshold ({mtdna_threshold}th percentile, masked by mito): {mtdna_threshold_val:.1f}")
        print(f"mito threshold (30th percentile): {mito_threshold:.1f}")
        print(f"septin threshold (95th percentile): {septin_threshold:.1f}")
        print(f"mtDNA mask voxels: {np.sum(mtdna_mask_3d)}")
        print(f"mito mask voxels: {np.sum(mito_mask_3d)}")
        print(f"septin mask voxels: {np.sum(septin_mask_3d)}")

        # Dilate mtDNA binary mask using the provided dilation parameter to create Area 1 (3D dilation)
        area1_mask_3d = ndimage.binary_dilation(mtdna_mask_3d, iterations=mtdna_dilation) * mito_mask_3d  # Ensure we only consider areas within mito

        print(f"Area 1 mask (dilated mtDNA) voxels: {np.sum(area1_mask_3d)}")

        # Area 2: Inside mito but NOT in area1 (more than 1 voxel from mtDNA)
        area2_mask = mito_mask_3d & (~area1_mask_3d)

        print(f"\nArea 1 (within 1 voxel of mtDNA): {np.sum(area1_mask_3d)} voxels")
        print(f"Area 2 (mito, >1 voxel from mtDNA): {np.sum(area2_mask)} voxels")

        # Calculate average septin density in each area
        septin_in_area1 = septin_ch[area1_mask_3d]
        septin_in_area2 = septin_ch[area2_mask]
        septin_in_mito = septin_ch[mito_mask_3d]

        # Calculate septin outside mito
        septin_outside_mito = septin_ch[~mito_mask_3d]

        avg_septin_area1 = np.mean(septin_in_area1) if len(septin_in_area1) > 0 else 0
        avg_septin_area2 = np.mean(septin_in_area2) if len(septin_in_area2) > 0 else 0
        avg_septin_mito = np.mean(septin_in_mito) if len(septin_in_mito) > 0 else 0
        avg_septin_outside_mito = np.mean(septin_outside_mito) if len(septin_outside_mito) > 0 else 0

        print(f"\nAverage septin intensity in Area 1: {avg_septin_area1:.2f}")
        print(f"Average septin intensity in Area 2: {avg_septin_area2:.2f}")
        print(f"Average septin intensity in mito (entire): {avg_septin_mito:.2f}")
        print(f"Average septin intensity outside mito: {avg_septin_outside_mito:.2f}")
        print(f"Ratio (Area1/Area2): {avg_septin_area1/avg_septin_area2:.2f}" if avg_septin_area2 > 0 else "N/A")

        # Store results
        results.append({
            'filename': tif_file.name,
            'z_slices': z_slices,
            'height': height,
            'width': width,
            'area1_voxels': np.sum(area1_mask_3d),
            'area2_voxels': np.sum(area2_mask),
            'mito_voxels': np.sum(mito_mask_3d),
            'outside_mito_voxels': np.sum(~mito_mask_3d),
            'avg_septin_area1': avg_septin_area1,
            'avg_septin_area2': avg_septin_area2,
            'avg_septin_mito': avg_septin_mito,
            'avg_septin_outside_mito': avg_septin_outside_mito,
            'ratio_area1_area2': avg_septin_area1/avg_septin_area2 if avg_septin_area2 > 0 else np.nan
        })

        # Create visualization
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Get central slice
        central_z = z_slices // 2
        print(f"Central slice: z={central_z}")

        # Plot 1: mtDNA channel with area1 overlay
        ax = axes[0, 0]
        mtdna_slice = mtdna_ch[central_z]
        im = ax.imshow(mtdna_slice, cmap='Blues', alpha=0.8)
        area1_overlay = ax.contour(area1_mask_3d[central_z], colors='red', linewidths=2, levels=[0.5])
        ax.set_title(f'mtDNA (Channel 2) - Central Slice z={central_z}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Intensity')
        ax.contourf(area1_mask_3d[central_z].astype(float), levels=[0.5, 1.5], colors=['red'], alpha=0.2)
        patch_area1 = mpatches.Patch(color='red', alpha=0.2, label='Area 1 (≤1 voxel from mtDNA)')
        ax.legend(handles=[patch_area1], loc='upper right')

        # Plot 2: mtDNA binary mask (final)
        ax = axes[0, 1]
        im = ax.imshow(mtdna_mask_3d[central_z].astype(int), cmap='Blues', alpha=0.8)
        ax.set_title(f'mtDNA Binary Mask (Final) - Central Slice z={central_z}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Binary')

        # Plot 3: mito channel
        ax = axes[0, 2]
        mito_slice = mito_ch[central_z]
        im = ax.imshow(mito_slice, cmap='Greens', alpha=0.8)
        ax.set_title(f'mito (Channel 1) - Central Slice z={central_z}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Intensity')

        # Plot 4: mito binary mask
        ax = axes[1, 0]
        mito_binary = mito_mask_3d[central_z].astype(int)
        im = ax.imshow(mito_binary, cmap='Greens', alpha=0.8)
        ax.set_title(f'mito Binary Mask - Central Slice z={central_z}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Binary')

        # Plot 5: septin channel with area1 and area2 shading
        ax = axes[1, 1]
        septin_slice = septin_ch[central_z]
        im = ax.imshow(septin_slice, cmap='Reds', alpha=0.8)
        ax.contourf(area1_mask_3d[central_z].astype(float), levels=[0.5, 1.5], colors=['cyan'], alpha=0.3, label='Area 1 (≤1 voxel from mtDNA)')
        ax.contourf(area2_mask[central_z].astype(float), levels=[0.5, 1.5], colors=['green'], alpha=0.3, label='Area 2 (mito, >1 voxel from mtDNA)')
        ax.set_title(f'septin (Channel 0) with Regions - Central Slice z={central_z}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Intensity')
        patch1 = mpatches.Patch(color='cyan', alpha=0.3, label='Area 1 (≤1 voxel from mtDNA)')
        patch2 = mpatches.Patch(color='green', alpha=0.3, label='Area 2 (mito, >1 voxel)')
        ax.legend(handles=[patch1, patch2], loc='upper right')

        # Plot 6: Area 2 mask with septin image
        ax = axes[1, 2]
        septin_slice = septin_ch[central_z]
        im = ax.imshow(septin_slice, cmap='Reds', alpha=0.8)
        ax.contourf(area2_mask[central_z].astype(float), levels=[0.5, 1.5], colors=['green'], alpha=0.3)
        ax.set_title(f'Area 2 + septin (Channel 0) - Central Slice z={central_z}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Intensity')
        patch_area2 = mpatches.Patch(color='green', alpha=0.3, label='Area 2 (mito, >1 voxel from mtDNA)')
        ax.legend(handles=[patch_area2], loc='upper right')

        plt.tight_layout()
        output_png = output_path / f"{tif_file.stem}_analysis.png"
        plt.savefig(str(output_png), dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_png}")
        plt.close()

        # Create a separate figure for histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.linspace(0, max(mtdna_ch.max(), mito_ch.max(), septin_ch.max()), 50)
        ax.hist(mtdna_ch.flatten(), bins=bins, alpha=0.5, label='mtDNA (Channel 2)', color='blue', edgecolor='blue')
        ax.hist(mito_ch.flatten(), bins=bins, alpha=0.5, label='mito (Channel 1)', color='green', edgecolor='green')
        ax.hist(septin_ch.flatten(), bins=bins, alpha=0.5, label='septin (Channel 0)', color='red', edgecolor='red')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title('Intensity Distribution - All Channels (Thresholded)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_histogram = output_path / f"{tif_file.stem}_histogram.png"
        plt.savefig(str(output_histogram), dpi=150, bbox_inches='tight')
        print(f"Saved histogram to: {output_histogram}")
        plt.close()

        print("\n" + "="*60)
        print("FILE COMPLETE")
        print("="*60)

    # Write results to CSV
    print("\n" + "="*80)
    print("WRITING RESULTS TO CSV")
    print("="*80)

    csv_file = output_path / "analysis_results.csv"

    if results:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to: {csv_file}")
    else:
        print("No results to save!")

    # Create box plots for septin ratios
    if results:
        print("\n" + "="*80)
        print("CREATING BOX PLOTS FOR SEPTIN RATIOS")
        print("="*80)
        
        # Calculate ratios for each image
        ratios_area1_mito = []
        ratios_area2_mito = []
        
        for result in results:
            avg_septin_mito = result['avg_septin_mito']
            if avg_septin_mito > 0:
                ratio_area1_mito = result['avg_septin_area1'] / avg_septin_mito
                ratio_area2_mito = result['avg_septin_area2'] / avg_septin_mito
                ratios_area1_mito.append(ratio_area1_mito)
                ratios_area2_mito.append(ratio_area2_mito)
        
        if ratios_area1_mito and ratios_area2_mito:
            # Create box plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            box_data = [ratios_area1_mito, ratios_area2_mito]
            bp = ax.boxplot(box_data, labels=['Area 1/mito', 'Area 2/mito'], patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Septin Intensity Ratio', fontsize=12)
            ax.set_title('Distribution of Septin Intensity Ratios Across All Images', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            output_boxplot = output_path / f"septin_ratios_boxplot_{mtdna_threshold}_{mtdna_dilation}.png"
            plt.savefig(str(output_boxplot), dpi=150, bbox_inches='tight')
            print(f"Saved box plot to: {output_boxplot}")
            print(f"\nBox plot statistics:")
            print(f"  Area 1/mito - Mean: {np.mean(ratios_area1_mito):.3f}, Median: {np.median(ratios_area1_mito):.3f}, Std: {np.std(ratios_area1_mito):.3f}")
            print(f"  Area 2/mito - Mean: {np.mean(ratios_area2_mito):.3f}, Median: {np.median(ratios_area2_mito):.3f}, Std: {np.std(ratios_area2_mito):.3f}")
            plt.close()
        else:
            print("Not enough data to create box plot")

    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE")
    print("="*80)
    print(f"Processed {len(tif_files)} TIF files")
    print(f"Results CSV: {csv_file}")


if __name__ == '__main__':
    main()

