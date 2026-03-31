# Mito Protein Scanner

Analyze mitochondrial networks and protein distribution in fluorescence microscopy images. This tool processes multi-channel TIFF images to identify mitochondrial cristae structure and quantify protein localization along the mitochondrial network.

## Features

- **Network Skeletonization**: Automatically extracts the mitochondrial network structure from binary images
- **Intensity Profiling**: Measures protein intensity along mitochondrial paths
- **Peak Detection**: Identifies peaks in protein distribution
- **Interactive Threshold Selection**: GUI-based threshold selection for precise segmentation
- **Batch Processing**: Process multiple TIFF images with consistent parameters
- **CSV Export**: Saves intensity profiles and visualizations for every mitochondrial segment

## Prerequisites

- Python 3.10 or later
- Conda (Miniconda or Anaconda)
- Multi-channel TIFF image files

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/GrotjahnLab/mito_linescan.git
cd mito_linescan
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
```

This creates a conda environment named `mito_protein_scanner` with all required dependencies:
- numpy, scipy, matplotlib, pandas
- scikit-image (for image processing)
- networkx (for graph analysis)
- click (for CLI)
- tifffile (TIFF I/O)
- sknw (skeleton network building)
- tqdm (progress bars)

### 3. Activate Environment

```bash
conda activate mito_protein_scanner
```

## Quick Start

### Basic Usage

```bash
python mito_protein_scanner.py
```

This runs with default parameters:
- Input directory: `20251021_decon_data/tiff`
- Input pattern: `snap*.tiff`
- Mitochondria channel: 0
- Protein channel: 2
- Interactive GUI enabled

### View Available Options

```bash
python mito_protein_scanner.py --help
```

## Usage Examples

### Process with Custom Input Directory

```bash
python mito_protein_scanner.py \
  --input-dir /path/to/images \
  --input-pattern "*.tiff"
```

### Use Different Channel Indices

If your TIFF files have different channel orders:

```bash
python mito_protein_scanner.py \
  --mito-channel 1 \
  --protein-channel 3
```

### Disable Interactive GUI (Batch Processing)

```bash
python mito_protein_scanner.py \
  --no-gui
```

### Custom Output Settings

```bash
python mito_protein_scanner.py \
  --mask-dir-output ./masks \
  --run-name my_experiment
```

### Adjust Scanning Parameters

```bash
python mito_protein_scanner.py \
  --scan-width 6 \
  --path-sampling 10 \
  --min-path-length 50
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | `20251021_decon_data/tiff` | Directory containing TIFF images |
| `--input-pattern` | `snap*.tiff` | Pattern to match input files |
| `--mask-dir-output` | `20251021_decon_data/tiff/masks` | Output directory for cell masks |
| `--mask-dir-input` | `20251021_decon_data/tiff/masks/` | Input directory for existing masks |
| `--run-name` | `run1` | Suffix for output directories |
| `--mito-channel` | `0` | 0-based index for mitochondria channel |
| `--protein-channel` | `2` | 0-based index for protein channel |
| `--use-gui` / `--no-gui` | `--use-gui` | Enable interactive GUI for threshold |
| `--scan-width` | `4` | Pixels on each side of path |
| `--path-sampling` | `5` | Subpixel samples along normal |
| `--min-path-length` | `30` | Minimum path length to process |

## Workflow

### 1. First Run with GUI

For the first analysis, use the GUI to verify settings:

```bash
python mito_protein_scanner.py \
  --input-dir ./my_data \
  --use-gui
```

### 2. Interactive Steps

When running with `--use-gui`:
- **Lasso Selection**: Draw around the cell of interest
- **Threshold Selection**: Adjust threshold to segment mitochondria
- **Confirm Threshold**: Review the skeleton and network visualization

### 3. Batch Processing

Once parameters are optimized, run without GUI:

```bash
python mito_protein_scanner.py \
  --input-dir ./my_data \
  --no-gui
```

## Output Files

For each image processed, the tool generates:

### Per-Mitochondrial Files
- **CSV**: `{basename}_mito_{id}.csv` - Intensity profiles with columns:
  - `Distance`: Position along mitochondrial path
  - `Mito_Intensity`: Mitochondria channel intensity
  - `Scan_Intensity`: Protein channel intensity

- **PNG**: `{basename}_mito_{id}_intensities.png` - Visualization showing:
  - Left: Mitochondria channel with path overlay
  - Middle: Protein channel with path overlay
  - Right: Intensity profiles and peak detection

### Masks
- **TIFF**: Cell masks saved to `mask_dir_output`

### Directory Structure
```
output_dir/
├── {image_name}run1/
│   ├── {image_name}_mito_1.csv
│   ├── {image_name}_mito_1_intensities.png
│   ├── {image_name}_mito_2.csv
│   ├── {image_name}_mito_2_intensities.png
│   └── ...
└── masks/
    ├── snap0001.tiff
    ├── snap0002.tiff
    └── ...
```

## Input Data Requirements

### TIFF Format
- Multi-channel stacked TIFF files
- Expected order: `[mito_channel, nuclear_channel, protein_channel, ...]`
- 16-bit or 8-bit images supported

### Channel Configuration
- **Mitochondria channel**: Usually MitoTracker or similar (binary/high contrast)
- **Protein channel**: The signal to analyze along mitochondrial paths

## Troubleshooting

### No images found
- Check that `--input-dir` exists and contains files matching `--input-pattern`
- Example: Use `--input-pattern "snap*.tiff"` for files named snap0001.tiff, etc.

### GUI not appearing
- Ensure X11 forwarding is enabled if using SSH
- Try `--no-gui` for non-interactive mode

### Memory issues with large images
- Process images with `--no-gui` flag
- Use `--scan-width` and `--path-sampling` parameters to adjust performance

### Channel mismatch
- Verify channel indices with `--mito-channel` and `--protein-channel`
- Use `--mito-channel 0 --protein-channel 1` etc. based on your TIFF structure

## Citations

This tool uses several open-source packages:
- scikit-image: Image processing
- NetworkX: Graph analysis
- sknw: Skeleton network analysis

## License

See LICENSE file in repository.

## Contact

For issues or questions, please open an issue on GitHub or contact the Grotjahn Lab.
