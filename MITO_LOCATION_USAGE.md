# Mito Protein Localization - Unified Mitochondrial Analysis Tool

## Overview

`mito_protein_localization` is a unified command-line tool that consolidates multiple mitochondrial analysis workflows. Instead of running separate scripts, you can execute different analysis modes through one command using a YAML configuration file.

## Supported Workflows

1. **draw_mask** - Interactive mask drawing for mitochondrial structures
2. **refine_mask** - Refine mask edges using intensity information  
3. **omm_normal_scan** - Scan protein on outer mitochondrial membrane using surface normals
4. **network_line_scan** - Scan protein distribution along mitochondrial network

## Installation

```bash
cd /scratch/mito_linescan
micromamba env create -f environment.yml
micromamba activate mito_protein_scanner
```

## Quick Start

### 1. Create a config file

Copy the template and edit with your parameters:

```bash
cp config.yaml.template config.yaml
# Edit config.yaml with your paths and parameters
```

### 2. Run a workflow

```bash
mito_protein_localization --config config.yaml draw_mask
mito_protein_localization --config config.yaml refine_mask
mito_protein_localization --config config.yaml omm_normal_scan
mito_protein_localization --config config.yaml network_line_scan
```

## Configuration File Format

The `config.yaml` file has sections for each workflow. Each section contains the parameters for that workflow.

### Example config.yaml

```yaml
draw_mask:
  input_directory: '/path/to/input/directory'
  manual_mask_directory: '/path/to/output/directory'
  mito_channel: 0
  target_channel: 1
  scan_width: 7
  sampling_radius: 3

refine_mask:
  input_directory: '/path/to/input/directory'
  refined_mask_directory: '/path/to/output/directory'
  mask_channel: 0
  mito_channel: 1
  target_channel: 2

omm_normal_scan:
  input_directory: '/path/to/input/directory'
  output_directory: '/path/to/output/directory'
  mito_channel: 1
  scan_channel: 0
  mask_channel: 2
  scan_width: 7
  sampling_radius: 3
  mito_thickness_threshold: 1

network_line_scan:
  input_dir: '/path/to/input/directory'
  input_pattern: 'snap*.tiff'
  mask_dir_output: '/path/to/output/masks'
  mask_dir_input: '/path/to/input/masks/'
  run_name: 'run1'
  mito_channel: 0
  protein_channel: 2
  use_gui: true
  scan_width: 4
  path_sampling: 5
  min_path_length: 30
```

## Parameter Details

### draw_mask
- `input_directory`: Input directory containing TIFF images
- `manual_mask_directory`: Output directory for manually drawn masks (optional, defaults to input directory)
- `mito_channel`: Channel index for mitochondria (0-based)
- `target_channel`: Channel index for target/scan signal (0-based)
- `scan_width`: Width of scan lines in pixels
- `sampling_radius`: Radius for weighted average sampling in pixels

### refine_mask
- `input_directory`: Input directory with mask images
- `refined_mask_directory`: Output directory
- `mask_channel`: Channel index for mask (0-based)
- `mito_channel`: Channel index for mitochondria (0-based)
- `target_channel`: Channel index for target/scan signal (0-based)

### omm_normal_scan
- `input_directory`: Input directory with TIFF images
- `output_directory`: Output directory
- `mito_channel`: Mitochondria channel index (0-based)
- `scan_channel`: Scan/protein channel index (0-based)
- `mask_channel`: Mask channel index (0-based)
- `scan_width`: Width of scan lines in pixels
- `sampling_radius`: Radius for weighted average sampling
- `mito_thickness_threshold`: Initial erosion value (1-20)

### network_line_scan
- `input_dir`: Input directory containing TIFF images
- `input_pattern`: Pattern to match TIFF files (e.g., "snap*.tiff")
- `mask_dir_output`: Output directory for masks
- `mask_dir_input`: Input directory for existing masks
- `run_name`: Run name suffix for output directories
- `mito_channel`: Mitochondria channel index (0-based)
- `protein_channel`: Protein channel index (0-based)
- `use_gui`: Use interactive GUI for threshold selection (true/false)
- `scan_width`: Pixels on each side of path for scanning
- `path_sampling`: Number of subpixel samples along normal
- `min_path_length`: Minimum path length to process

## Individual Scripts

You can still call the individual scripts directly if needed:

```bash
mito_mask --i /path/to/input --o /path/to/output
mito_mask_refine --i /path/to/input --o /path/to/output
mito_protein_omm_localization --i /path/to/input --o /path/to/output
mito_protein_line_scanner --input-dir /path/to/input
```

## Notes

- All paths in the config file should be absolute or properly formatted
- Boolean values (`use_gui`) should be `true` or `false` (lowercase)
- For parameters with hyphens (like `input-dir`), use underscores in the config file
