# Mito Protein Scanner

Analyze mitochondrial networks and protein distribution in fluorescence microscopy images. This unified tool provides multiple workflows to process multi-channel TIFF images, identify mitochondrial structures, and quantify protein localization along the mitochondrial network.

## Features

- **Multiple Analysis Workflows**: Five specialized workflows for different analysis needs
- **Network Skeletonization**: Automatically extracts the mitochondrial network structure from binary images
- **Interactive Mask Drawing**: GUI-based tool for manually creating mitochondrial masks
- **Mask Refinement**: Refine mask edges using intensity information for improved accuracy
- **Outer Mitochondrial Membrane (OMM) Scanning**: Specialized scanning along the outer membrane using surface normals
- **Network Line Scanning**: Protein intensity profiling along mitochondrial network paths
- **Peak Detection & Analysis**: Identifies and analyzes peaks in protein distribution
- **Configuration-Driven**: Use YAML config files for reproducible, batch processing workflows
- **Batch Processing**: Process multiple TIFF images with consistent parameters
- **CSV Export**: Saves intensity profiles, visualizations, and analysis results

## Prerequisites

- Python 3.10 or later
- Conda (Miniconda or Anaconda, or micromamba)
- Multi-channel TIFF image files
- 2-4 channels typical (mitochondria, scan signal, optional mask, etc.)

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

Or with micromamba:

```bash
micromamba env create -f environment.yml
```

This creates a conda environment named `mito_protein_scanner` with all required dependencies:
- numpy, scipy, matplotlib, pandas, scikit-image, networkx
- click (for CLI), tifffile (TIFF I/O), sknw (skeleton network building), tqdm (progress bars)

### 3. Activate Environment

```bash
conda activate mito_protein_scanner
```

## Configuration-Based Workflow

The recommended way to use this tool is through the YAML configuration file, which enables reproducible analysis with all parameters in one place.

### 1. Set Up Config File

Copy the template and edit with your parameters:

```bash
cp config.yaml.template config.yaml
```

Edit `config.yaml` and configure the workflows you want to use. Each workflow section contains the parameters for that analysis.

### 2. Available Workflows

#### **draw_mask** - Interactive Mask Drawing
Manually draw masks for mitochondrial structures with interactive GUI.

```yaml
draw_mask:
  input_directory: '/path/to/input/directory'
  manual_mask_directory: '/path/to/output/directory'
  mito_channel: 0          # Channel index for mitochondria (0-based)
  target_channel: 1        # Channel index for target/scan signal
  scan_width: 7            # Width of scan lines in pixels
  sampling_radius: 3       # Radius for weighted average sampling
```

#### **refine_mask** - Mask Edge Refinement
Refine mask edges using intensity information for improved accuracy.

```yaml
refine_mask:
  input_directory: '/path/to/input/directory'
  refined_mask_directory: '/path/to/output/directory'
  mask_channel: 0          # Channel index for mask
  mito_channel: 1          # Channel index for mitochondria
  target_channel: 2        # Channel index for target/scan signal
```

#### **omm_normal_scan** - Outer Mitochondrial Membrane Scanning
Scan protein distribution on the outer mitochondrial membrane using surface normals.

```yaml
omm_normal_scan:
  input_directory: '/path/to/images/'
  output_directory: '/path/to/output/'
  mito_channel: 1                    # Mitochondria channel
  scan_channel: 0                    # Scan/protein channel
  mask_channel: 2                    # Mask channel
  scan_width: 7                      # Width of scan lines
  sampling_radius: 3                 # Radius for weighted sampling
  mito_thickness_threshold: 1        # Initial erosion value (1-20)
```

#### **network_line_scan** - Mitochondrial Network Analysis
Scan protein distribution along the mitochondrial network with optional interactive GUI for threshold selection.

```yaml
network_line_scan:
  input_dir: '/path/to/input'
  input_pattern: 'snap*.tiff'              # Pattern to match files
  mask_dir_output: '/path/to/masks'        # Output mask directory
  mask_dir_input: '/path/to/masks/'        # Input mask directory
  run_name: 'run1'                         # Run name suffix for outputs
  mito_channel: 0                          # Mitochondria channel
  protein_channel: 2                       # Protein channel
  use_gui: true                            # Interactive threshold selection
  scan_width: 4                            # Pixels on each side of path
  path_sampling: 5                         # Subpixel samples along normal
  min_path_length: 30                      # Minimum path length to process
```

#### **analyze_omm_scans** - Peak Analysis
Analyze intensity profiles and detect peaks from OMM scanning results.

```yaml
analyze_omm_scans:
  input_directory: '/path/to/pickle/files/'
  output_directory: '/path/to/output/'     # Optional, defaults to input
  peak_threshold: 0.3                      # Peak threshold (0.0-1.0)
  peak_prominence: 0.1                     # Peak prominence threshold (0.0-1.0)
```

### 3. Run Workflows

Execute any workflow using the unified command with the config file:

```bash
mito_protein_localization --config config.yaml draw_mask
mito_protein_localization --config config.yaml refine_mask
mito_protein_localization --config config.yaml omm_normal_scan
mito_protein_localization --config config.yaml network_line_scan
mito_protein_localization --config config.yaml analyze_omm_scans
```

Or run all workflows in sequence:

```bash
mito_protein_localization --config config.yaml draw_mask && \
mito_protein_localization --config config.yaml refine_mask && \
mito_protein_localization --config config.yaml omm_normal_scan && \
mito_protein_localization --config config.yaml analyze_omm_scans
```

## Usage Examples

### Example 1: Basic Network Line Scanning with GUI

Set up your config.yaml:

```yaml
network_line_scan:
  input_dir: './my_data/tiff'
  input_pattern: '*.tiff'
  mask_dir_output: './my_data/masks'
  mito_channel: 0
  protein_channel: 2
  use_gui: true
  scan_width: 4
```

Then run:

```bash
mito_protein_localization --config config.yaml network_line_scan
```

### Example 2: Batch OMM Scanning Without GUI

```yaml
omm_normal_scan:
  input_directory: '/data/2024_experiments/'
  output_directory: '/data/2024_experiments/omm_results/'
  mito_channel: 0
  scan_channel: 1
  mask_channel: 2
  mito_thickness_threshold: 2
```

Run the analysis:

```bash
mito_protein_localization --config config.yaml omm_normal_scan
```

### Example 3: Complete Analysis Pipeline

Create a comprehensive config.yaml with all workflows and run them in sequence:

```bash
mito_protein_localization --config config.yaml draw_mask && \
mito_protein_localization --config config.yaml omm_normal_scan && \
mito_protein_localization --config config.yaml analyze_omm_scans
```

## Direct Script Usage (Advanced)

For advanced use cases, you can still call individual scripts directly:

```bash
python bin/mito_mask.py --i /path/to/input --o /path/to/output
python bin/mito_mask_refine.py --i /path/to/input --o /path/to/output
python bin/mito_protein_omm_normal_scanner.py --i /path/to/input --o /path/to/output
python bin/mito_protein_line_scanner.py --input-dir /path/to/input
```

View help for individual scripts:

```bash
python bin/mito_protein_line_scanner.py --help
python bin/mito_protein_omm_normal_scanner.py --help
```

## Configuration File Reference

The `config.yaml` file uses YAML syntax. Key points:

- **Indentation**: Use 2 spaces (not tabs)
- **Channel indices**: 0-based (0, 1, 2, ...)
- **Boolean values**: Use `true` or `false` (lowercase)
- **Paths**: Use absolute paths or relative paths from where you run the command
- **Underscores**: Use underscores in parameter names (not hyphens) in config files

### Configuration Tips

1. **Multiple workflows in one file**: You can define all five workflows in a single config.yaml
2. **Selective execution**: Only the workflow you specify on the command line will run
3. **Template**: Always start with `config.yaml.template` as your reference
4. **Paths**: Ensure all input directories exist and contain valid TIFF files
5. **Channel order**: Verify your TIFF file channel order before setting indices

## Output Files

Output structure depends on which workflow you run:

### network_line_scan Outputs
```
{run_name}_output/
├── {image_name}_mito_1.csv       # Intensity profile along mitochondrial path
├── {image_name}_mito_1_intensities.png  # Visualization with path overlay
├── {image_name}_mito_2.csv
├── {image_name}_mito_2_intensities.png
└── ...
```

CSV file columns:
- `Distance`: Position along mitochondrial path
- `Mito_Intensity`: Mitochondria channel intensity (normalized 0-1)
- `Scan_Intensity`: Protein channel intensity (normalized 0-1)

### omm_normal_scan Outputs
```
output_directory/
├── pickle_files/
│   ├── {image_name}_profile.pkl  # Serialized intensity profiles
│   └── ...
└── visualizations/
    ├── {image_name}_profile.png
    └── ...
```

### draw_mask / refine_mask Outputs
```
manual_mask_directory/ or refined_mask_directory/
├── mask_{image_name}.tiff
├── mask_{image_name}.png
└── ...
```

### analyze_omm_scans Outputs
```
output_directory/
├── peaks_summary.csv     # Summary of detected peaks
├── peak_plots/           # Peak visualization plots
│   ├── {image_name}_peaks.png
│   └── ...
└── intensity_analysis/
    └── ...
```

## Input Data Requirements

### TIFF Format
- **Multi-channel stacked TIFF files** with intensity data as 16-bit or 8-bit
- Channels should be ordered consistently across all images
- Example: Channel 0 = mitochondria (MitoTracker), Channel 1 = protein signal

### Typical Channel Configuration
- **Channel 0**: Mitochondria marker (MitoTracker, TOM20 staining, etc.)
- **Channel 1**: Optional intermediate channel
- **Channel 2**: Protein of interest (your scan signal)
- **Additional channels**: Optional additional markers

### File Organization
```
data/
├── image_001.tiff
├── image_002.tiff
├── image_003.tiff
└── ...
```

## Troubleshooting

### Command not found: mito_protein_localization
- Ensure the environment is activated: `conda activate mito_protein_scanner`
- Verify installation completed successfully
- Try: `python -m bin.mito_protein_localization --help`

### YAML parsing errors
- Verify YAML indentation uses 2 spaces (not tabs)
- Check that paths in config are valid
- Boolean values should be `true` or `false` (lowercase)

### No images found
- Verify `input_directory` exists and is spelled correctly
- Check that TIFF files match the `input_pattern`
- Example: For pattern `snap*.tiff`, files should be named `snap0001.tiff`, etc.

### Memory issues with large images
- **For network_line_scan**: Set `use_gui: false` for batch processing
- Reduce `scan_width` and `path_sampling` values
- Process fewer files at a time

### GUI not appearing (Linux/SSH)
- Ensure X11 forwarding is enabled: `ssh -X user@host`
- For headless systems, set `use_gui: false` in config
- Try: `mito_protein_localization --config config.yaml network_line_scan`

### Channel mismatch errors
- Verify channel indices match your TIFF file structure
- Use 0-based indexing (first channel = 0, second = 1, etc.)
- Check TIFF metadata with: `python -c "from tifffile import TiffFile; t = TiffFile('yourfile.tiff'); print(len(t.series[0].pages))"`

## Performance Considerations

- **scan_width**: Larger values = more computation. Start with 4, increase if needed
- **path_sampling**: Controls subpixel accuracy. Use 5 for speed, 10+ for precision
- **min_path_length**: Filters out small/noise segments. Increase to skip isolated structures
- **Peak detection**: Adjust `peak_threshold` and `peak_prominence` to filter false positives

## Supported File Formats

- **Input**: TIFF files (`.tiff`, `.tif`)
- **Output**: CSV, PNG, pickle
- **Config**: YAML (`.yaml`, `.yml`)

## Related Documentation

- [CLI_USAGE.md](CLI_USAGE.md) - Direct command-line script usage
- [MITO_LOCATION_USAGE.md](MITO_LOCATION_USAGE.md) - Config-based workflow documentation
- `config.yaml.template` - Complete configuration file template with explanations
