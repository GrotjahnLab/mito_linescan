# Mito Protein Scanner

Analyze mitochondrial networks and protein distribution in fluorescence microscopy images. This unified tool provides multiple workflows to process multi-channel TIFF images, identify mitochondrial structures, and quantify protein localization along the mitochondrial network.

## Features

- **Multiple Analysis Workflows**: Five specialized workflows for different analysis needs
- **Network Skeletonization**: Automatically extracts the mitochondrial network structure from binary images
- **Interactive Mask Drawing**: GUI-based tool for manually creating mitochondrial masks
- **Mask Refinement**: Refine mask edges using intensity information for improved accuracy
- **Outer Mitochondrial Membrane (OMM) Scanning**: Specialized scanning along the outer membrane using surface normals
- **Network Line Scanning**: Protein intensity profiling along mitochondrial network paths, built on a ridge-filter mask pipeline with an interactive tuning GUI and an optional local-thickness post-skeletonization filter
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
- localthickness (optional local-thickness filter in `network_line_scan`)

### 3. Activate Environment

```bash
conda activate mito_protein_scanner
```

## Configuration-Based Workflow

The recommended way to use this tool is through the YAML configuration file, which enables reproducible analysis with all parameters in one place.

### 1. Set Up Config File

Generate a default `config.yaml` with the bundled `create-config` command:

```bash
mito_protein_localization create-config                 # writes ./config.yaml
mito_protein_localization create-config --output ./my_config.yaml
mito_protein_localization create-config --force         # overwrite an existing file
```

`create-config` copies `config.yaml.template` byte-for-byte and refuses to
clobber an existing file unless `--force` is passed. If you prefer to do it
by hand, you can still:

```bash
cp config.yaml.template config.yaml
```

Either way, edit `config.yaml` and configure the workflows you want to use.
Each workflow section contains the parameters for that analysis.

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
Scan protein distribution along the mitochondrial network. The binary mask is
built by a multi-scale ridge-filter pipeline (white top-hat → Meijering ridge
filter → Otsu) and can be tuned either through an interactive GUI or via
config/CLI parameters for batch processing. An optional local-thickness filter
prunes the skeleton after skeletonization so it does not perturb the network
topology.

```yaml
network_line_scan:
  input_dir: '/path/to/input'
  input_pattern: 'snap*.tiff'              # Pattern to match files
  mask_dir_output: '/path/to/masks'        # Output mask directory
  mask_dir_input: '/path/to/masks/'        # Input mask directory
  run_name: 'run1'                         # Run name suffix for outputs
  mito_channel: 0                          # Mitochondria channel
  protein_channel: 2                       # Protein channel
  use_gui: true                            # Interactive mask GUI on/off
  scan_width: 4                            # Pixels on each side of path
  path_sampling: 5                         # Subpixel samples along normal
  min_path_length: 30                      # Minimum path length to process
  # --- Ridge-filter mask pipeline (also live-tunable in the GUI) ---
  tubule_radius: 2.0                       # Tubule radius (px); drives
                                           # top-hat disk + ridge sigmas
  sensitivity: 1.0                         # Multiplier on Otsu cut of ridge
                                           # response (1.0=Otsu, <1=more
                                           # permissive, >1=stricter)
  min_object_size: 30                      # Drop small CCs (px)
  gap_closing: 1                           # Binary-closing disk radius (px)
                                           # to bridge 1-2 px breaks
  # --- Local-thickness filter (applied AFTER skeletonization) ---
  use_thickness_filter: false              # Toggle local-thickness filtering
  min_thickness: 1.0                       # Keep skeleton px with thickness >=
  max_thickness: 20.0                      # Keep skeleton px with thickness <=
  # --- Final binary mask export (reusable downstream) ---
  binary_mask_dir_output: ''               # If set, save the final (post-
                                           # thickness-filter) mito binary as
                                           # {basename}_mito_binary.tif
                                           # (uint8 0/255). Empty = skip.
```

##### Interactive mask GUI

When `use_gui: true`, an interactive figure opens with live overlays and
controls:

- **Tubule radius (px)** — sets the structural scale of the ridge filter
  (top-hat disk and Meijering sigmas). Recomputation is slow, so it updates
  only when you press the **Recompute ridge** button. For ~0.17 µm/px decon
  data with ~0.3–0.6 µm mito tubules, ~2 px is a good starting point.
- **Sensitivity** — multiplier on the Otsu cut of the ridge response.
  `1.0` = pure Otsu; `<1` catches dim tubules; `>1` is stricter.
- **Min size (px)** — drops binary connected components smaller than this.
- **Gap closing (px)** — binary-closing disk radius to bridge small breaks
  before skeletonization.
- **Min / Max thickness (px)** — together define the allowed local-thickness
  range. When `Filter ON` is checked, skeleton pixels whose underlying local
  thickness is outside `[min, max]` are dropped from the returned skeleton.
  Crucially, this exclusion is applied **after** `skeletonize`, so the
  topology of the network is determined by the full binary; thickness only
  removes pixels from the final skeleton and rebuilds the network graph.

The view toggles (Binary / Skeleton / Nodes / Ridge / Excluded / Filter ON)
let you overlay any combination of intermediates, and a stats line below the
figure reports binary %, skeleton pixels (filtered), nodes/edges, paths
≥ min_path_length, current threshold, and thickness distribution.

##### Local-thickness dependency

The optional thickness filter uses the
[`localthickness`](https://pypi.org/project/localthickness/) package, which is
installed as part of `environment.yml`. If the package is missing for any
reason, the filter is silently disabled (a warning is printed) and the
pipeline falls back to the un-filtered skeleton.

#### **analyze_omm_scans** - Peak Analysis
Analyze intensity profiles and detect peaks from OMM scanning results.

```yaml
analyze_omm_scans:
  input_directory: '/path/to/pickle/files/'
  output_directory: '/path/to/output/'     # Optional, defaults to input
  peak_threshold: 0.3                      # Peak threshold (0.0-1.0)
  peak_prominence: 0.1                     # Peak prominence threshold (0.0-1.0)
```

#### **plot_peak_distance_analysis** - Consecutive Peak Distance Histogram
Pool every per-mito CSV produced by `network_line_scan`, detect peaks on the
`Scan_Intensity` column (the protein / target channel) with intensity +
prominence thresholds, and write a histogram of the **consecutive**
peak-to-peak distances. Outputs go into `output_directory` (defaults to
`csv_directory`):

- `peak_distance_histogram.png` — pooled histogram with median + mean overlays
- `peak_distances.csv` — one row per consecutive pair: `image_name`, `mito_id`, `distance`
- `peak_distance_per_track.csv` — per-track summary (n_peaks, path_length, mean/median distance)
- `peak_distance_summary.txt` — global stats

```yaml
plot_peak_distance_analysis:
  csv_directory: '/path/to/line_scan/output_dir/'  # Where the per-mito CSVs live
  input_pattern: '*_mito_*.csv'                    # Glob for the CSV files
  output_directory: ''                             # Default: same as csv_directory
  peak_min_intensity: 0.3                          # Min Scan_Intensity (0-1)
  peak_min_prominence: 0.1                         # Min prominence (0-1)
  peak_prominence_wlen: 10                         # peak_prominences wlen
  bin_width: 5.0                                   # Histogram bin width (px)
  max_distance: 0.0                                # X-axis cap; 0 = auto (data p99)
  recursive: false                                 # Recurse into subdirectories
```

Distances are reported in whatever units the CSV's `Distance` column is in
(pixels, by default).

#### **sted_colocalization_3d** - 3D STED mtDNA / mito / septin Analysis
Full 3D analysis pipeline for multi-channel STED stacks (default channel
ordering `0=mtDNA, 1=mito, 2=septin`). Per image, builds 3D binary masks from
percentile thresholds, computes the original Area1 (within K voxels of
mtDNA, inside mito) vs Area2 (mito, away from mtDNA) septin intensity
analysis, and on top of that produces a **per-nucleoid 3D radial intensity
profile** in all three channels using spherical shells out to a configurable
voxel radius. mtDNA "nucleoids" are connected components of the 3D mtDNA
binary mask, filtered by a minimum voxel count.

Per-image outputs (in `{output_dir}/{basename}{run_name}/`):
- `{basename}_{septin|mito|mtDNA}.mrc` — single-channel float32 MRC exports
- `{basename}_analysis.png` — 2×3 central-slice visualization
- `{basename}_histogram.png` — per-channel intensity histogram (log y)
- `{basename}_radial_profiles.csv` — long-format `image_name, nucleoid_id, z,
  y, x, on_mito, distance, mtdna_intensity, mito_intensity, septin_intensity`
- `{basename}_radial_profiles.png` — per-image 3-panel mean ± SEM plot
  (mtDNA / mito / septin) computed from this image's nucleoids only; same
  visual style as the pooled plot below for easy comparison
- `per_nucleoid/{basename}_{run_name}_nucleoid_{id:03d}_z{z}_y{y}_x{x}.png` — 2×3 plot per
  detected nucleoid. Top row: radial profile in mtDNA / mito / septin.
  Bottom row: black-background true-color crops at the centroid's Z slice —
  left = mtDNA+septin, middle = mtDNA alone, right = septin alone. Each
  bottom panel has the scan-time mito mask drawn as a **lime contour**
  ("mito outline"), a yellow `+` at the centroid, and a dashed yellow
  circle at the scan radius. Subdirectory keeps the per-image dir tidy for
  high-nucleoid images. Disable with `save_per_nucleoid_png: false`.

Pooled outputs (in `{output_dir}/`):
- `analysis_results.csv` — per-image Area1/Area2 voxel counts + ratios
- `septin_ratios_boxplot_<thr>_<dilation>.png` — Area1/mito vs Area2/mito
- `puncta_radial_profiles_pooled.csv` — every per-image radial CSV pooled
- `puncta_radial_profiles.png` — three-panel mean ± SEM plot (mtDNA / mito /
  septin)

```yaml
sted_colocalization_3d:
  input_dir: '/path/to/3d/sted/data/'
  input_pattern: '*.tif'
  output_dir: ''                       # Default: same as input_dir
  run_name: 'run1'
  mtdna_channel: 0
  mito_channel: 1
  protein_channel: 2
  mito_threshold_percentile: 30        # Percentile of nonzero mito voxels
  mito_dilation: 3                     # Binary-dilation iterations
  mtdna_threshold_percentile: 99
  mtdna_dilation: 3                    # Area1 = dilated mtDNA ∩ mito
  septin_threshold_percentile: 95      # For stats reporting
  punct_scan_radius: 20                # Radial profile radius (voxels)
  min_nucleoid_voxels: 5               # Drop tiny mtDNA components
  # Scan-time mito mask: only voxels INSIDE this mask are averaged into the
  # radial profile (so intensity vs distance reflects mito interior only).
  radial_scan_mito_threshold_percentile: 99
  radial_scan_mito_dilation: 3
  save_channel_mrcs: true
  save_analysis_png: true
  save_histogram_png: true
  save_per_nucleoid_png: true          # One PNG per detected nucleoid
```

Distances in the radial CSV are in voxels and assume isotropic spacing. For
anisotropic acquisitions, scale the `distance` column by the appropriate
factor offline.

### 3. Run Workflows

Execute any workflow using the unified command with the config file:

```bash
mito_protein_localization --config config.yaml draw_mask
mito_protein_localization --config config.yaml refine_mask
mito_protein_localization --config config.yaml omm_normal_scan
mito_protein_localization --config config.yaml network_line_scan
mito_protein_localization --config config.yaml analyze_omm_scans
mito_protein_localization --config config.yaml plot_peak_distance_analysis
mito_protein_localization --config config.yaml sted_colocalization_3d
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

binary_mask_dir_output/           # (optional, only if configured)
├── {image_name}_mito_binary.tif  # Final mito binary mask (uint8, 0/255),
│                                 # post-thickness-filter; ready to be reused
│                                 # as input to other workflows.
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
