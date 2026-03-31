# Mito Protein Scanner - Command Line Usage

The mito_protein_scanner.py script has been converted to use Click for command-line execution.

## Basic Usage

Run the script with default parameters:
```bash
python mito_protein_scanner.py
```

View help and available options:
```bash
python mito_protein_scanner.py --help
```

## Command-Line Options

### Main Parameters

- `--input-dir` (TEXT): Input directory containing TIFF images  
  Default: `20251021_decon_data/tiff`

- `--input-pattern` (TEXT): Pattern to match input TIFF files  
  Default: `snap*.tiff`

- `--mask-dir-output` (PATH): Output directory for masks  
  Default: `20251021_decon_data/tiff/masks`

- `--mask-dir-input` (PATH): Input directory for existing masks  
  Default: `20251021_decon_data/tiff/masks/`

- `--run-name` (TEXT): Run name suffix for output directories  
  Default: `run1`

### Channel Selection

- `--mito-channel` (INTEGER): 0-based index for mitochondria channel  
  Default: `0`

- `--protein-channel` (INTEGER): 0-based index for protein channel  
  Default: `2`

### Processing Options

- `--use-gui / --no-gui`: Use interactive GUI for threshold selection  
  Default: `--use-gui`

- `--scan-width` (INTEGER): Pixels on each side of the path for scanning  
  Default: `4`

- `--path-sampling` (INTEGER): Number of subpixel samples along the normal  
  Default: `5`

- `--min-path-length` (INTEGER): Minimum path length to process  
  Default: `30`

## Examples

### Process with custom input directory and disable GUI
```bash
python mito_protein_scanner.py \
  --input-dir /path/to/my/tiff/files \
  --input-pattern "*.tiff" \
  --no-gui
```

### Process with custom channel indices and output directory
```bash
python mito_protein_scanner.py \
  --input-dir ./data \
  --mito-channel 1 \
  --protein-channel 3 \
  --mask-dir-output ./output/masks \
  --run-name my_experiment
```

### Use different scanning parameters
```bash
python mito_protein_scanner.py \
  --scan-width 6 \
  --path-sampling 10 \
  --min-path-length 50
```

### Full custom example
```bash
python mito_protein_scanner.py \
  --input-dir ./2025_data \
  --input-pattern "snap*.tiff" \
  --mask-dir-output ./2025_data/masks_output \
  --mask-dir-input ./2025_data/masks_input \
  --run-name experiment_1 \
  --mito-channel 0 \
  --protein-channel 2 \
  --use-gui \
  --scan-width 4 \
  --path-sampling 5 \
  --min-path-length 30
```

## Creating a Script to Run with Specific Parameters

Create a file called `run_analysis.sh`:
```bash
#!/bin/bash

python mito_protein_scanner.py \
  --input-dir ./data/20251021_decon \
  --mito-channel 0 \
  --protein-channel 2 \
  --scan-width 4 \
  --run-name final_run
```

Then run it:
```bash
chmod +x run_analysis.sh
./run_analysis.sh
```
