#!/usr/bin/env python3

"""
Unified mitochondrial analysis tool with multiple workflows.
Supports draw_mask, refine_mask, omm_normal_scan, and network_line_scan operations via config.yaml
"""

import os
import sys
import click
import yaml
from pathlib import Path

# Import the main functions from individual modules
from bin.mito_mask import main as mito_mask_main
from bin.mito_mask_refine import main as mito_mask_refine_main
from bin.mito_protein_omm_normal_scanner import main as mito_protein_omm_normal_scanner_main
from bin.mito_protein_line_scanner import main as mito_protein_line_scanner_main


def load_config(config_file):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def config_to_args(config_dict):
    """Convert config dictionary to click command line arguments."""
    args = []
    for key, value in config_dict.items():
        # Convert underscores to hyphens for click options
        key = key.replace('_', '-')
        
        # Handle boolean flags
        if isinstance(value, bool):
            if value:
                args.append(f'--{key}')
            else:
                args.append(f'--no-{key}')
        else:
            args.append(f'--{key}')
            args.append(str(value))
    
    return args


CONFIG_TEMPLATE = """# Mito Location Configuration File
# Configure the parameters for different mitochondrial analysis workflows

# Draw mask: Interactive mask drawing for mitochondrial structures
draw_mask:
  input_directory: '/path/to/input/directory'  # Input directory with TIFF images
  manual_mask_directory: '/path/to/output/directory'  # Output directory for manually drawn masks (optional, defaults to input directory)

# Mask refine: Refine mask edges using intensity information
refine_mask:
  input_directory: '/path/to/input/directory'  # Input directory with mask images
  refined_mask_directory: '/path/to/output/directory'  # Output directory for refined masks
  mask_channel: 0  # Channel index for mask (0-based)
  mito_channel: 1  # Channel index for mitochondria (0-based)
  target_channel: 2  # Channel index for target/scan signal (0-based)

# OMM Normal Scan: Scan protein on outer mitochondrial membrane using surface normals
omm_normal_scan:
  input_directory: '/path/to/images/'  # Input directory with TIFF images
  output_directory: '/path/to/output/'  # Output directory
  mito_channel: 1  # Mitochondria channel index (0-based)
  scan_channel: 0  # Scan/protein channel index (0-based)
  mask_channel: 2  # Mask channel index (0-based)
  scan_width: 7  # Width of scan lines in pixels
  sampling_radius: 3  # Radius for weighted average sampling in pixels
  mito_thickness_threshold: 1  # Initial erosion value for mask (1-20)

# Line scan: Scan protein distribution along mitochondrial network
network_line_scan:
  input_dir: '20251021_decon_data/tiff'  # Input directory containing TIFF images
  input_pattern: 'snap*.tiff'  # Pattern to match input TIFF files
  mask_dir_output: '20251021_decon_data/tiff/masks'  # Output directory for masks
  mask_dir_input: '20251021_decon_data/tiff/masks/'  # Input directory for existing masks
  run_name: 'run1'  # Run name suffix for output directories
  mito_channel: 0  # 0-based index for mitochondria channel
  protein_channel: 2  # 0-based index for protein channel
  use_gui: true  # Use interactive GUI for threshold selection
  scan_width: 4  # Pixels on each side of the path for scanning
  path_sampling: 5  # Number of subpixel samples along the normal
  min_path_length: 30  # Minimum path length to process
"""


def create_config_file(output_file):
    """Create a template config file at the specified output path.
    
    Args:
        output_file: Path where the config template file should be written
        
    Returns:
        str: Path to the created config file
    """
    output_path = Path(output_file)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the template to the file
    with open(output_path, 'w') as f:
        f.write(CONFIG_TEMPLATE)
    
    return str(output_path)


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Path to config.yaml file')
@click.pass_context
def cli(ctx, config):
    """Mitochondrial analysis tool with multiple workflows."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config


@cli.command()
@click.option('--output', type=click.Path(), required=True, help='Output path for config file')
def create_config(output):
    """Create a template config file."""
    try:
        config_path = create_config_file(output)
        click.echo(f"✓ Config template created at: {config_path}")
        click.echo("Edit this file with your directory paths and parameters, then use it with other commands")
    except Exception as e:
        raise click.ClickException(f"Failed to create config file: {e}")


@cli.command()
@click.pass_context
def draw_mask(ctx):
    """Draw mitochondrial mask (mito_mask.py)."""
    config_file = ctx.obj['config']
    if not config_file:
        raise click.ClickException("--config option is required")
    
    config = load_config(config_file)
    if 'draw_mask' not in config:
        raise click.ClickException("'draw_mask' section not found in config.yaml")
    
    draw_mask_config = config['draw_mask']
    args = config_to_args(draw_mask_config)
    
    click.echo(f"Running draw_mask with config: {draw_mask_config}")
    
    # Call the original mito_mask main function with args
    sys.argv = ['mito_mask'] + args
    mito_mask_main(standalone_mode=False)


@cli.command()
@click.pass_context
def refine_mask(ctx):
    """Refine mask edges (mito_mask_refine.py)."""
    config_file = ctx.obj['config']
    if not config_file:
        raise click.ClickException("--config option is required")
    
    config = load_config(config_file)
    if 'refine_mask' not in config:
        raise click.ClickException("'refine_mask' section not found in config.yaml")
    
    refine_mask_config = config['refine_mask']
    args = config_to_args(refine_mask_config)
    
    click.echo(f"Running refine_mask with config: {refine_mask_config}")
    
    # Call the original mito_mask_refine main function with args
    sys.argv = ['mito_mask_refine'] + args
    mito_mask_refine_main(standalone_mode=False)


@cli.command()
@click.pass_context
def omm_normal_scan(ctx):
    """Scan protein on outer mitochondrial membrane using surface normals (mito_protein_omm_normal_scanner.py)."""
    config_file = ctx.obj['config']
    if not config_file:
        raise click.ClickException("--config option is required")
    
    config = load_config(config_file)
    if 'omm_normal_scan' not in config:
        raise click.ClickException("'omm_normal_scan' section not found in config.yaml")
    
    omm_normal_scan_config = config['omm_normal_scan']
    args = config_to_args(omm_normal_scan_config)
    
    click.echo(f"Running omm_normal_scan with config: {omm_normal_scan_config}")
    
    # Call the original mito_protein_omm_normal_scanner main function with args
    sys.argv = ['mito_protein_omm_normal_scanner'] + args
    mito_protein_omm_normal_scanner_main(standalone_mode=False)


@cli.command()
@click.pass_context
def network_line_scan(ctx):
    """Scan protein along mitochondrial network (mito_protein_line_scanner.py)."""
    config_file = ctx.obj['config']
    if not config_file:
        raise click.ClickException("--config option is required")
    
    config = load_config(config_file)
    if 'network_line_scan' not in config:
        raise click.ClickException("'network_line_scan' section not found in config.yaml")
    
    network_line_scan_config = config['network_line_scan']
    args = config_to_args(network_line_scan_config)
    
    click.echo(f"Running network_line_scan with config: {network_line_scan_config}")
    
    # Call the original mito_protein_line_scanner main function with args
    sys.argv = ['mito_protein_line_scanner'] + args
    mito_protein_line_scanner_main(standalone_mode=False)


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
