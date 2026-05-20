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

# NOTE: the workflow modules (bin.mito_mask, bin.mito_protein_line_scanner, ...)
# pull in the full scientific stack (numpy, scipy, scikit-image, matplotlib,
# sknw, pandas, tifffile, networkx, ...). Importing them at module load adds
# multiple seconds of startup cost to *every* invocation of this CLI, including
# `--help`. To keep the dispatcher snappy, each workflow is imported lazily
# inside its own click command, so only the workflow you actually run pays
# the import cost.


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
        # Skip None values (from null in YAML)
        if value is None:
            continue
        
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


# --- Default config template ----------------------------------------------
# The on-disk `config.yaml.template` at the project root is the single source
# of truth for the default config layout. We read it at module import time
# so `create-config` always reflects the latest options without anyone having
# to maintain a parallel hardcoded copy here. If the file is missing (e.g.
# when running from a pip-installed wheel that didn't bundle the template),
# fall back to a minimal stub that tells the user where to fetch it.
_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / 'config.yaml.template'
try:
    CONFIG_TEMPLATE = _TEMPLATE_PATH.read_text()
except OSError:
    CONFIG_TEMPLATE = (
        "# config.yaml.template was not bundled with this install.\n"
        "# Fetch it from the project repository:\n"
        "#   https://github.com/GrotjahnLab/mito_linescan/blob/main/config.yaml.template\n"
        "# and place it next to your config.yaml before re-running create-config.\n"
    )


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Path to config.yaml file')
@click.pass_context
def cli(ctx, config):
    """Mitochondrial analysis tool with multiple workflows."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config


@cli.command(name='create-config')
@click.option('--output', '-o', type=click.Path(), default='config.yaml',
              show_default=True,
              help='Where to write the default config file.')
@click.option('--force/--no-force', default=False,
              help='Overwrite the output file if it already exists.')
def create_config(output, force):
    """Create a default config file (a copy of config.yaml.template).

    Looks for config.yaml.template alongside the installed package first; if
    that's not found (e.g. running from a pip-installed wheel without the
    source tree), falls back to the embedded CONFIG_TEMPLATE string. Refuses
    to clobber an existing output file unless --force is passed.
    """
    template_path = Path(__file__).resolve().parent.parent / 'config.yaml.template'
    if template_path.exists():
        content = template_path.read_text()
        source = str(template_path)
    else:
        content = CONFIG_TEMPLATE
        source = 'embedded CONFIG_TEMPLATE (config.yaml.template not found on disk)'

    out_path = Path(output)
    if out_path.exists() and not force:
        raise click.ClickException(
            f"{out_path} already exists. Pass --force to overwrite."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    click.echo(f"✓ Wrote default config to {out_path}")
    click.echo(f"  (source: {source})")
    click.echo("Edit this file with your directory paths and parameters, "
               "then pass it to the workflow commands via --config.")


@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to config.yaml file')
@click.pass_context
def draw_mask(ctx, config):
    """Draw mitochondrial mask (mito_mask.py)."""
    config_file = config
    if not config_file:
        raise click.ClickException("--config option is required")
    
    config_data = load_config(config_file)
    if 'draw_mask' not in config_data:
        raise click.ClickException("'draw_mask' section not found in config.yaml")
    
    draw_mask_config = config_data['draw_mask']
    args = config_to_args(draw_mask_config)
    
    click.echo(f"Running draw_mask with config: {draw_mask_config}")

    # Lazy import: only load the heavy workflow when this command runs.
    from bin.mito_mask import main as mito_mask_main

    # Call the original mito_mask main function with args
    sys.argv = ['mito_mask'] + args
    mito_mask_main(standalone_mode=False)


@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to config.yaml file')
@click.pass_context
def refine_mask(ctx, config):
    """Refine mask edges (mito_mask_refine.py)."""
    config_file = config
    if not config_file:
        raise click.ClickException("--config option is required")
    
    config_data = load_config(config_file)
    if 'refine_mask' not in config_data:
        raise click.ClickException("'refine_mask' section not found in config.yaml")
    
    refine_mask_config = config_data['refine_mask']
    args = config_to_args(refine_mask_config)
    
    click.echo(f"Running refine_mask with config: {refine_mask_config}")

    # Lazy import: only load the heavy workflow when this command runs.
    from bin.mito_mask_refine import main as mito_mask_refine_main

    # Call the original mito_mask_refine main function with args
    sys.argv = ['mito_mask_refine'] + args
    mito_mask_refine_main(standalone_mode=False)


@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to config.yaml file')
@click.pass_context
def omm_normal_scan(ctx, config):
    """Scan protein on outer mitochondrial membrane using surface normals (mito_protein_omm_normal_scanner.py)."""
    config_file = config
    if not config_file:
        raise click.ClickException("--config option is required")
    
    config_data = load_config(config_file)
    if 'omm_normal_scan' not in config_data:
        raise click.ClickException("'omm_normal_scan' section not found in config.yaml")
    
    omm_normal_scan_config = config_data['omm_normal_scan']
    args = config_to_args(omm_normal_scan_config)
    
    click.echo(f"Running omm_normal_scan with config: {omm_normal_scan_config}")

    # Lazy import: only load the heavy workflow when this command runs.
    from bin.mito_protein_omm_normal_scanner import main as mito_protein_omm_normal_scanner_main

    # Call the original mito_protein_omm_normal_scanner main function with args
    sys.argv = ['mito_protein_omm_normal_scanner'] + args
    mito_protein_omm_normal_scanner_main(standalone_mode=False)


@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to config.yaml file')
@click.pass_context
def network_line_scan(ctx, config):
    """Scan protein along mitochondrial network (mito_protein_line_scanner.py)."""
    config_file = config
    if not config_file:
        raise click.ClickException("--config option is required")
    
    config_data = load_config(config_file)
    if 'network_line_scan' not in config_data:
        raise click.ClickException("'network_line_scan' section not found in config.yaml")
    
    network_line_scan_config = config_data['network_line_scan']
    args = config_to_args(network_line_scan_config)
    
    click.echo(f"Running network_line_scan with config: {network_line_scan_config}")

    # Lazy import: only load the heavy workflow when this command runs.
    from bin.mito_protein_line_scanner import main as mito_protein_line_scanner_main

    # Call the original mito_protein_line_scanner main function with args
    sys.argv = ['mito_protein_line_scanner'] + args
    mito_protein_line_scanner_main(standalone_mode=False)


@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to config.yaml file')
@click.pass_context
def analyze_omm_scans(ctx, config):
    """Analyze intensity profiles from OMM normal scan results (analyze_omm_scans.py)."""
    config_file = config
    if not config_file:
        raise click.ClickException("--config option is required")
    
    config_data = load_config(config_file)
    if 'analyze_omm_scans' not in config_data:
        raise click.ClickException("'analyze_omm_scans' section not found in config.yaml")
    
    analyze_config = config_data['analyze_omm_scans']
    input_directory = analyze_config.get('input_directory')
    output_directory = analyze_config.get('output_directory')
    peak_threshold = analyze_config.get('peak_threshold', 0.3)
    peak_prominence = analyze_config.get('peak_prominence', 0.1)
    
    if not input_directory:
        raise click.ClickException("'input_directory' parameter required in analyze_omm_scans section")
    
    click.echo(f"Running analyze_omm_scans with config: input_directory={input_directory}, output_directory={output_directory}, peak_threshold={peak_threshold}, peak_prominence={peak_prominence}")

    # Lazy import: only load the heavy workflow when this command runs.
    from bin.analyze_omm_scans import process_directory as analyze_omm_scans_main

    # Call the analyze_omm_scans function
    analyze_omm_scans_main(input_directory, output_directory, peak_threshold, peak_prominence)


@cli.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to config.yaml file')
@click.pass_context
def plot_peak_distance_analysis(ctx, config):
    """Plot peak distance analysis with outlier detection (plot_peak_distance_analysis.py)."""
    config_file = config
    if not config_file:
        raise click.ClickException("--config option is required")
    
    config_data = load_config(config_file)
    if 'plot_peak_distance_analysis' not in config_data:
        raise click.ClickException("'plot_peak_distance_analysis' section not found in config.yaml")
    
    plot_config = config_data['plot_peak_distance_analysis']
    args = config_to_args(plot_config)
    
    click.echo(f"Running plot_peak_distance_analysis with config: {plot_config}")

    # Lazy import: only load the heavy workflow when this command runs.
    from bin.plot_peak_distance_analysis import main as plot_peak_distance_analysis_main

    # Call the plot_peak_distance_analysis main function with args
    sys.argv = ['plot_peak_distance_analysis'] + args
    plot_peak_distance_analysis_main(standalone_mode=False)


@cli.command(name='peak-spacing-histogram')
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Path to config.yaml file')
@click.pass_context
def peak_spacing_histogram(ctx, config):
    """Pool every line-scan CSV, find peaks on Scan_Intensity, and histogram
    the distance between consecutive peaks (plot_peak_spacing_histogram.py)."""
    config_file = config
    if not config_file:
        raise click.ClickException("--config option is required")

    config_data = load_config(config_file)
    if 'peak_spacing_histogram' not in config_data:
        raise click.ClickException(
            "'peak_spacing_histogram' section not found in config.yaml"
        )

    sh_config = config_data['peak_spacing_histogram']
    args = config_to_args(sh_config)

    click.echo(f"Running peak_spacing_histogram with config: {sh_config}")

    # Lazy import: only load the heavy workflow when this command runs.
    from bin.plot_peak_spacing_histogram import main as plot_peak_spacing_histogram_main

    sys.argv = ['plot_peak_spacing_histogram'] + args
    plot_peak_spacing_histogram_main(standalone_mode=False)


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
