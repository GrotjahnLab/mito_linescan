#!/usr/bin/env python3

"""
Standalone utility: given a TIFF containing a binary mask, compute the
local thickness of the mask and save both as a 2-channel TIFF.

Output convention (ImageJ hyperstack axes):
    channel 0 : the input mask (float32, 0.0 outside, 1.0 inside)
    channel 1 : the local thickness (float32, in voxels; 0.0 outside the mask)

The thickness is computed via the `localthickness` PyPI package, matching
the logic used by `network_line_scan`'s thickness filter — so if a mask
looks reasonable there, the numbers here will be consistent.

The script prints mean / median / max thickness computed over the MASK
voxels only (never the whole volume, so background zeros don't drag the
average down).

Not wired into the `mito_protein_localization` dispatcher — this is a
one-shot side utility, invoked directly. Example:

    mask_thickness --input examples/SEPT9_REPRESENTATIVE/manual_masks/7_OV5_i11_mito_mask.tif

or via the module:

    python -m bin.mask_thickness_util --input <path.tif>

Input handling: the utility auto-detects channel vs volumetric axes for
common shapes:
  - 2D  (H, W)          : treated as the mask directly (no channel axis)
  - 3D  (C, H, W)       : channels leading, `--channel` picks the mask
                          (heuristic: leading dim ≤ 4 -> channels)
  - 3D  (Z, H, W)       : volumetric mask (all Z slices)
                          (heuristic: leading dim > 4 -> Z stack)
  - 4D  (C, Z, H, W)    : `--channel` picks the mask along axis 0
  - 4D  (Z, C, H, W)    : `--channel` picks the mask along axis 1
                          (heuristic used when axis 1 is smaller)

Pass `--channel-axis` to override the heuristic if the guess is wrong.
"""

import os
import sys

import click
import numpy as np
import tifffile as tf


# -------- local thickness helper -----------------------------------------

def _local_thickness(binary):
    """Local thickness map of a 2D or 3D binary array (in voxel units).

    Uses the `localthickness` PyPI package with the same fallback logic as
    `network_line_scan`. Raises `click.ClickException` (rather than
    returning None) so this side-utility fails loudly if the package isn't
    installed — you can always add the required dependency without hunting
    through logs.
    """
    try:
        import localthickness as lt
    except ImportError as exc:
        raise click.ClickException(
            f"The `localthickness` PyPI package is required "
            f"({exc}). Install with `pip install localthickness` "
            f"or add it to your conda env."
        )

    bn = np.ascontiguousarray(binary.astype(np.uint8))
    try:
        thk = lt.local_thickness(bn)
    except AttributeError:
        try:
            thk = lt.local_thickness_2d(bn)
        except Exception as exc:
            raise click.ClickException(
                f"localthickness call failed on shape {bn.shape}: {exc}"
            )
    except Exception as exc:
        # Some versions of localthickness only support 2D; for a 3D mask
        # they raise on the 3D call. Fall back to slice-by-slice.
        if bn.ndim == 3:
            click.echo(
                f"  [thickness] 3D call raised ({exc!s}); falling back to "
                f"per-slice 2D thickness. Note: this measures per-slice "
                f"thickness, not true 3D thickness."
            )
            try:
                slices = [lt.local_thickness(bn[z]) for z in range(bn.shape[0])]
            except AttributeError:
                slices = [lt.local_thickness_2d(bn[z]) for z in range(bn.shape[0])]
            thk = np.stack(slices, axis=0)
        else:
            raise click.ClickException(
                f"localthickness call failed on shape {bn.shape}: {exc}"
            )

    thk = np.asarray(thk, dtype=np.float32)
    if thk.shape != bn.shape:
        raise click.ClickException(
            f"localthickness returned shape {thk.shape}, "
            f"expected {bn.shape}."
        )
    return thk


# -------- input shape handling -------------------------------------------

def _extract_mask(arr, channel, channel_axis):
    """Pick the mask plane out of the input array.

    Returns (mask, was_volumetric) where:
      - `mask` is a 2D (H, W) or 3D (Z, H, W) array containing the mask
      - `was_volumetric` says whether the mask spans multiple Z slices,
        used to decide the output axes ('CYX' vs 'ZCYX').

    `channel_axis` overrides the automatic detection; pass None for auto.
    """
    if arr.ndim == 2:
        return arr, False
    if arr.ndim == 3:
        # Ambiguous: could be (C, H, W) or (Z, H, W).
        auto_axis = None
        if channel_axis is not None:
            auto_axis = channel_axis
        else:
            auto_axis = 0 if arr.shape[0] <= 4 else None
        if auto_axis is None:
            # Volumetric mask, no channel axis. `channel` is ignored.
            return arr, True
        return np.take(arr, channel, axis=auto_axis), False
    if arr.ndim == 4:
        if channel_axis is not None:
            axis = channel_axis
        elif arr.shape[0] <= 4:
            axis = 0
        elif arr.shape[1] <= 4:
            axis = 1
        else:
            raise click.ClickException(
                f"Ambiguous 4D shape {arr.shape}: can't tell which axis "
                f"is channels. Pass --channel-axis explicitly (0..3)."
            )
        sliced = np.take(arr, channel, axis=axis)
        # After taking a channel, the array is 3D. If the remaining leading
        # axis is Z (i.e. we sliced axis 0 or 1 on a 4D array), it's
        # volumetric; if it was already 2D with a channel dim we wouldn't
        # be in the 4D branch. So this is always volumetric here.
        return sliced, True
    raise click.ClickException(
        f"Unsupported input dimensionality: shape {arr.shape}"
    )


# -------- CLI ------------------------------------------------------------

@click.command()
@click.option('--input', 'input_path', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Input TIFF path. Can be 2D (H,W), 3D volumetric (Z,H,W), '
                   '3D channelled (C,H,W), or 4D (C,Z,H,W)/(Z,C,H,W).')
@click.option('--output', 'output_path', default='', type=str,
              help='Output TIFF path. Defaults to '
                   '{input-stem}_with_thickness.tif in the same directory.')
@click.option('--channel', default=0, type=int, show_default=True,
              help='Index into the channel axis; ignored for 2D or 3D '
                   'volumetric input (no channel axis).')
@click.option('--channel-axis', default=None, type=int,
              help='Force which axis is channels (0 for 3D shapes like '
                   '(C,H,W); 0 or 1 for 4D shapes). Default: auto-detect '
                   'by picking the smaller-index axis whose size is ≤ 4.')
def main(input_path, output_path, channel, channel_axis):
    """Add a thickness channel to a binary mask TIFF."""
    arr = tf.imread(input_path)
    click.echo(f'input : {input_path}')
    click.echo(f'         shape={arr.shape}, dtype={arr.dtype}, '
               f'min={arr.min()}, max={arr.max()}')

    mask, was_volumetric = _extract_mask(arr, channel, channel_axis)
    binary = (mask > 0)
    click.echo(f'mask  : shape={binary.shape}, {int(binary.sum())} voxels '
               f'({100.0 * binary.mean():.2f}% of extracted plane)')

    thk = _local_thickness(binary)

    # Statistics computed over the mask voxels ONLY — background zeros
    # would dominate any all-volume mean and hide the real distribution.
    if binary.any():
        vals = thk[binary]
        mean_thk = float(vals.mean())
        med_thk = float(np.median(vals))
        max_thk = float(vals.max())
        std_thk = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    else:
        mean_thk = med_thk = max_thk = std_thk = 0.0

    click.echo(f'thickness (mask voxels only, in voxels):')
    click.echo(f'  mean   = {mean_thk:.3f}')
    click.echo(f'  median = {med_thk:.3f}')
    click.echo(f'  std    = {std_thk:.3f}')
    click.echo(f'  max    = {max_thk:.3f}')

    # Assemble a 2-channel ImageJ hyperstack. For a 2D mask that's (C, Y, X);
    # for a 3D volumetric mask that's (Z, C, Y, X) — the order ImageJ /
    # FIJI expects so the channel toggle in the ImageJ toolbar works.
    mask_f = binary.astype(np.float32)
    if was_volumetric:
        combined = np.stack([mask_f, thk], axis=1)  # (Z, 2, Y, X)
        axes = 'ZCYX'
    else:
        combined = np.stack([mask_f, thk], axis=0)  # (2, Y, X)
        axes = 'CYX'

    if not output_path:
        stem = os.path.splitext(input_path)[0]
        # Handle .ome.tif safely (splitext only strips the last extension)
        if stem.lower().endswith('.ome'):
            stem = stem[:-len('.ome')]
        output_path = f"{stem}_with_thickness.tif"

    tf.imwrite(output_path, combined, imagej=True,
               metadata={'axes': axes})
    click.echo(f'wrote : {output_path}')
    click.echo(f'         shape={combined.shape}, axes={axes}, '
               f'dtype={combined.dtype}')
    click.echo(f'         channel 0 = mask, channel 1 = thickness')


if __name__ == '__main__':
    main()
