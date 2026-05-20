#!/usr/bin/env python3
"""
Preview a ridge-filter-based binary mask for the mitochondrial network.

Designed as a side-by-side comparison against the current global-threshold
approach used in mito_protein_line_scanner.py. Run this under the
`mito_protein_scanner` conda env.

Usage:
    python bin/preview_mito_mask.py \
        --input  /path/to/file.tif \
        --output /path/to/previews/  \
        --mito-channel 0

Pipeline (per image):
    1. white top-hat (disk radius ~ 2x tubule radius) flattens background
    2. multiscale Meijering ridge filter highlights tubular structures
    3. Otsu on positive ridge values -> binary mask
    4. small-object removal + skeletonize for the network graph
"""
import os
import click
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from skimage.filters import threshold_otsu, meijering
from skimage.morphology import (
    white_tophat, disk, remove_small_objects, remove_small_holes,
    skeletonize, binary_closing,
)


def mono_cmap(rgb):
    return LinearSegmentedColormap.from_list("m", [(0, 0, 0), rgb])


def proposed_mask(mito, tophat_radius=8, sigmas=(1.0, 1.5, 2.0, 2.5),
                  min_object=30, close_radius=1):
    """Return (binary, ridge_response, skeleton)."""
    mito = mito.astype(np.float32)
    # 1. normalize to [0, 1] for the filters to behave consistently
    m = mito - np.percentile(mito, 1)
    m /= max(np.percentile(m, 99.5), 1e-9)
    m = np.clip(m, 0, 1)
    # 2. white top-hat -> flatten background
    th = white_tophat(m, disk(tophat_radius))
    # 3. Meijering ridge filter at multiple scales
    ridge = meijering(th, sigmas=list(sigmas), black_ridges=False)
    # 4. Otsu on positive ridge values
    pos = ridge[ridge > 0]
    t = threshold_otsu(pos) if pos.size else 0.0
    binary = ridge > t
    # 5. cleanup: close small gaps, drop small specks, fill pinholes
    if close_radius > 0:
        binary = binary_closing(binary, disk(close_radius))
    binary = remove_small_objects(binary, min_size=min_object)
    binary = remove_small_holes(binary, area_threshold=min_object)
    # 6. skeleton
    skel = skeletonize(binary)
    return binary, ridge, skel


def simple_mask(mito, min_object=30):
    """Current-style: Otsu on raw intensity."""
    t = threshold_otsu(mito)
    binary = mito > t
    binary = remove_small_objects(binary, min_size=min_object)
    skel = skeletonize(binary)
    return binary, t, skel


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Input TIFF file or directory")
@click.option("--output", "output_dir", required=True, type=click.Path(),
              help="Output directory for preview PNGs")
@click.option("--mito-channel", default=0, show_default=True,
              help="Mito channel index (0-based)")
@click.option("--tophat-radius", default=8, show_default=True,
              help="White top-hat disk radius (px); ~ 2x tubule radius")
@click.option("--sigmas", default="1.0,1.5,2.0,2.5", show_default=True,
              help="Comma-separated scales for the ridge filter (px)")
@click.option("--min-object", default=30, show_default=True,
              help="Minimum connected-component size to keep (px)")
def main(input_path, output_dir, mito_channel, tophat_radius, sigmas, min_object):
    os.makedirs(output_dir, exist_ok=True)
    sigmas = tuple(float(s) for s in sigmas.split(","))

    if os.path.isdir(input_path):
        files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith((".tif", ".tiff"))
        )
    else:
        files = [input_path]

    for f in files:
        base = os.path.basename(f)
        base = base[: base.rfind(".tif")] if ".tif" in base else base
        img = tf.imread(f).astype(np.float32)
        if img.ndim == 2:
            mito = img
        else:
            mito = img[mito_channel]
        print(f"[{base}] mito shape={mito.shape} dtype={mito.dtype}")

        binary, ridge, skel = proposed_mask(
            mito, tophat_radius=tophat_radius, sigmas=sigmas, min_object=min_object,
        )
        simple_bin, simple_t, simple_skel = simple_mask(mito, min_object=min_object)
        print(f"  proposed: {binary.sum():,} px ({100*binary.mean():.2f}%)  "
              f"skel: {skel.sum():,}  |  simple Otsu t={simple_t:.1f} -> "
              f"{simple_bin.sum():,} px skel: {simple_skel.sum():,}")

        vmin = float(np.percentile(mito, 1))
        vmax = float(np.percentile(mito, 99.5))
        magenta = mono_cmap((1, 0, 1))

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="black")
        for ax in axes.flat:
            ax.set_facecolor("black"); ax.axis("off")

        axes[0, 0].imshow(mito, cmap=magenta, vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("Raw mito", color="white")

        axes[0, 1].imshow(ridge, cmap="inferno",
                          vmin=0, vmax=float(np.percentile(ridge, 99.5)))
        axes[0, 1].set_title(f"Meijering ridge σ={list(sigmas)}", color="white")

        axes[0, 2].imshow(binary, cmap="gray")
        axes[0, 2].set_title(f"Proposed binary (top-hat→ridge→Otsu→cleanup)",
                             color="white")

        # proposed skeleton over mito
        axes[1, 0].imshow(mito, cmap="gray", vmin=vmin, vmax=vmax)
        ov = np.zeros((*mito.shape, 4))
        ov[skel, 1] = 1.0; ov[skel, 3] = 1.0
        axes[1, 0].imshow(ov)
        axes[1, 0].set_title("Proposed skeleton (green)", color="white")

        # current-style skeleton over mito
        axes[1, 1].imshow(mito, cmap="gray", vmin=vmin, vmax=vmax)
        ov2 = np.zeros((*mito.shape, 4))
        ov2[simple_skel, 0] = 1.0; ov2[simple_skel, 3] = 1.0
        axes[1, 1].imshow(ov2)
        axes[1, 1].set_title(f"Current style (Otsu on raw, t={simple_t:.0f})",
                             color="white")

        # zoom
        h, w = mito.shape
        y0, y1 = int(h * 0.40), int(h * 0.40) + 350
        x0, x1 = int(w * 0.20), int(w * 0.20) + 350
        z = mito[y0:y1, x0:x1]
        zn = np.clip((z - vmin) / (vmax - vmin), 0, 1)
        rgb = np.stack([zn, zn, zn], axis=-1)
        pb = skel[y0:y1, x0:x1]
        sb = simple_skel[y0:y1, x0:x1]
        rgb[..., 1] = np.where(pb, np.maximum(rgb[..., 1], 1.0), rgb[..., 1])
        rgb[..., 0] = np.where(sb & ~pb, np.maximum(rgb[..., 0], 1.0), rgb[..., 0])
        axes[1, 2].imshow(np.clip(rgb, 0, 1))
        axes[1, 2].set_title("Zoom: green=proposed only, red=current only",
                             color="white")

        fig.suptitle(base, color="white", fontsize=10)
        fig.tight_layout()
        out_png = os.path.join(output_dir, f"{base}_mask_preview.png")
        fig.savefig(out_png, dpi=110, facecolor="black", bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
