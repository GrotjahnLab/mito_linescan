# Blind Genotype Review Utility

## Overview

`mito_blind_review` is a **standalone** utility for blindly scoring microscopy
images as `MorphologyA`, `MorphologyB`, or `IDK`. It exists so that the manual
genotype-scoring procedure is documented and reproducible for reviewers.

**It is not part of the analysis pipeline.** Nothing in the pipeline imports it,
and it imports nothing from the pipeline modules. Removing it would not affect
any analysis workflow.

A reviewer is shown one grayscale image at a time, in a per-reviewer randomized
order, with the real filename hidden. Their calls are written to a per-reviewer
CSV and merged as a new column into a master genotype sheet.

## Installation

The tool ships with the package and adds no new dependencies:

```bash
cd mito_linescan
micromamba activate mito_protein_scanner
pip install -e .
```

This registers the `mito_blind_review` console command.

## Quick Start

```bash
mito_blind_review \
  --input-directory example_blindreview/blinded_mitos \
  --reviewer "Jane Doe" \
  --genotype-csv example_blindreview/Sept9_genotype.csv
```

A window opens showing the first image. Score each image with a button or a
keyboard shortcut; the tool advances automatically.

## CLI options

| Option | Required | Description |
|--------|----------|-------------|
| `--input-directory DIR` | yes | Directory of TIFFs to review. |
| `--reviewer NAME` | no | Reviewer's name. If omitted you are prompted before the first image is shown. |
| `--genotype-csv PATH` | no | Master sheet with a single `FILE` column of stems (e.g. `S9MEF_01`). If given, calls are merged in as a new column. |
| `--output-directory DIR` | no | Where to write outputs. Defaults to `--input-directory`. |
| `--seed INT` | no | Shuffle seed. Default derives a **stable** seed from the reviewer name. |
| `--input-pattern GLOB` | no | Glob for TIFFs within the input dir (default `*.tif`). |
| `--p-low FLOAT` | no | Lower percentile for auto-contrast `vmin` (default `1.0`). |
| `--p-high FLOAT` | no | Upper percentile for auto-contrast `vmax` (default `99.5`). |
| `--allow-new-rows` | no | Append master-sheet rows for input files not already listed. |
| `--overwrite-column` | no | Overwrite an existing reviewer column instead of erroring. |
| `--ground-truth PATH` | no | CSV of true genotypes (`BLINDED`, `Ground_Truth` columns). Pops up a results window at the end with the reviewer's accuracy. |
| `--dry-run` | no | Score nothing (stub every image as `IDK`), no GUI. Exercises discovery + merge for headless smoke testing. |

## Ground-truth scoring

If `--ground-truth PATH` is given, after scoring the tool compares the
reviewer's calls to the truth sheet and, at the very end, pops up a results
window showing the **percent accuracy**, a score-dependent cartoon face, and a
(hopefully) funny message. The accuracy is also printed to the terminal.

- The sheet is auto-delimited (the example is tab-separated) and uses the
  `BLINDED` (stem) and `Ground_Truth` (call) columns, falling back to the first
  and last columns if those names are absent.
- Legacy `WT`/`KO` truth values (any case) are automatically mapped onto
  `MorphologyA`/`MorphologyB`, so an older ground-truth sheet still scores
  correctly.
- Only images that were **scored and present in the truth sheet** are counted.
- An `IDK` never matches a MorphologyA/MorphologyB truth, so it counts as
  incorrect (but is also tallied separately in the printed summary).
- With `--dry-run`, accuracy is computed and printed but no window opens.

```bash
mito_blind_review \
  --input-directory example_blindreview/blinded_mitos \
  --reviewer "Jane Doe" \
  --ground-truth example_blindreview/Ground_Truth.csv
```

## Controls

| Action | Button | Keys |
|--------|--------|------|
| Call **MorphologyA** | green `MorphologyA` | `1` or `a` |
| Call **MorphologyB** | red `MorphologyB` | `2` or `b` |
| Call **IDK** (unsure) | gray `IDK` | `3` or `i` |
| Reset contrast to auto | — | `r` |

Below the image, two horizontal sliders set the display **min** (`vmin`) and
**max** (`vmax`) limits over the current image's raw intensity range. `min` is
always kept below `max`.

This is a **single forward pass**: no back-navigation and no resume. If you
close the window early, whatever you scored so far is written out, and the tool
prints how many images were scored vs. skipped.

## Blinding

- The filename is never shown. The title shows only `Image 7 / 40` and the
  reviewer's name.
- Presentation order is randomized per reviewer via
  `numpy.random.default_rng(seed)`. The default seed is a stable hash of the
  reviewer name, so a given reviewer always sees the same order. The seed used
  is recorded in the session JSON.
- Real filenames appear only in the output CSVs.

## Image loading

Only one 2D plane is ever displayed:

- **2D `(Y, X)`** — shown as-is.
- **3D** — the smallest axis is inspected. If its length is `<= 4` it is treated
  as a channel axis and the **last** channel is shown (handles both `(C, Y, X)`
  and `(Y, X, C)`). Otherwise the array is treated as a **z-series** and a
  **max-intensity projection** along the leading axis is shown.

Display is grayscale, `interpolation='nearest'`, no ticks, equal aspect.

## Auto-contrast

On every image the display limits are auto-chosen with a robust percentile
stretch: `vmin = p1`, `vmax = p99.5` of the **nonzero** pixels (falls back to
all pixels if the image has no zeros, and to the full min/max if the percentiles
collapse). The sliders reset to these values on each advance; press `r` to
recompute for the current image. Percentile constants are the module-level
`DEFAULT_PLOW` / `DEFAULT_PHIGH` and are overridable with `--p-low` / `--p-high`.

## Outputs

Written into `--output-directory` (default: the input directory):

### 1. `<reviewer>.csv`

One row per scored image:

```
FILE,call,reviewer,presentation_index,vmin,vmax,seconds_on_image,timestamp_utc
```

`FILE` is the file **stem** (e.g. `S9MEF_01`) so it joins the master sheet. The
filename is sanitized (lowercase, non-alphanumerics → `_`), but the `reviewer`
column keeps the original string. `seconds_on_image` is the time from display to
click, recorded for later QC.

### 2. `<reviewer>_session.json`

Sidecar with the seed, input directory, percentile settings, tool version, and
start/end times.

### 3. `<reviewer>_genotype_results.csv`

A standalone per-reviewer results table with just two columns — `FILE` (stem)
and a call column named after the reviewer. This is always written. When a
`--genotype-csv` master is supplied, the table is taken from the merged master
so its `FILE` list and empty cells match the master; otherwise it lists only the
files this reviewer scored.

### 4. Updated `--genotype-csv`

The reviewer's calls are added as a **new column** named after the reviewer,
joined on the `FILE` stem:

- Existing column order is preserved; the new column is appended at the right.
- Reviewed files absent from the master sheet are **warned and listed**; they
  are not added unless `--allow-new-rows` is passed.
- Master-sheet rows with no call are left empty.
- If a column with the reviewer's name already exists, the tool **errors out**
  unless `--overwrite-column` is passed.
- The sheet is written atomically (temp file + replace) and the original is
  backed up to `<name>.csv.bak` before the first write.
