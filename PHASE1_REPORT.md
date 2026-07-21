# Phase 1 — Critical fixes (branch `fix/phase1-critical`)

Unbroke the four headless workflows and eliminated the silent-wrong-number
bugs, test-first: each fix has a regression test that was **red on the
pre-fix code** and **green after** the smallest surgical change. One commit per
fix (test + change together), each message citing the finding ID.

Environment: conda env `mito_protein_scanner` (Python 3.10); `pytest` added to
the env as the harness runner. Test harness seed: `tests/` + `tests/conftest.py`
(headless matplotlib) + `tests/data` → `examples/` + `[tool.pytest.ini_options]`
in `pyproject.toml`.

## The 8 fixes

| # | ID | What changed | File(s) | Test | Red → Green |
|---|----|--------------|---------|------|-------------|
| 1 | DEBT-1 / BP-1 | Removed the `--use-gui/--no-gui` option (config emitted `--no-use-gui`, which never matched); hard-coded the headless mask path. GUI still reachable via `select_mask_gui` API. | `mito_protein_line_scanner.py`, `examples/config_headless.yaml` | `tests/test_cli_flag_parity.py` | red: `--no-use-gui` not a valid option → green: all emitted flags valid |
| 2 | DEP-2 / BP-11 | Moved `import mrcfile` out of module top into `save_channel_mrcs` (graceful skip if absent, mirrors the `localthickness` pattern); added `mrcfile` to `environment.yml`. | `sted_colocalization_3d.py`, `environment.yml` | `tests/test_import_smoke.py::test_sted_imports_without_mrcfile` (+ workflow import-smoke) | red: top-level `import mrcfile` fails when mrcfile hidden → green |
| 3 | BP-2 | Unpacked 5 values (not 4) from the mask builders in both branches; producers now return a `MitoMaskResult` NamedTuple so future arity drift fails loudly. | `puncta_nn_scan.py`, `mito_protein_line_scanner.py` | `tests/test_puncta_mask_arity.py` | red: `too many values to unpack (expected 4)` (both branches) → green |
| 4 | BP-3 | Deleted the hardcoded `mito_channel=1 / target_channel=0` override so config channels flow through; dropped the dead `--scan-width` / `--sampling-radius` options + their orphaned config keys. | `mito_mask.py`, `config.yaml`, `config.yaml.template`, `examples/config.yml` | `tests/test_mito_mask_channels.py` | red: mito came through as channel 1 → green: config channels 0/2 used |
| 5 | SCI-1 | Stopped smoothing the distance (coordinate) axis; smooth only intensities. Distances stay raw and strictly monotonic as the `xp` grid for `np.interp`. | `analyze_omm_scans.py` | `tests/test_omm_distance_monotonic.py` | red: `xp = [-3,-4,-3,…]` (non-monotonic, ≠ raw) → green |
| 6 | SCI-2 | Added `validate_channel_order`: logs the assumed `0=mtDNA,1=mito,2=septin` mapping with per-channel ranges and warns (not errors) on empty/degenerate channels. Order stays fixed; no metadata read, no new option. | `sted_colocalization_3d.py` | `tests/test_sted_channel_order.py` | red: helper absent (ImportError) → green |
| 7 | DEBT-2 / DEP-8 | Removed the dangling `mito_protein_omm_localization` `console_scripts` entry (module does not exist). | `setup.py` | `tests/test_entry_points.py` | red: `No module named …omm_localization` → green |
| 8 | DEBT-3 / BP-10 | Deleted the dead `select_threshold` / `select_threshold_gui` pair (superseded by `select_mask_gui`; the GUI variant also had a latent `NameError` on `mito_cmap`). | `mito_protein_line_scanner.py` | `tests/test_dead_code_removed.py` (+ import-smoke) | red: symbols present → green (−184 lines) |

`pytest`: **43 passed**.

## Headless smoke on `examples/` (config `examples/config_headless.yaml`, `use_gui: false`)

Run from the repo root, e.g.
`mito_protein_localization network-line-scan --config examples/config_headless.yaml`
(note: click command names are hyphenated). Outputs go to the git-ignored
`.phase1_smoke_out/` scratch dir.

| Workflow | Exit | Outputs |
|----------|------|---------|
| `network-line-scan` | 0 ✅ | 119 per-mito CSVs + intensity PNGs (previously crashed on `--no-use-gui`) |
| `analyze-omm-scans` | 0 ✅ | `scan_data.csv` (10 KB) + 7 cumulative-profile PNGs |
| `plot-peak-distance-analysis` | 0 ✅ | `peak_distances.csv` (96 KB), `peak_distance_per_track.csv`, histogram PNG, summary (482 CSVs pooled) |
| `sted-colocalization-3d` | 0 ✅ | Runs end-to-end without the import crash. **No analysis output from `examples/`** — see SCI-2 status below. |

All four run to completion. Three produce rich non-empty outputs directly from
`examples/`.

## SCI-2 real-data status — **PENDING**

The two `examples/SEPT9_REPRESENTATIVE/*_3D_frame01.tif` files are shape
`(2, Y, X)` — 2-slice single-channel frames, **not** 3-channel 3D STED stacks —
so `read_3d_tiff` rejects them (`Total slices (2) is not divisible by 3
channels`) *before* the channel-order guard runs. There is no valid 3-channel
3D stack under `examples/` yet.

- The guard is **implemented and unit-tested** (`test_sted_channel_order.py`).
- It was **verified end-to-end on a synthetic 3-channel `(Z,C,Y,X)` stack**
  (not committed): the log emitted
  `channel 0 -> mtDNA: min=… p99=…` for all three channels with no false
  warnings, and STED produced full outputs (analysis CSV, radial profiles,
  half-max distances).
- **Action needed:** drop a genuine 3-channel 3D STED stack at `examples/3d/`
  (or provide a path) and re-run `sted-colocalization-3d` to confirm the logged
  per-channel ranges look sane on real data.

## `git status` / audit

`audit/` is untouched (added to `.gitignore` so it stays off the branch;
`git status` shows nothing under it). Working tree is otherwise clean apart from
an unrelated pre-existing `.DS_Store` (left for the Phase-5 cleanup).

## Self-eval (honest, two-axis)

- **Correctness & completeness — 9/10.** All 8 fixes reproduce red, are fixed
  with the smallest change, and go green; 43 tests pass; 4 workflows run
  headless. Devil's advocate: STED "non-empty output from `examples/`" is not
  met (the example data isn't valid 3-channel 3D), and the SCI-2 real-data
  check is therefore pending — both are data-availability limits, not code
  gaps, but they are real gaps against a literal reading of "non-empty outputs
  for all four."
- **Craft & discipline — 9/10.** Strict test-first with observed red output,
  one focused commit per fix, minimal surgical diffs (507+/209−), no scope
  creep into Phases 2–5. Devil's advocate: a few tests exercise the fix through
  stubs/monkeypatch (BP-2, BP-3) rather than the full pipeline (deliberate
  "thin unit" per the brief), and I left stale doc/comment references to the
  deleted `select_threshold_gui` in place (see below) rather than risk doc-sweep
  scope creep.

## Spotted but deferred (NOT touched — out of Phase-1 scope)

- Stale references to the now-deleted `select_threshold_gui` remain in comments/
  docstrings in `mito_protein_line_scanner.py` (e.g. the "still available as
  select_threshold_gui" comment near the `process_images` mask call, and the
  `select_mask_gui` docstring that says "same 4-tuple shape" — it is a 5-tuple).
  → Phase 5 doc/lint sweep.
- `.DS_Store` files tracked/dirty across the repo. → Phase 5.
- `setup.py` `install_requires` is unpinned. → Phase 3 (dependency pinning).
