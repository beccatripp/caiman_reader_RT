# Changes from the original caiman_reader

This fork starts from the Fleischmann lab repository
(https://gitlab.com/fleischmann-lab/calcium-imaging/caiman_reader, commit `ddee72e`,
"Edit: change to current Oscar configuration: 6.23.25"). It was adapted to read the
CaImAn outputs of the OLL pipeline (`OLL_Processing/oll_caiman_segmentation.py`).

`filters.py` was not in the original repository. It was supplied separately by the
original author and is included here with the changes listed under
[filters.py](#filterspy).

What did **not** change: the GUI layout, the review labels (Accept / Reject / Flag)
and their colours, how each frame is displayed (resized, then each frame scaled to its
own min–max as 8-bit; only the size changed, see §10), and how outlines are drawn
(convex hull of the footprint). Footprint orientation is now detected per plane; see
[ROI orientation on OLL outputs](#roi-orientation-on-oll-outputs).

---

## caiman-reader.py

### 1. Which cells are shown

| | Original | Now |
|---|---|---|
| Default | Always ran `filters.ROISet`: r-value prepass (> 0.3), then drop the weaker of overlapping (> 45%), correlated (> 0.1) ROI pairs | CaImAn's accepted components only (`estimates/idx_components`), no extra filtering |
| Optional | — | `python caiman-reader.py --overlap-filter` runs `filters.ROISet` with the original parameters |

The overlap filter was designed for a different dataset, so it is opt-in. When it
runs, the terminal prints how many accepted components it kept.

In the original, initial labels were `"PR"` for components outside the filtered set.
That branch could never run, because only filtered components were given outlines, and
`"PR"` has no colour entry, so it would have crashed. Every shown component now starts
unlabelled (`''`).

### 2. Choosing a folder and finding files

| | Original | Now |
|---|---|---|
| Search scope | Recursive (`os.walk`) through the selected folder and all subfolders | The selected folder only |
| HDF5 | First `.hdf5` found anywhere below the folder | First `.hdf5` in the folder (sorted) |
| Movie | First `*preprocessed.tif(f)` below the folder; crashed if none | `*preprocessed.tif(f)` in the folder, otherwise the `input_tif` path recorded in the OLL `run_parameters.txt` (the deepCAD TIFF) |
| Cancelling the dialog | Continued with the previous folder | Does nothing |
| Missing HDF5 or movie | Crash | Error dialog; the currently open plane stays loaded |

New helper: `find_movie(fold)`.

### 3. Loading is all-or-nothing

Originally, global state (folder, cell list, save path) changed partway through
loading, so a failure could leave the reader showing one plane while saving to
another. Now the HDF5 is read into local variables inside a `with` block, and the
reader switches to the new plane only once everything has loaded. If there are no
components to review (none accepted, or the overlap filter removed all of them), an
error dialog appears and the previous plane stays open.

`update_mpl()` is now called after `active_cell` is set to the new plane's first cell.
The original called it before, while `active_cell` still held the old plane's value.

### 4. ΔF/F trace

The original read `estimates/F_dff` directly. CaImAn stores it as the string
`'NoneType'` unless `detrend_df_f()` was run, which the OLL pipeline did not do before
OLL_Processing commit `2af3a1d`, so the reader crashed.

New helper `load_dff(h5)`:
- uses `estimates/F_dff` when it is a real array the same shape as `C`;
- otherwise computes ΔF/F from `C` with the OLL pipeline's formula: F0 = 8th
  percentile per component, floored at 1e-6, ΔF/F = (C − F0) / F0 (constants
  `F0_PERCENTILE`, `F0_FLOOR`). This matches `traces_dff.npy`.

`oll_caiman_segmentation.py` now writes `F_dff` itself, so newer HDF5 files use the
stored array.

### 5. SNR and r-value

The original read `estimates/SNR_comp` and `estimates/r_values` directly, which crashed
when CaImAn's `evaluate_components` had failed (stored as `'NoneType'`).

New helper `load_quality_metric(...)`: HDF5 value if it is a 1-D array, otherwise the
OLL `quality_scores.npz` in the same folder, otherwise NaN. In that fallback, r-values
are 0 (not measured), so `--overlap-filter` removes every cell on those planes.

### 6. Where review labels are saved

| | Original | Now |
|---|---|---|
| File | Built by splitting the HDF5 path on `_` and dropping the last piece | `<hdf5 name>_quality.csv` next to the HDF5, i.e. `results_quality.csv` in the plane folder |
| Reload | Any `*quality.csv` found recursively | That exact file |

With OLL paths the original produced
`.../OLL_DATA_S2026_Processed/Segmentation_quality.csv` for **every** plane, so each
save overwrote the last, and the file was never found again on reload.

Other changes:
- Saving with no plane open does nothing (originally it crashed).
- When loading a CSV, entries for components not in the current list are ignored.
  Components missing from the CSV start unlabelled. This lets labels carry over when
  `--overlap-filter` is toggled.

New helper: `quality_csv_path(h5_file)`.

### 7. Last component was never shown

`generate_footprints` looped over `range(feet.shape[1]-1)`, so the last accepted
component never got an outline and was left out of the review list. It now loops over
every component. It also returns early with no outlines when the component list is
empty; the original crashed on an empty list.

### 8. Movie loading

The original loaded the whole movie up front: 20 `joblib` worker processes, each
holding float copies of its upscaled chunk, then a list-plus-concatenate copy of the
full 8-bit movie. On OSCAR this was killed for running out of memory, and it also
crashed on movies with fewer than 20 frames (a chunk step of 0).

Frames are now produced on demand by a `LazyMovie` class (returned by
`load_tiffstack`):
- **Uncompressed TIFFs** are opened with `tifffile.memmap`, so nothing is read until a
  frame is shown.
- **Compressed TIFFs** are decoded one page at a time.
- **The last 64 prepared frames are cached,** so scrubbing back and forth doesn't redo
  the work.
- **The pixel processing is unchanged:** `skimage.transform.resize` to 2×, then each
  frame scaled to its own min–max as 8-bit. Frames were checked to be identical to the
  previous loader for uint16 and float32, uncompressed and zlib-compressed, including
  non-square frames.
- **Cost and trade-off:** memory no longer grows with recording length, and a plane
  opens almost immediately. Preparing a new 512×512 frame takes about 20 ms, so
  playback over frames not yet shown may run slightly below the 20 ms timer.

`joblib` is no longer used.

### 9. Smaller changes

- **Imports:** removed `yaml` (unused) and `joblib` (no longer needed); added `sys` (for
  `--overlap-filter`), `tkinter.messagebox` (for the error dialogs) and
  `matplotlib.path.Path` (for clicking outlines, §11).
- **Welcome text:** now tells the user to choose a plane folder, e.g.
  `Segmentation_caImAn/<date>/<mouse>/plane0`.

### 10. Movie size fitted to the screen

The original always showed the movie at 2× its native size. A 512×512 plane then
takes 1024 px of height before the trace plot and buttons, which is more than an
OSCAR Desktop screen, so the bottom of the window could not be reached.

The scale is now worked out when a plane is opened (`fit_scale`): the largest scale,
up to 2×, at which the whole window fits on the screen. The margins
(`SCREEN_MARGIN_W`, `SCREEN_MARGIN_H`) leave room for the title bar and the desktop's
panels. The terminal prints the scale used. Examples for a 512×512 plane: 0.59× on a
1024×768 screen, 0.97× on 1536×960, 1.91× on 2560×1440.

The outlines use the same scale. To keep them in line with the movie at fractional
scales:
- **Rounding:** the movie size is rounded the same way `skimage.transform.rescale`
  sizes the outlines.
- **No anti-aliasing:** the footprint rescale no longer uses anti-aliasing. It had no
  effect when enlarging, but when shrinking its blur would widen every outline.
  Results at 2× are unchanged.

### 11. Selecting cells by clicking the movie

Clicking inside an outline drawn on the movie makes that cell the active one. This is
the same as choosing it in the cell list: the trace, readout and Accept / Reject / Flag
buttons switch to it, and the list scrolls to it.
- **Which outlines:** only the outlines currently drawn can be clicked, so tick
  **Display all** (or one of the per-label boxes) to pick cells from the movie.
- **Overlaps:** where outlines overlap, the smallest one containing the click wins.
- **Active cell:** when several outlines are shown, the active cell is drawn in
  yellow, thicker, on top. Before, every outline was red, so the active cell could not
  be told apart.

New helpers: `draw_outline`, `select_on_movie`, `select_cell`, and `readout_text` (split
out of `update_readout` so `fit_scale` can measure the readout's size).

**Movie frames no longer pile up.** `tif2frame` added a new image to the canvas for
every frame shown and never removed the old ones, so playback kept adding items to the
canvas. The previous frame is now deleted first.

---

## filters.py

Starting point: the `filters.py` supplied by the original author. The `ROISet` logic
and its thresholds are unchanged.

1. **`ROISet.__init__(cm_data, F_dff=None, snr=None, rval=None)`:** the optional
   arguments override the values read from the HDF5. The reader passes in its own
   ΔF/F, SNR and r-values (sections 4 and 5), because OLL HDF5 files could store them
   as `'NoneType'`.
2. **`find_overlap` computes the pixel intersections as a sparse matrix product.** The
   original tiled every mask pair (N × N × H × W), which needs tens of GB for a few
   hundred ROIs on a 512 × 512 frame. For binary masks,
   (aᵢ + aⱼ − Σ|p − q|) / 2 is the intersection, which is exactly `B @ B.T`. On random
   test sets this gave identical `overlap_percents` and identical kept ROIs to the
   original.
3. **Zero ROIs after the prepass:** `find_overlap` no longer crashes on an empty array.
   The reshape uses an explicit pixel count instead of `-1`.

---

## README.md

- Added the "OLL fork notes" section: where the code came from, how it maps to OLL
  outputs, the known issue below, and how to set up and run it on OSCAR.
- The original sentence "Creates a new file in the parent directory" is kept, but in
  this fork the CSV is written in the plane folder itself (section 6).

---

## ROI orientation on OLL outputs

On OLL `results.hdf5` files, ROI outlines used to be drawn **transposed** (mirrored across
the diagonal) relative to the movie: a cell at row r, column c was outlined at row c,
column r, which looked like outlines shifted off the cells.

**Cause:** `oll_caiman_segmentation.py` writes CaImAn's memmap in row-major order, while
CaImAn reads it column-major, so CaImAn segments a transposed movie. The pipeline's own
outputs (`roi_masks.npy`, `stat.npy`, `roi_outlines.png`) undo this, but the reader read
`estimates/A` with the standard CaImAn (Fortran) layout. The pipeline leaves
`results.hdf5` as-is on purpose, so orientation stays the same across old and new runs.

**Now:** the new helper `footprint_order(h5, fold)` picks the pixel order per plane, and
`generate_footprints` reshapes each footprint with it:
- **With `roi_masks.npy`:** the order ('C' or 'F') whose accepted footprints overlap the
  pipeline's masks more.
- **Without it:** 'C' if the folder has a `run_parameters.txt` (an OLL output),
  otherwise CaImAn's 'F'.

The terminal prints the order used. The 'F' reshape is unchanged from the original
(`reshape((dims[1], dims[0])).T` is the same as `reshape(dims, order='F')`). Checked on
synthetic planes in both layouts: outlines sit on the cells in each. Trace, SNR and
r-value per component ID were never affected. Non-square frames are scrambled in CaImAn
itself, so their outlines can't be fully right either way.
