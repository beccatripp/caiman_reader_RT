## CaImAn Reader
### A tkinter-based GUI for visualizing and manually reviewing segmentation quality in CaImAn hdf5 outputs.
Uses stereotyped input folders including (minimum):
- CaImAn output hdf5 file
- TIFF file of neural data (can expand to other video formats if requested!)

Creates a new file in the parent directory with mutable quality values for auto-accepted components
- TODO: add flexibility for component type selection

Initial Contributors:
- Camille Donoho

## OLL fork notes
Fork of https://gitlab.com/fleischmann-lab/calcium-imaging/caiman_reader, adapted for the OLL pipeline
outputs (`OLL_Processing/oll_caiman_segmentation.py`). Full list of changes: `CHANGES.md`.

- Open a plane folder, e.g. `.../Segmentation_caImAn/26-02-24-OLL/RB06/plane0`
- Cells shown are CaImAn's accepted components (`estimates/idx_components`); no extra filtering by default
- Optional `--overlap-filter` applies `filters.ROISet`: drops accepted cells with r-value <= 0.3, then for ROI pairs
  overlapping > 45% with dF/F correlation > 0.1 keeps only the higher r-value x SNR one (filtered cells are hidden)
- Movie: a `*preprocessed.tif` in the folder if present, otherwise the `input_tif` listed in `run_parameters.txt` (the deepCAD TIFF); frames are read and resized on demand, so memory use does not grow with recording length
- dF/F: `estimates/F_dff` if saved, otherwise computed from `C` with the pipeline's formula (8th-percentile F0)
- SNR / r-value: from the HDF5, falling back to `quality_scores.npz`
- Review labels are saved to `results_quality.csv` in the plane folder
- The movie is sized so the whole window fits the screen (at most 2x)
- Trace plot: orange = dF/F on the left axis; faint blue = raw (`C + YrA`) and dashed = baseline F0, both
  on the right axis in fluorescence units. Two independent boxes in the plot toolbar:
  - **Rolling F0**: F0 is the 8th percentile of `C` in a sliding 60 s window (frame rate from the HDF5 or
    `run_parameters.txt`; 500 frames if none) instead of over the whole session, which removes slow drift
  - **Z-score**: dF/F z-scored per cell over the whole session, (x - mean) / std, as in `traces_zscore.npy`
- **Eraser (drag to reject)**: with the box ticked, drag across the movie; every visible outline whose line
  the cursor crosses is labelled Reject. **Undo** (or Ctrl+Z) restores the last stroke. Rejected outlines
  are dashed. Tick **Display all** (or **Unreviewed ROIs**) first so the outlines you want to erase are shown
- Click an outline on the movie to select that cell; tick **Display all** first to see every outline. The active cell is drawn in yellow.

**ROI orientation on OLL outputs.** `oll_caiman_segmentation.py` writes CaImAn's memmap
in C (row-major) pixel order, but CaImAn reads it as Fortran order, so the footprints in
an OLL `results.hdf5` are transposed relative to the movie. The reader detects this and
reads them in C order, so outlines sit on the right cells. It compares the footprints
with the pipeline's `roi_masks.npy` (which is in the movie's orientation), or, without
that file, treats a folder with `run_parameters.txt` as an OLL output. The terminal
prints which order was used. Non-square frames are scrambled in CaImAn itself, so their
outlines can't be fully right either way.

Runs as a GUI: use an OSCAR OnDemand Desktop session, not the login node or Code Server.
```bash
conda create -y -n caiman_reader_env python=3.10 numpy scipy matplotlib tifffile h5py pillow scikit-image tk
conda activate caiman_reader_env
python caiman-reader.py                    # CaImAn-accepted cells
python caiman-reader.py --overlap-filter   # + filters.ROISet
```
