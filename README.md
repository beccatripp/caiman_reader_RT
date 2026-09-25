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
- Click an outline on the movie to select that cell; tick **Display all** first to see every outline. The active cell is drawn in yellow.

**Known issue: ROI outlines are transposed on OLL outputs.** `oll_caiman_segmentation.py`
writes CaImAn's memmap in C (row-major) pixel order, but CaImAn reads it as Fortran
order, so CaImAn segments the movie transposed. The pipeline's own outputs
(`roi_masks.npy`, `stat.npy`, `roi_outlines.png`) undo this and are correct, but this
reader follows the CaImAn convention, so on OLL `results.hdf5` files every outline is
drawn mirrored across the diagonal relative to the movie (a cell at row r, column c is
outlined at row c, column r). Trace, SNR and r-value per cell ID are unaffected; do not
judge cells by which part of the movie their outline covers. Only square frames give a
pure transpose; non-square frames would be scrambled in CaImAn itself.

Runs as a GUI: use an OSCAR OnDemand Desktop session, not the login node or Code Server.
```bash
conda create -y -n caiman_reader_env python=3.10 numpy scipy matplotlib tifffile h5py pillow scikit-image tk
conda activate caiman_reader_env
python caiman-reader.py                    # CaImAn-accepted cells
python caiman-reader.py --overlap-filter   # + filters.ROISet
```
