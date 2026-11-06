# Auto-QC classifier for CaImAn ROIs

Learns the manual Accept / Reject labels saved by caiman-reader (`results_quality.csv`) and writes
`results_quality_model_<tag>.csv`, in the same format, for planes nobody has reviewed. `<tag>` names the model
family (`model_tag` in `config.yaml`): `xgboost` for this model; later models get their own tag
(`xgboost-context`, `xgboost-pretrained`, `cnn-resnet`, ...) so their files can sit side by side. Modelled on
the maze-runners XGBoost auto-QC (omckissi, `notebooks/S03_auto_qc_classifier_xgboost.ipynb`), adapted to the
OLL S2026 outputs. caiman-reader loads a plane's model file by default (and asks which, if there are several)
so you can check or correct the model's labels. Save writes `results_quality_model_<tag>_useredits.csv`, so
the model's own file is never changed, and appends a row to `review_log.csv` (below).

- Code: `caiman_reader_RT/classifier/`
- Everything it produces except `results_quality_model_<tag>.csv`: `/oscar/home/rtripp/data/rtripp/caiman_classifier_model/`
- Settings, training planes and locked test planes: `config.yaml`

## Setup (once)

```bash
cd /oscar/home/rtripp/data/rtripp/caiman_reader_RT
module load miniconda3          # or however you usually get conda on OSCAR
conda env create -f classifier/environment.yml      # -> caiman_classifier_env
mkdir -p /oscar/home/rtripp/data/rtripp/caiman_classifier_model/{logs,lists}
```

The SLURM scripts call `~/.conda/envs/caiman_classifier_env/bin/python` directly. Set `CMCLF_PY` to use
another interpreter, and `CMCLF_REPO` to use another checkout. Submit every job from the repo root, as below.
The scripts refuse to run on a login node outside a SLURM job. They do run in an `interact` session or on the
VSCode node.

## Workflow

```bash
cd /oscar/home/rtripp/data/rtripp/caiman_reader_RT
OUT=/oscar/home/rtripp/data/rtripp/caiman_classifier_model
V=v20261105                      # model version name, any string

# 1. Smoke test one plane: interact -n 2 -m 16g -t 30:00, then
conda activate caiman_classifier_env
python classifier/scripts/extract_features.py --plane .../Segmentation_caImAn_002/26-03-12-OLL/RB01/plane0
cat $OUT/features/26-03-12-OLL_RB01_plane0.checks.json   # mask_dice ~1, dff_corr ~1, order "C"

# 2. Features for the 45 training planes (one array task per plane, ~1-2 min each)
sbatch --array=0-44 classifier/slurm/extract_features.sbatch --training

# 3. Train: nested CV for all models, leave-one-animal-out, ablation, final models (a few hours)
sbatch classifier/slurm/train.sbatch --version $V
#    -> $OUT/reports/$V/report.md, cv_summary.csv, plots;  $OUT/models/$V/{dev,deploy}

# 3b. Optional: movie features (G9), then train again with them
sbatch --array=0-44 classifier/slurm/movie_features.sbatch --training
sbatch classifier/slurm/train.sbatch --version ${V}_movie --with-movie
#    keep full_movie only if it beats full in cv_summary.csv / ablation_summary.csv;
#    to deploy it: train.sbatch --version ${V}_movie --final-set full_movie --stages final

# 4. Score the locked test set: ONCE per version, after all decisions are made
sbatch classifier/slurm/evaluate.sbatch --version $V --locked-test

# 5. Blind prospective check: 3 random unreviewed planes
python classifier/scripts/make_plane_list.py --random 3 --seed 1 --out $OUT/lists/blind_check.txt
sbatch --array=0 classifier/slurm/apply.sbatch --version $V --plane-list $OUT/lists/blind_check.txt
#    review those planes with caiman-reader.py --no-model-labels, save, then:
sbatch classifier/slurm/evaluate.sbatch --version $V --compare --plane-list $OUT/lists/blind_check.txt
#    prints PASS/FAIL against prospective_pass in config.yaml

# 5b. One animal, e.g. all unreviewed RB06 planes (58 planes -> 12 tasks of 5)
python classifier/scripts/make_plane_list.py --mouse RB06 --out $OUT/lists/RB06_unlabeled.txt
sbatch --array=0-11 classifier/slurm/apply.sbatch --version $V --plane-list $OUT/lists/RB06_unlabeled.txt
#    then review with: python caiman-reader.py   (loads the model file; Save writes ..._xgboost_useredits.csv
#    and logs time and changes)

# 6. Everything else unreviewed (5 planes per task, at most 20 tasks at once)
python classifier/scripts/make_plane_list.py --out $OUT/lists/unlabeled.txt
N=$(wc -l < $OUT/lists/unlabeled.txt)
sbatch --array=0-$(( (N + 4) / 5 - 1 ))%20 classifier/slurm/apply.sbatch --version $V --plane-list $OUT/lists/unlabeled.txt
grep -h -E "wrote|skipped|Failed" $OUT/logs/cmclf_apply_*.out | sort | uniq -c | sort -rn | head
```

`make_plane_list.py` filters: `--mouse RB06 RB07`, `--session 26-04-10-OLL`, `--plane plane1`,
`--include-labeled`. Run a single plane with `apply.py --version $V --plane DIR`, and add `--dry-run` to print
the counts without writing anything.

## What gets written where

| File | Written by | Notes |
|---|---|---|
| `<plane>/results_quality_model_<tag>.csv` | `apply.py` | no header, `roi_id,A\|R`, one row per `estimates/idx_components` in that order (byte-for-byte the reader's format) |
| `$OUT/predictions/<version>/<pid>.csv` | `apply.py` | `roi_id, p_accept, label, threshold, model_version, model_tag` |
| `$OUT/features[_movie]/<pid>.parquet`, `<pid>.checks.json` | `extract_features.py`; `apply.py` caches core features it had to compute | cached; `--overwrite` to recompute |
| `<plane>/results_quality_model_<tag>_useredits[-vNN].csv` | `caiman-reader.py` | your corrections, when you started from a model file; same format. Starting again from the model's labels writes a new `-vNN` version. Not read by training |
| `$OUT/review_log.csv` | `caiman-reader.py` | one row per Save that started from a model file: minutes open, cells, model vs saved accepts, A→R / R→A / other changes, blanks, edits resumed at load. Append-only |
| `$OUT/models/<version>/{dev,deploy}/{model.json,metadata.json}` | `train.py` | XGBoost JSON; metadata has features, threshold, parameters, training planes, git commit, CV summary |
| `$OUT/reports/<version>/` | `train.py`, `evaluate.py` | see below |

**Never touched:** `results_quality.csv`. `apply.py` skips planes that already have a manual
`results_quality.csv` (`--include-labeled` to apply anyway, e.g. to compare) and planes that already have a
`results_quality_model_<tag>.csv` for the same tag (other tags' files are never touched). With `--replace`, the
old file is renamed to `results_quality_model_<tag>__<version that wrote it>.csv` first, so nothing is
deleted. Models trained before tags existed (`v20261105`, `v20261105_kappa`) use `model_tag` from the config. It fingerprints every file in
the plane folder before and after writing, and fails if anything other than the new file changed.

## Labels and planes

- 45 training planes (`config.yaml`). Differences from the original request: RB05 plane0 26-03-29 → 26-03-21
  (no RB05 folder on 26-03-29). RB07 plane0 26-03-26 has no labels and 26-03-24 is only partly labelled (23 of
  157 cells), so RB07 plane0 has neither.
- A = 1, R = 0. Blank (unreviewed) and F labels are excluded, not treated as Reject. `label_summary_*.csv`
  counts them per plane.
- Candidates are every CaImAn-accepted component (`idx_components`), which is the pool the reader shows.
  No `filters.ROISet` pre-filter.
- **Locked test set:** one plane0 per animal, drawn with `numpy.random.default_rng(0)` before any training:
  RB01 26-04-09, RB05 26-03-21, RB06 26-03-24, RB07 26-03-15, RB08 26-03-05, RB09 26-03-01. `train.py` never
  loads them until the deployment refit at the very end, and never scores them.
- Corrections of model labels go to `results_quality_model_<tag>_useredits.csv`, not `results_quality.csv`,
  so training never picks them up by accident. Using them for training needs a deliberate code change.

## Features (`cmclf/features.py`)

Built from `results.hdf5`, `ops.npy` (`meanImg`) and `run_parameters.txt`. Footprints are `estimates/A`
thresholded at 0.2 × peak (as `roi_masks.npy`), in the movie orientation (C order on OLL outputs, detected as in
the reader). dF/F is the reader's / `traces_dff.npy` formula. Neighbour features search all components,
including CaImAn rejects.

| Group | Features | Pattern it targets |
|---|---|---|
| G1 CaImAn | `snr`, `r_value`, `noise_sn` | SNR |
| G2 shape | `area`, `area_rel`, `fp_max`, `ecc`, `compactness`, `solidity`, `n_islands`, `spread` | ROI size, footprint |
| G3 location | `dist_center_norm`, `dist_bright_center_norm` (to the optical centre of `meanImg`), `dist_edge_px`, `radial_rank` | centre vs periphery, outer ring |
| G4 brightness | `img_roi_rel`, `img_annulus_rel`, `img_contrast` (vs a ring excluding other ROIs), `bg_rel` (CaImAn background under the ROI) | bright centre, high background |
| G5 signal | `dff_peak`, `dff_p95`, `z_peak`, `skew`, `kurt`, `events_per_min`, `autocorr1` (denoised dF/F); `robust_snr`, `resid_noise_dff`, `resid_to_range`, `hf_power_frac` (raw `C + YrA`) | SNR, dF/F vs z-score |
| G6 neighbour | within 75 px, the most correlated component: `nbr_max_corr` (raw), `nbr_max_corr_C`, `nbr_dist`, and this ROI's ratios to it (`nbr_dff_ratio`, `nbr_z_ratio`, `nbr_bright_ratio`, `nbr_area_ratio`, `nbr_snr_ratio`), `nbr_is_accepted`; `n_corr_nbrs` (r > 0.5), `max_dice` | ghost ROIs from bleed-over |
| G7 global | `bg_f_corr` (with CaImAn's background traces), `pop_corr`, `pop_diff_corr` (fast shared fluctuations), `pop_rhythm`, `plane_mean_corr` | global shivering |
| G8 context | plane medians of SNR, r and area, `n_accepted`, `caiman_accept_frac`, within-plane ranks of SNR, dF/F peak, area, robust SNR and brightness | session-to-session drift |
| G9 movie (optional) | `mv_core_rim_corr`, `mv_rim_core_std_ratio`, `mv_shift_corr`, `mv_shift_speed_corr`, `mv_cn_roi`, `mv_cn_contrast`, `mv_shift_std`, `mv_shift_speed_mean` (`cmclf/movie_features.py`) | shivering / motion |

`rep_*` columns re-create the previous student's 28 features (session statistics per plane here, since each
training plane is its own FOV). They are used only by the `replica` baseline.

## Training and validation (`cmclf/model.py`, `scripts/train.py`)

- **Nested leave-one-plane-out CV** on the 40 dev planes. For each held-out plane, the inner 5-fold GroupKFold
  by plane picks the XGBoost settings (grid in `config.yaml`), the number of trees (minimum of the mean inner
  log-loss curve) and the **decision threshold that maximises Cohen's κ** (`threshold_objective`; balanced
  accuracy put it at ~0.21 and accepted ~44% more cells than the manual labels). All of these come from the
  inner folds only.
- **Models compared:** `always_reject`, `snr_logistic`, `logistic_all`, `replica` (the previous student's
  features and settings, threshold 0.5), `full` (G1–G8, tuned), and `full_movie`.
- **Leave-one-animal-out** for `replica` and the final feature set shows how well it transfers to a new animal.
- **Ablation:** `full` minus each group, with `delta_*` columns relative to `full`.
- **Final:** tuned and fitted on the 40 dev planes (`models/<v>/dev`, scored once on the locked test set by
  `evaluate.py --locked-test`), then refit with the same settings on all 45 planes (`models/<v>/deploy`, used by
  `apply.py`). Feature importance is the mean |SHAP|.

**Metrics** (pooled per cell, and per-plane mean ± SD). Only ~25% of cells are accepted, so "always reject"
already scores ~75% accuracy. Read balanced accuracy, PR-AUC (chance = the accept rate) and κ first. Also
reported: ROC-AUC, accuracy next to the majority baseline, Accept precision / recall / F1, MCC, Brier score and a
calibration plot, and the per-plane error in the number of accepted cells (absolute and %).
`cv_disagreements_*.csv` lists the most confident disagreements (plane, roi_id) to look at in the reader.

## xgboost-context (in progress): cross-session registration

Stage 1 registers sessions of the same mouse x plane to each other, so later stages can ask whether a footprint
is also present in nearby sessions, which is evidence that it is a real cell.
`scripts/register_fov.py`, `cmclf/registration.py`; settings under `registration:` in `config.yaml`.

- Images: each session's `ops.npy` `meanImg`, high-pass filtered and contrast normalised.
- Rigid: rotation search (+-12 deg in 1 deg steps, then finer) with phase correlation for the shift. Nonrigid:
  phase correlation in 128 px blocks overlapping by half, outlier blocks replaced, lightly smoothed, refined 3
  times. On synthetic images with a known rotation, shift and smooth warp it recovers the mapping to 0.2 px
  median, 0.5 px 95th percentile, and a 7 deg / 40 px offset exactly (`tests/test_registration.py`).
- **Pairs, not one reference.** The OLL FOV rotates by up to ~7 deg and shifts by 30+ px between sessions, and
  drifts over weeks: on RB01 plane0, registering consecutive days directly gave image correlation 0.83, while
  registering both to a session two weeks earlier gave 0.3-0.5. So each session is registered directly to its
  next 3 sessions by date (`pairs/`), which is what the context features need. All sessions are also registered
  to one reference (`ref/`), but only for the overview image.
- QC per registration, with no registration / rigid / nonrigid: correlation of the enhanced images (border
  excluded), and the fraction of CaImAn-accepted centroids within 3 px of one in the other session **minus
  chance** (the same fraction after a 40 px offset). Most accepted components aren't cells and needn't reappear,
  so the raw fraction is low even when aligned. Nonrigid is kept only if it makes neither worse than rigid. A
  registration **fails** if the correlation is < 0.5 or the beyond-chance match is < 0.05 (provisional).
- ROICaT's existing tracking (`CellTracking_ROICat/`) is not used. `registration/v1` (single reference, +-3 deg)
  was superseded by `v2`.

```bash
python classifier/scripts/register_fov.py --list                   # groups and their array index
sbatch --array=0-11 classifier/slurm/register.sbatch --run v2     # one mouse x plane per task, ~20-30 min
```

Output per group, `$OUT/registration/<run>/<mouse>_<plane>/`: `pairs/<pidA>__<pidB>.npz` and `ref/<pid>.npz`
(transforms, A-space -> B-space pull maps, see `cmclf/registration.Transform`), `pairs.csv`, `ref.csv`,
`summary.json`, `pairs_overlays_*.png` (consecutive sessions, earlier magenta and later green; aligned
structure is white), `pairs_qc.png` (QC against date), `stack.png` (reference vs the average of the sessions
that pass against it). An existing group folder is never overwritten (`--run`).
**Gate 1:** look at these before anything uses the transforms.

## Layout

```
classifier/
  config.yaml  environment.yml  README.md
  cmclf/   io.py (planes, labels, HDF5)  features.py (G1-G8, replica)  movie_features.py (G9)
           model.py (CV, tuning, fits)   report.py (metrics, plots)    jobs.py (login guard, array task)
           registration.py (cross-session rigid + nonrigid, QC)
  scripts/ make_plane_list.py  extract_features.py  train.py  evaluate.py  apply.py  register_fov.py
  slurm/   extract_features.sbatch  movie_features.sbatch  train.sbatch  evaluate.sbatch  apply.sbatch
           register.sbatch
  tests/   test_registration.py (synthetic, no data needed)
```
