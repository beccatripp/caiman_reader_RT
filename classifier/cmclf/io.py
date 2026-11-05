"""Plane discovery, label files and loading of OLL CaImAn plane folders.

The orientation, quality-metric and frame-rate logic mirrors caiman-reader.py
(footprint_order, load_quality_metric, load_frame_rate), which can't be imported
because it builds its Tk window at import time."""
import csv
import glob
import os
import subprocess
from dataclasses import dataclass

import h5py
import numpy as np
import yaml
from scipy.sparse import csc_matrix

HDF5_NAME = "results.hdf5"
MANUAL_CSV = "results_quality.csv"
MODEL_CSV = "results_quality_model.csv"

CLASSIFIER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    with open(path or os.path.join(CLASSIFIER_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


def git_commit():
    try:
        return subprocess.run(["git", "-C", CLASSIFIER_DIR, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ── Planes ───────────────────────────────────────────────────────────────────

def plane_id(session, mouse, plane):
    return f"{session}_{mouse}_{plane}"


def plane_from_dir(fold):
    """.../<session>/<mouse>/<plane> -> plane dict."""
    parts = os.path.normpath(fold).split(os.sep)
    session, mouse, plane = parts[-3], parts[-2], parts[-1]
    return dict(session=session, mouse=mouse, plane=plane, dir=fold,
                pid=plane_id(session, mouse, plane))


def _expand(cfg, key):
    """config {mouse: {plane: [dates]}} -> list of plane dicts."""
    out = []
    for mouse, planes in cfg[key].items():
        for plane, dates in planes.items():
            for d in dates:
                fold = os.path.join(cfg["data_root"], f"{d}-OLL", mouse, plane)
                out.append(plane_from_dir(fold))
    return sorted(out, key=lambda p: p["pid"])


def training_planes(cfg):
    return _expand(cfg, "training_planes")


def test_planes(cfg):
    test = _expand(cfg, "test_planes")
    train_ids = {p["pid"] for p in training_planes(cfg)}
    missing = [p["pid"] for p in test if p["pid"] not in train_ids]
    if missing:
        raise ValueError(f"test_planes not in training_planes: {missing}")
    return test


def dev_planes(cfg):
    test_ids = {p["pid"] for p in test_planes(cfg)}
    return [p for p in training_planes(cfg) if p["pid"] not in test_ids]


def discover_planes(data_root):
    """Every <session>/<mouse>/<plane> folder with a results.hdf5."""
    hits = sorted(glob.glob(os.path.join(data_root, "*", "*", "plane*", HDF5_NAME)))
    return [plane_from_dir(os.path.dirname(h)) for h in hits]


def read_plane_list(path):
    with open(path) as f:
        dirs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [plane_from_dir(d) for d in dirs]


def task_slice(items, task_id, chunk):
    """Items handled by one SLURM array task."""
    if task_id is None:
        return items
    return items[task_id * chunk:(task_id + 1) * chunk]


# ── Label files (same format as caiman-reader's save_project) ───────────────

def read_labels(path):
    """[(roi_id, label), ...] in file order; label is '' for unreviewed."""
    with open(path, newline="") as f:
        return [(int(k), v.strip()) for k, v in csv.reader(f)]


def write_labels(path, rows):
    """Written exactly as caiman-reader's save_project: csv.writer, no header."""
    with open(path, "x", newline="") as f:   # "x": never overwrite
        w = csv.writer(f)
        for k, v in rows:
            w.writerow([k, v])


# ── HDF5 ─────────────────────────────────────────────────────────────────────

def _sparse(group):
    return csc_matrix((group["data"][()], group["indices"][()], group["indptr"][()]),
                      shape=tuple(group["shape"][()]))


def _quality_metric(est, fold, h5_key, npz_key, n):
    ds = est.get(h5_key)
    if isinstance(ds, h5py.Dataset) and ds.ndim == 1 and ds.dtype.kind == "f":
        return np.asarray(ds[()], dtype=float)
    npz = os.path.join(fold, "quality_scores.npz")
    if os.path.exists(npz):
        return np.asarray(np.load(npz)[npz_key], dtype=float)
    return np.full(n, np.nan)


def _parse_float_array(ds, n):
    """neurons_sn is saved as byte strings in the OLL outputs."""
    if not isinstance(ds, h5py.Dataset) or ds.ndim != 1 or ds.shape[0] != n:
        return np.full(n, np.nan)
    out = np.full(n, np.nan)
    for i, v in enumerate(ds[()]):
        try:
            out[i] = float(v.decode() if isinstance(v, bytes) else v)
        except (TypeError, ValueError):
            pass
    return out


def _frame_rate(h5, fold):
    try:
        return float(h5["params"]["data"]["fr"][()])
    except (KeyError, TypeError, ValueError):
        pass
    return run_parameter(fold, "fs", float)


def run_parameter(fold, key, cast=str):
    path = os.path.join(fold, "run_parameters.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            k, _, v = line.partition("=")
            if k.strip() == key:
                try:
                    return cast(v.split("#")[0].strip())
                except ValueError:
                    return None
    return None


def _footprint_order(A, dims, idx, fold):
    """'C' for OLL outputs (transposed memmap), 'F' for CaImAn's convention.
    Same test as caiman-reader.footprint_order."""
    H, W = dims
    masks_path = os.path.join(fold, "roi_masks.npy")
    if os.path.exists(masks_path):
        masks = np.load(masks_path, mmap_mode="r")
        if masks.ndim == 3 and masks.shape[:2] == (H, W) and masks.shape[2] and len(idx):
            covered = np.asarray((A[:, idx] != 0).sum(axis=1)).ravel() > 0
            target = np.asarray(masks).any(axis=2)
            match = {o: (covered.reshape(H, W, order=o) & target).sum() for o in "CF"}
            return "C" if match["C"] >= match["F"] else "F"
    return "C" if os.path.exists(os.path.join(fold, "run_parameters.txt")) else "F"


@dataclass
class Plane:
    fold: str
    pid: str
    session: str
    mouse: str
    plane: str
    A: csc_matrix          # (pixels, K), CaImAn pixel order
    dims: tuple            # (rows, cols) of the movie
    order: str             # pixel order of A ('C' or 'F')
    idx: np.ndarray        # idx_components (the reviewed candidates), file order
    C: np.ndarray          # (K, T)
    YrA: np.ndarray        # (K, T)
    F_dff: np.ndarray      # (K, T) or None
    b: np.ndarray          # (pixels, nb) or None
    f: np.ndarray          # (nb, T) or None
    snr: np.ndarray
    rval: np.ndarray
    noise: np.ndarray
    fs: float
    mean_img: np.ndarray   # (rows, cols) movie orientation, or None

    @property
    def K(self):
        return self.C.shape[0]

    def rc(self, pix):
        """CaImAn pixel indices -> (row, col) in the movie's orientation."""
        H, W = self.dims
        if self.order == "C":
            return pix // W, pix % W
        return pix % H, pix // H


def load_plane(fold):
    meta = plane_from_dir(fold)
    with h5py.File(os.path.join(fold, HDF5_NAME), "r") as h5:
        est = h5["estimates"]
        A = _sparse(est["A"])
        dims = tuple(int(d) for d in est["dims"][()])
        idx = np.asarray(est["idx_components"][()], dtype=int)
        C = np.asarray(est["C"][()], dtype=float)
        K = C.shape[0]
        YrA = np.asarray(est["YrA"][()], dtype=float)
        F_dff = est.get("F_dff")
        F_dff = np.asarray(F_dff[()], dtype=float) if isinstance(F_dff, h5py.Dataset) and F_dff.shape == C.shape else None
        b, f = est.get("b"), est.get("f")
        try:
            b = _sparse(b).toarray() if isinstance(b, h5py.Group) else np.asarray(b[()], dtype=float)
            f = np.asarray(f[()], dtype=float)
            b = b[:, None] if b.ndim == 1 else b
            f = f[None, :] if f.ndim == 1 else f
            if b.ndim != 2 or b.shape[0] != A.shape[0] or b.shape[1] != f.shape[0] or f.shape[1] != C.shape[1]:
                b = f = None
        except (TypeError, KeyError, ValueError):
            b = f = None
        snr = _quality_metric(est, fold, "SNR_comp", "snr", K)
        rval = _quality_metric(est, fold, "r_values", "rval", K)
        noise = _parse_float_array(est.get("neurons_sn"), K)
        fs = _frame_rate(h5, fold)
    order = _footprint_order(A, dims, idx, fold)
    plane = Plane(fold=fold, A=A, dims=dims, order=order, idx=idx, C=C, YrA=YrA, F_dff=F_dff,
                  b=b, f=f, snr=snr, rval=rval, noise=noise, fs=fs, mean_img=None,
                  **{k: meta[k] for k in ("pid", "session", "mouse", "plane")})
    plane.mean_img = _mean_image(plane)
    return plane


def _mean_image(plane):
    """ops.npy meanImg, in the movie orientation. On a square frame the
    orientation is ambiguous, so take whichever of img / img.T matches the
    footprints' coverage better."""
    path = os.path.join(plane.fold, "ops.npy")
    if not os.path.exists(path):
        return None
    try:
        img = np.asarray(np.load(path, allow_pickle=True).item()["meanImg"], dtype=float)
    except (KeyError, ValueError, AttributeError):
        return None
    H, W = plane.dims
    cands = [c for c in (img, img.T) if c.shape == (H, W)]
    if not cands:
        return None
    if len(cands) == 2:
        cov = np.asarray(plane.A[:, plane.idx].sum(axis=1)).ravel()
        r, c = plane.rc(np.arange(plane.A.shape[0]))
        cover = np.zeros((H, W))
        cover[r, c] = cov
        scores = [np.corrcoef(x.ravel(), cover.ravel())[0, 1] for x in cands]
        return cands[int(np.nanargmax(scores))]
    return cands[0]
