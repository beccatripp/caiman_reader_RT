"""G9: movie-based "shivering" features (optional; one pass over the movie).

The movie is the one caiman-reader shows: a *preprocessed.tif in the plane
folder, else the input_tif in run_parameters.txt (the deepCAD TIFF CaImAn
segmented). Frames are read in chunks, so memory stays at chunk_frames frames.

- Core vs rim: mean signal in the eroded ROI core and in a 3-px ring outside it.
  Motion of a bright structure gives fast, opposite-sign core/rim changes.
- Frame shift: rigid shift of each frame against the first chunk's mean, by
  phase correlation on a central crop; ROI signal that tracks the shift is
  motion, not activity.
- Local correlation image (Cn): each pixel's mean temporal correlation with its
  8 neighbours, averaged inside the ROI and in a surrounding ring."""
import os

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csc_matrix

from .features import EPS, corr_rows, footprint_geometry
from .io import run_parameter

MOVIE_FEATURES = ["mv_core_rim_corr", "mv_rim_core_std_ratio", "mv_shift_corr", "mv_shift_speed_corr",
                  "mv_cn_roi", "mv_cn_contrast", "mv_shift_std", "mv_shift_speed_mean"]

OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


def find_movie(fold):
    for f in sorted(os.listdir(fold)):
        if f.lower().endswith(("preprocessed.tif", "preprocessed.tiff")):
            return os.path.join(fold, f)
    return run_parameter(fold, "input_tif")


def iter_chunks(path, chunk):
    try:
        # Uncompressed, contiguous TIFF: map the file itself (no copy)
        mm = tifffile.memmap(path, mode="r")
    except (ValueError, TypeError):
        mm = None
    if mm is not None and mm.ndim == 3:
        for s in range(0, mm.shape[0], chunk):
            yield np.asarray(mm[s:s + chunk], dtype=np.float32)
        return
    with tifffile.TiffFile(path) as tif:
        buf = []
        for page in tif.pages:
            buf.append(page.asarray().astype(np.float32))
            if len(buf) == chunk:
                yield np.stack(buf)
                buf = []
        if buf:
            yield np.stack(buf)


def core_rim_masks(geom, idx, dims, union):
    H, W = dims
    core_q, core_k, rim_q, rim_k, ring_q, ring_k = [], [], [], [], [], []
    for i, k in enumerate(idx):
        r, c, _ = geom["pix_rc"][k]
        if len(r) == 0:
            continue
        r0, r1 = max(r.min() - 12, 0), min(r.max() + 13, H)
        c0, c1 = max(c.min() - 12, 0), min(c.max() + 13, W)
        crop = np.zeros((r1 - r0, c1 - c0), bool)
        crop[r - r0, c - c0] = True
        inside = distance_transform_edt(crop)
        outside = distance_transform_edt(~crop)
        others = union[r0:r1, c0:c1] & ~crop
        core = inside > 2
        if not core.any():
            core = crop
        rim = (outside > 0) & (outside <= 3) & ~others
        ring = (outside > 2) & (outside <= 10) & ~others
        for m, qs, ks in ((core, core_q, core_k), (rim, rim_q, rim_k), (ring, ring_q, ring_k)):
            rr, cc = np.nonzero(m)
            qs.append((rr + r0) * W + cc + c0)
            ks.append(np.full(len(rr), i))

    def mat(qs, ks):
        q, k = np.concatenate(qs), np.concatenate(ks)
        m = csc_matrix((np.ones(len(q), np.float32), (q, k)), shape=(H * W, len(idx)))
        return m.multiply(1.0 / np.maximum(np.asarray(m.sum(axis=0)).ravel(), 1)).tocsc()
    return mat(core_q, core_k), mat(rim_q, rim_k), mat(ring_q, ring_k)


def offset_slices(dy, dx, H, W):
    """Slices pairing each pixel with its (dy, dx) neighbour; dy >= 0."""
    if dx >= 0:
        return np.s_[:H - dy, :W - dx], np.s_[dy:, dx:]
    return np.s_[:H - dy, -dx:], np.s_[dy:, :W + dx]


def phase_shift(frame, ref_fft):
    F = np.fft.fft2(frame)
    R = F * np.conj(ref_fft)
    R /= np.maximum(np.abs(R), EPS)
    corr = np.abs(np.fft.ifft2(R))
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = corr.shape
    return (dy - h if dy > h // 2 else dy), (dx - w if dx > w // 2 else dx)


def extract_movie_features(plane, mcfg, fcfg):
    path = find_movie(plane.fold)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"no movie for {plane.fold}")
    H, W = plane.dims
    idx = plane.idx
    geom = footprint_geometry(plane, fcfg["mask_threshold"])
    union = (np.asarray(geom["M"].sum(axis=1)).ravel() > 0).reshape(H, W)
    Mcore, Mrim, Mring = core_rim_masks(geom, idx, plane.dims, union)
    crop = mcfg["shift_crop_px"]
    y0, x0 = max((H - crop) // 2, 0), max((W - crop) // 2, 0)

    core, rim, shifts = [], [], []
    S1 = np.zeros((H, W))
    S2 = np.zeros((H, W))
    SX = {o: np.zeros((H - o[0], W - abs(o[1]))) for o in OFFSETS}
    ref_fft = None
    T = 0
    for X in iter_chunks(path, mcfg["chunk_frames"]):
        if X.shape[1:] != (H, W):
            raise ValueError(f"movie frame {X.shape[1:]} != dims {plane.dims}")
        flat = X.reshape(len(X), -1)
        core.append(np.asarray(Mcore.T @ flat.T).T)
        rim.append(np.asarray(Mrim.T @ flat.T).T)
        crops = X[:, y0:y0 + crop, x0:x0 + crop]
        if ref_fft is None:
            ref_fft = np.fft.fft2(crops.mean(axis=0))
        shifts.extend(phase_shift(fr, ref_fft) for fr in crops)
        Xd = X.astype(np.float64)
        S1 += Xd.sum(axis=0)
        S2 += (Xd ** 2).sum(axis=0)
        for o, acc in SX.items():
            sa, sb = offset_slices(*o, H, W)
            acc += (Xd[(slice(None),) + sa] * Xd[(slice(None),) + sb]).sum(axis=0)
        T += len(X)

    core = np.concatenate(core).T   # (n, T)
    rim = np.concatenate(rim).T
    shifts = np.asarray(shifts, float)
    mag = np.hypot(shifts[:, 0], shifts[:, 1])
    speed = np.hypot(*np.diff(shifts, axis=0).T)

    mean = S1 / T
    sd = np.sqrt(np.maximum(S2 / T - mean ** 2, 0))
    cn_sum = np.zeros((H, W))
    cn_n = np.zeros((H, W))
    for o, acc in SX.items():
        sa, sb = offset_slices(*o, H, W)
        c =(acc / T - mean[sa] * mean[sb]) / np.maximum(sd[sa] * sd[sb], EPS)
        cn_sum[sa] += c
        cn_n[sa] += 1
        cn_sum[sb] += c
        cn_n[sb] += 1
    cn = (cn_sum / np.maximum(cn_n, 1)).ravel()

    dcore, drim = np.diff(core, axis=1), np.diff(rim, axis=1)
    out = pd.DataFrame({"roi_id": idx})
    out["mv_core_rim_corr"] = corr_rows(dcore, drim)
    out["mv_rim_core_std_ratio"] = drim.std(axis=1) / np.maximum(dcore.std(axis=1), EPS)
    out["mv_shift_corr"] = np.abs(corr_rows(core, np.tile(mag, (len(idx), 1)))) if mag.std() > 0 else 0.0
    out["mv_shift_speed_corr"] = (corr_rows(np.abs(dcore), np.tile(speed, (len(idx), 1)))
                                  if speed.std() > 0 else 0.0)
    Mroi = geom["M"][:, idx]
    cn_roi = np.asarray(Mroi.T @ cn).ravel() / np.maximum(np.asarray(Mroi.sum(axis=0)).ravel(), 1)
    out["mv_cn_roi"] = cn_roi
    out["mv_cn_contrast"] = cn_roi - np.asarray(Mring.T @ cn).ravel()
    out["mv_shift_std"] = mag.std()
    out["mv_shift_speed_mean"] = speed.mean() if len(speed) else np.nan
    return out.replace([np.inf, -np.inf], np.nan)
