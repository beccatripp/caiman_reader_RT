"""Per-ROI features for one plane: groups G1-G8 and the previous student's
28-feature set (the "replica", columns prefixed rep_).

One row per CaImAn-accepted component (estimates/idx_components), which is the
pool reviewed in caiman-reader. Neighbour features search all components,
including CaImAn rejects, since a ghost's real partner may have been rejected.
Footprints are thresholded at mask_threshold x peak (the pipeline's
roi_mask_threshold) and placed in the movie's orientation (Plane.rc)."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy.ndimage import label as ndlabel
from scipy.signal import find_peaks
from scipy.sparse import csc_matrix
from scipy.spatial import ConvexHull, QhullError

EPS = 1e-6

FEATURE_GROUPS = {
    "G1_caiman": ["snr", "r_value", "noise_sn"],
    "G2_shape": ["area", "area_rel", "fp_max", "ecc", "compactness", "solidity", "n_islands", "spread"],
    "G3_location": ["dist_center_norm", "dist_bright_center_norm", "dist_edge_px", "radial_rank"],
    "G4_brightness": ["img_roi_rel", "img_annulus_rel", "img_contrast", "bg_rel"],
    "G5_signal": ["dff_peak", "dff_p95", "robust_snr", "z_peak", "skew", "kurt", "events_per_min",
                  "autocorr1", "resid_noise_dff", "resid_to_range", "hf_power_frac"],
    "G6_neighbor": ["nbr_max_corr", "nbr_max_corr_C", "nbr_dist", "nbr_dff_ratio", "nbr_z_ratio",
                    "nbr_bright_ratio", "nbr_area_ratio", "nbr_snr_ratio", "nbr_is_accepted",
                    "n_corr_nbrs", "max_dice"],
    "G7_global": ["bg_f_corr", "pop_corr", "pop_diff_corr", "pop_rhythm", "plane_mean_corr"],
    "G8_context": ["plane_median_snr", "plane_median_rval", "plane_median_area", "n_accepted",
                   "caiman_accept_frac", "snr_rank", "dff_rank", "area_rank", "rsnr_rank", "bright_rank"],
}
CORE_FEATURES = [c for g in FEATURE_GROUPS.values() for c in g]

# The previous student's 28 features (maze-runners S03), recomputed on this data:
# A_thr -> the thresholded footprint, session stats -> per plane.
REPLICA_FEATURES = [
    "rep_snr", "rep_r_value", "rep_noise_sn",
    "rep_footprint_area", "rep_footprint_max", "rep_footprint_ecc", "rep_footprint_n_islands",
    "rep_distance_from_center_norm",
    "rep_trace_skew", "rep_trace_kurt", "rep_trace_max_dff", "rep_trace_n_peaks", "rep_trace_autocorr",
    "rep_session_snr_mean", "rep_session_snr_median", "rep_session_snr_std",
    "rep_session_rvalue_mean", "rep_session_rvalue_median", "rep_session_rvalue_std", "rep_session_noise_mean",
    "rep_snr_noise_ratio", "rep_trace_cv_proxy", "rep_autocorr_squared", "rep_footprint_compact",
    "rep_peak_burstiness", "rep_has_better_overlap_neighbor", "rep_neighbor_trace_correlation",
    "rep_population_rhythm",
]

ID_COLUMNS = ["pid", "session", "mouse", "plane", "roi_id", "order_idx"]


# ── Footprints ───────────────────────────────────────────────────────────────

def footprint_geometry(plane, thr):
    """Thresholded masks of all K components as a (H*W, K) sparse matrix in the
    movie orientation (flat index row*W + col), plus per-component area, peak,
    weighted centroid and moments."""
    H, W = plane.dims
    A = plane.A
    K = A.shape[1]
    rows_l, cols_l = [], []
    area = np.zeros(K)
    fp_max = np.zeros(K)
    cy = np.full(K, H / 2.0)
    cx = np.full(K, W / 2.0)
    ecc = np.ones(K)
    spread = np.zeros(K)
    pix_rc = []
    for k in range(K):
        s, e = A.indptr[k], A.indptr[k + 1]
        pix, v = A.indices[s:e], A.data[s:e]
        if len(v) == 0 or v.max() <= 0:
            pix_rc.append((np.array([], int), np.array([], int), np.array([])))
            continue
        keep = v >= thr * v.max()
        r, c = plane.rc(pix[keep])
        w = v[keep]
        pix_rc.append((r, c, w))
        rows_l.append(r * W + c)
        cols_l.append(np.full(len(r), k))
        area[k] = len(r)
        fp_max[k] = v.max()
        cy[k] = (r * w).sum() / w.sum()
        cx[k] = (c * w).sum() / w.sum()
        if len(r) > 5:
            mu20 = (w * (c - cx[k]) ** 2).sum() / w.sum()
            mu02 = (w * (r - cy[k]) ** 2).sum() / w.sum()
            mu11 = (w * (c - cx[k]) * (r - cy[k])).sum() / w.sum()
            root = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
            lam1 = (mu20 + mu02) / 2 + root
            lam2 = (mu20 + mu02) / 2 - root
            ecc[k] = np.sqrt(max(1 - lam2 / (lam1 + 1e-9), 0))
            spread[k] = np.sqrt(mu20 + mu02)
    q = np.concatenate(rows_l) if rows_l else np.array([], int)
    kk = np.concatenate(cols_l) if cols_l else np.array([], int)
    M = csc_matrix((np.ones(len(q), dtype=np.float32), (q, kk)), shape=(H * W, K))
    return dict(M=M, area=area, fp_max=fp_max, cy=cy, cx=cx, ecc=ecc, spread=spread, pix_rc=pix_rc)


def shape_details(r, c):
    """Perimeter-based compactness, solidity and island count of one mask."""
    area = len(r)
    if area == 0:
        return np.nan, np.nan, 0
    r0, c0 = r.min() - 1, c.min() - 1
    crop = np.zeros((r.max() - r0 + 2, c.max() - c0 + 2), bool)
    crop[r - r0, c - c0] = True
    perim = (crop & ~binary_erosion(crop)).sum()
    compactness = 4 * np.pi * area / max(perim, 1) ** 2
    try:
        pts = np.concatenate([np.stack([r + dr, c + dc], 1) for dr in (-.5, .5) for dc in (-.5, .5)])
        solidity = area / ConvexHull(pts).volume
    except (QhullError, ValueError):
        solidity = 1.0
    n_islands = ndlabel(crop)[1]
    return compactness, solidity, n_islands


def annulus_brightness(img, r, c, union, inner, outer):
    """Mean image in a ring around the mask, excluding every component's mask."""
    H, W = img.shape
    r0, r1 = max(r.min() - outer - 1, 0), min(r.max() + outer + 2, H)
    c0, c1 = max(c.min() - outer - 1, 0), min(c.max() + outer + 2, W)
    crop = np.zeros((r1 - r0, c1 - c0), bool)
    crop[r - r0, c - c0] = True
    d = distance_transform_edt(~crop)
    ring = (d > inner) & (d <= outer) & ~union[r0:r1, c0:c1]
    return img[r0:r1, c0:c1][ring].mean() if ring.any() else np.nan


# ── Traces ───────────────────────────────────────────────────────────────────

def dff_traces(plane, pct):
    """dF/F as caiman-reader / traces_dff.npy (CaImAn detrend_df_f, whole-session
    baseline): (F - Fd) / (Df + Fd), F = ||A|| C, Df the baseline of the
    background under the cell. Returns dff, raw_dff (the same transform of
    C + YrA, same baselines), Fd, Df, nA (all components)."""
    A, C = plane.A, plane.C
    raw = C + plane.YrA
    nA = np.sqrt(np.ravel(A.power(2).sum(axis=0)))
    nA = np.maximum(nA, np.finfo(np.float32).eps)
    if plane.b is not None:
        B = (np.asarray(A.T @ plane.b) / nA[:, None]) @ plane.f
        Fd = np.nanpercentile(nA[:, None] * C, pct, axis=1)
        Df = np.nanpercentile(B, pct, axis=1)
        denom = np.maximum(Df + Fd, EPS)[:, None]
        dff = (nA[:, None] * C - Fd[:, None]) / denom
        raw_dff = (nA[:, None] * raw - Fd[:, None]) / denom
    else:
        F0 = np.nanpercentile(C, pct, axis=1)
        F0 = np.where(np.isfinite(F0) & (F0 > EPS), F0, EPS)[:, None]
        dff, raw_dff = (C - F0) / F0, (raw - F0) / F0
        Fd, Df = F0.ravel() * nA, np.full(len(F0), np.nan)
    return dff, raw_dff, Fd, Df, nA


def standardize(X):
    X = X - X.mean(axis=1, keepdims=True)
    return X / np.maximum(X.std(axis=1, keepdims=True), EPS)


def corr_rows(X, Y):
    """Row-wise Pearson correlation between matching rows of X and Y."""
    return (standardize(X) * standardize(Y)).mean(axis=1)


def hf_power_fraction(raw, fs, cutoff):
    if not fs or fs / 2 <= cutoff:
        return np.full(raw.shape[0], np.nan)
    P = np.abs(np.fft.rfft(raw - raw.mean(axis=1, keepdims=True), axis=1)) ** 2
    freqs = np.fft.rfftfreq(raw.shape[1], 1 / fs)
    return P[:, freqs > cutoff].sum(axis=1) / np.maximum(P[:, 1:].sum(axis=1), EPS)


def population_rhythm(traces):
    """Previous student's population rhythm: correlation of |z(trace)| with |PC1|
    of the candidates' population activity (0 with 5 or fewer cells)."""
    n = traces.shape[0]
    if n <= 5:
        return np.zeros(n)
    X = traces.T - traces.T.mean(axis=0)
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    motion = np.abs(u[:, 0] * s[0])
    motion = (motion - motion.mean()) / (motion.std() + EPS)
    z = (traces - traces.mean(axis=1, keepdims=True)) / (traces.std(axis=1, keepdims=True) + EPS)
    return np.clip(corr_rows(np.abs(z), np.tile(np.abs(motion), (n, 1))), -1, 1)


# ── Plane features ───────────────────────────────────────────────────────────

def extract_plane_features(plane, fcfg):
    idx = plane.idx
    n, K = len(idx), plane.K
    H, W = plane.dims
    T = plane.C.shape[1]
    fs = plane.fs
    geom = footprint_geometry(plane, fcfg["mask_threshold"])
    area, M = geom["area"], geom["M"]
    union = (np.asarray(M.sum(axis=1)).ravel() > 0).reshape(H, W)

    dff, raw_dff, Fd, Df, nA = dff_traces(plane, fcfg["baseline_percentile"])
    z = standardize(dff)
    raw = plane.C + plane.YrA
    Zraw = standardize(raw)
    corr_raw = Zraw @ Zraw.T / T
    ZC = standardize(plane.C)
    corr_C = ZC @ ZC.T / T

    out = pd.DataFrame({"roi_id": idx, "order_idx": np.arange(n)})
    for k in ("pid", "session", "mouse", "plane"):
        out[k] = getattr(plane, k)

    # G1
    snr = np.where(np.isfinite(plane.snr), plane.snr, np.nan)
    out["snr"] = snr[idx]
    out["r_value"] = plane.rval[idx]
    out["noise_sn"] = plane.noise[idx]

    # G2
    out["area"] = area[idx]
    out["area_rel"] = area[idx] / max(np.median(area[idx]), 1)
    out["fp_max"] = geom["fp_max"][idx]
    out["ecc"] = geom["ecc"][idx]
    out["spread"] = geom["spread"][idx]
    det = np.array([shape_details(*geom["pix_rc"][k][:2]) for k in idx]).reshape(-1, 3)
    out["compactness"], out["solidity"], out["n_islands"] = det[:, 0], det[:, 1], det[:, 2]

    # G3
    cy, cx = geom["cy"], geom["cx"]
    half = np.hypot(H / 2.0, W / 2.0)
    dist_center = np.hypot(cy - H / 2.0, cx - W / 2.0)
    out["dist_center_norm"] = dist_center[idx] / half
    out["dist_edge_px"] = np.minimum.reduce([cy[idx], cx[idx], H - 1 - cy[idx], W - 1 - cx[idx]])
    out["radial_rank"] = pd.Series(dist_center[idx]).rank(pct=True).values

    # G4
    img = plane.mean_img
    if img is not None:
        img_med = max(np.median(img), EPS)
        flat = img.ravel()
        img_roi = np.asarray(M.T @ flat).ravel() / np.maximum(area, 1)
        ann = np.array([annulus_brightness(img, *geom["pix_rc"][k][:2], union,
                                           fcfg["annulus_inner_px"], fcfg["annulus_outer_px"])
                        if area[k] else np.nan for k in idx])
        out["img_roi_rel"] = img_roi[idx] / img_med
        out["img_annulus_rel"] = ann / img_med
        out["img_contrast"] = (img_roi[idx] - ann) / (np.abs(ann) + EPS)
        sm = gaussian_filter(img, 25)
        w = np.clip(sm - np.median(sm), 0, None)
        if w.sum() > 0:
            rr, cc = np.mgrid[:H, :W]
            by, bx = (rr * w).sum() / w.sum(), (cc * w).sum() / w.sum()
            out["dist_bright_center_norm"] = np.hypot(cy[idx] - by, cx[idx] - bx) / half
        else:
            out["dist_bright_center_norm"] = np.nan
    else:
        img_roi = np.full(K, np.nan)
        for c in ("img_roi_rel", "img_annulus_rel", "img_contrast", "dist_bright_center_norm"):
            out[c] = np.nan
    # Background baseline under the cell. (C's own baseline is removed by
    # CaImAn, so Fd is ~0 and is not usable as a brightness measure.)
    out["bg_rel"] = Df[idx] / max(np.nanmedian(Df[idx]), EPS) if np.isfinite(Df[idx]).any() else np.nan

    # G5: shape of the reader's dF/F (from the denoised C); noise-based SNR from
    # the raw dF/F (C + YrA), since C is exactly 0 in quiet stretches
    d, rd = dff[idx], raw_dff[idx]
    p99 = np.percentile(rd, 99, axis=1)
    med = np.median(rd, axis=1)
    mad = 1.4826 * np.median(np.abs(rd - med[:, None]), axis=1)
    dff_peak = d.max(axis=1)
    z_peak = z[idx].max(axis=1)
    out["dff_peak"] = dff_peak
    out["dff_p95"] = np.percentile(d, 95, axis=1)
    out["robust_snr"] = (p99 - med) / np.maximum(mad, EPS)
    out["z_peak"] = z_peak
    out["skew"] = stats.skew(d, axis=1)
    out["kurt"] = stats.kurtosis(d, axis=1)
    dist_frames = max(1, int(round(fs))) if fs else 7
    minutes = T / fs / 60 if fs else np.nan
    out["events_per_min"] = [len(find_peaks(zz, height=fcfg["event_z"], distance=dist_frames)[0]) / minutes
                             for zz in z[idx]]
    out["autocorr1"] = corr_rows(d[:, :-1], d[:, 1:])
    yra_sd = plane.YrA[idx].std(axis=1)
    out["resid_noise_dff"] = (rd - d).std(axis=1)
    Ci = plane.C[idx]
    out["resid_to_range"] = yra_sd / np.maximum(np.percentile(Ci, 99, axis=1) - np.percentile(Ci, 1, axis=1), EPS)
    out["hf_power_frac"] = hf_power_fraction(raw[idx], fs, fcfg["hf_cutoff_hz"])

    # G6: neighbours among all K components
    dmat = np.hypot(cy[:, None] - cy[None, :], cx[:, None] - cx[None, :])
    inter = np.asarray((M.T @ M).todense())
    asum = area[:, None] + area[None, :]
    dice = np.where(asum > 0, 2 * inter / np.maximum(asum, 1), 0)
    accepted = np.zeros(K, bool)
    accepted[idx] = True
    peak_all = dff.max(axis=1)
    zpeak_all = z.max(axis=1)
    R = fcfg["neighbor_radius_px"]
    nb = {c: np.full(n, np.nan) for c in FEATURE_GROUPS["G6_neighbor"]}
    for i, k in enumerate(idx):
        others = np.flatnonzero((dmat[k] < R) & (np.arange(K) != k) & (area > 0))
        dk = dice[k].copy()
        dk[k] = 0
        nb["max_dice"][i] = dk.max() if K > 1 else 0
        if len(others) == 0:
            nb["n_corr_nbrs"][i] = 0
            continue
        cr = corr_raw[k, others]
        j = others[np.argmax(cr)]
        nb["nbr_max_corr"][i] = cr.max()
        nb["nbr_max_corr_C"][i] = corr_C[k, others].max()
        nb["nbr_dist"][i] = dmat[k, j]
        nb["nbr_dff_ratio"][i] = peak_all[k] / max(abs(peak_all[j]), EPS)
        nb["nbr_z_ratio"][i] = zpeak_all[k] / max(abs(zpeak_all[j]), EPS)
        nb["nbr_bright_ratio"][i] = img_roi[k] / img_roi[j] if np.isfinite(img_roi[j]) and img_roi[j] > 0 else np.nan
        nb["nbr_area_ratio"][i] = area[k] / max(area[j], 1)
        nb["nbr_snr_ratio"][i] = snr[k] / snr[j] if np.isfinite(snr[j]) and snr[j] > 0 else np.nan
        nb["nbr_is_accepted"][i] = float(accepted[j])
        nb["n_corr_nbrs"][i] = (cr > fcfg["neighbor_corr_thresh"]).sum()
    for c, v in nb.items():
        out[c] = v

    # G7
    if plane.f is not None:
        out["bg_f_corr"] = np.abs(standardize(raw[idx]) @ standardize(plane.f).T / T).max(axis=1)
    else:
        out["bg_f_corr"] = np.nan
    P = raw[idx]
    tot = P.sum(axis=0)
    out["pop_corr"] = corr_rows(P, (tot[None, :] - P) / max(n - 1, 1)) if n > 1 else np.nan
    dP = np.diff(P, axis=1)
    dtot = dP.sum(axis=0)
    out["pop_diff_corr"] = corr_rows(dP, (dtot[None, :] - dP) / max(n - 1, 1)) if n > 1 else np.nan
    trace_src = plane.F_dff if plane.F_dff is not None else dff
    out["pop_rhythm"] = population_rhythm(trace_src[idx])
    sub = corr_raw[np.ix_(idx, idx)]
    out["plane_mean_corr"] = (sub.sum() - n) / max(n * (n - 1), 1)

    # G8
    out["plane_median_snr"] = np.nanmedian(snr[idx])
    out["plane_median_rval"] = np.nanmedian(plane.rval[idx])
    out["plane_median_area"] = np.median(area[idx])
    out["n_accepted"] = n
    out["caiman_accept_frac"] = n / K
    for src, dst in (("snr", "snr_rank"), ("dff_peak", "dff_rank"), ("area", "area_rank"),
                     ("robust_snr", "rsnr_rank"), ("img_roi_rel", "bright_rank")):
        out[dst] = out[src].rank(pct=True)

    add_replica(out, plane, trace_src, geom, inter, area, snr)
    return out.replace([np.inf, -np.inf], np.nan), dict(geom=geom, dff=dff)


def add_replica(out, plane, trace_src, geom, inter, area, snr):
    idx = plane.idx
    out["rep_snr"] = out["snr"]
    out["rep_r_value"] = out["r_value"]
    out["rep_noise_sn"] = out["noise_sn"]
    out["rep_footprint_area"] = out["area"]
    out["rep_footprint_max"] = out["fp_max"]
    out["rep_footprint_ecc"] = out["ecc"]
    out["rep_footprint_n_islands"] = out["n_islands"]
    out["rep_distance_from_center_norm"] = out["dist_center_norm"]
    rows = []
    for t in trace_src[idx]:
        if t.std() < 1e-9:
            rows.append((0., 0., 0., 0, 0.))
            continue
        peaks, _ = find_peaks(t, height=t.mean() + 2 * t.std(), distance=10)
        rows.append((stats.skew(t), stats.kurtosis(t), t.max(), len(peaks), np.corrcoef(t[:-1], t[1:])[0, 1]))
    rows = np.array(rows, dtype=float).reshape(-1, 5)
    for i, c in enumerate(("skew", "kurt", "max_dff", "n_peaks", "autocorr")):
        out[f"rep_trace_{c}"] = rows[:, i]
    for name, col in (("snr", "snr"), ("rvalue", "r_value")):
        out[f"rep_session_{name}_mean"] = out[col].mean()
        out[f"rep_session_{name}_median"] = out[col].median()
        out[f"rep_session_{name}_std"] = out[col].std()
    out["rep_session_noise_mean"] = out["noise_sn"].mean()
    out["rep_snr_noise_ratio"] = out["snr"] / (out["noise_sn"] + 1e-6)
    out["rep_trace_cv_proxy"] = out["rep_trace_max_dff"] / (out["noise_sn"] + 1e-6)
    out["rep_autocorr_squared"] = out["rep_trace_autocorr"] ** 2
    out["rep_footprint_compact"] = 1.0 - out["ecc"]
    out["rep_peak_burstiness"] = (out["rep_trace_n_peaks"] + 1) / (out["rep_trace_max_dff"] + 1e-6)
    # Jaccard > 0.40 overlap with a candidate better on both SNR and r_value
    sub = inter[np.ix_(idx, idx)]
    a = area[idx]
    jac = sub / np.maximum(a[:, None] + a[None, :] - sub, 1)
    np.fill_diagonal(jac, 0)
    s, r = snr[idx], plane.rval[idx]
    better = (jac > 0.40) & (s[None, :] > s[:, None]) & (r[None, :] > r[:, None])
    deficit = ((s[None, :] - s[:, None]) / (s[:, None] + 1e-6) + (r[None, :] - r[:, None]) / (r[:, None] + 1e-6)) / 2
    out["rep_has_better_overlap_neighbor"] = better.any(axis=1).astype(float)
    out["rep_neighbor_trace_correlation"] = np.where(better, deficit, 0).max(axis=1)
    out["rep_population_rhythm"] = out["pop_rhythm"]


# ── Sanity checks against the pipeline's own outputs ────────────────────────

def plane_checks(plane, geom, dff):
    """Dice of our masks with roi_masks.npy and correlation of our dF/F with
    traces_dff.npy, per accepted component (both files are in idx order)."""
    import os
    res = {"pid": plane.pid, "order": plane.order, "n_components": int(plane.K),
           "n_accepted": int(len(plane.idx)), "has_mean_img": plane.mean_img is not None,
           "has_background": plane.b is not None, "fs": plane.fs}
    H, W = plane.dims
    mpath = os.path.join(plane.fold, "roi_masks.npy")
    if os.path.exists(mpath):
        masks = np.load(mpath)
        if masks.ndim == 3 and masks.shape[:2] == (H, W) and masks.shape[2] == len(plane.idx):
            P = csc_matrix(masks.reshape(H * W, -1).astype(np.float32))
            Mi = geom["M"][:, plane.idx]
            inter = np.asarray(P.multiply(Mi).sum(axis=0)).ravel()
            tot = np.asarray(P.sum(axis=0)).ravel() + np.asarray(Mi.sum(axis=0)).ravel()
            dice = 2 * inter / np.maximum(tot, 1)
            res.update(mask_dice_median=float(np.median(dice)), mask_dice_min=float(dice.min()))
    tpath = os.path.join(plane.fold, "traces_dff.npy")
    if os.path.exists(tpath):
        td = np.load(tpath)
        if td.shape == (len(plane.idx), dff.shape[1]):
            c = corr_rows(td.astype(float), dff[plane.idx])
            res.update(dff_corr_median=float(np.nanmedian(c)), dff_corr_min=float(np.nanmin(c)))
    return res
