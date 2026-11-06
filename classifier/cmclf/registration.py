"""Cross-session FOV registration (xgboost-context, stage 1).

Every session of one mouse and plane is registered to a reference session:
1. Rigid: a rotation search, with phase correlation for the translation at
   each angle (the angle with the strongest correlation peak wins).
2. Nonrigid: phase correlation in overlapping blocks of the rigid-aligned
   image (piecewise rigid, as NoRMCorre / Suite2p do within a session),
   unreliable blocks filled in, smoothed, and interpolated to a dense field.
Registration runs on contrast-enhanced images (high-pass, local contrast
normalisation), so slow brightness differences between days don't matter.

Convention: a Transform is a *pull* map g from reference coordinates
(row, col) to the session's coordinates, so session_img(g(p)) ~ ref_img(p).
warp_image() samples a session image at g(p) to put it in reference space;
push() takes session points (e.g. centroids) to reference space (g inverse)."""
import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree

EPS = 1e-9


# ── Images ───────────────────────────────────────────────────────────────────

def enhance(img, sigma):
    """High-pass filter and local contrast normalisation, clipped to +-4."""
    img = np.nan_to_num(np.asarray(img, float))
    hp = img - ndimage.gaussian_filter(img, sigma)
    sd = np.sqrt(ndimage.gaussian_filter(hp ** 2, sigma))
    return np.clip(hp / (sd + 0.1 * np.median(sd) + EPS), -4, 4)


def tukey2d(shape, frac=0.15):
    def tukey(n):
        w = np.ones(n)
        m = max(int(frac * n), 1)
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        w[:m], w[n - m:] = ramp, ramp[::-1]
        return w
    return np.outer(tukey(shape[0]), tukey(shape[1]))


def phase_corr(a, b, max_shift=None):
    """Shift s = (row, col), subpixel, such that a(p + s) ~ b(p), and the peak
    strength (peak height in SDs of the correlation surface)."""
    R = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    R /= np.maximum(np.abs(R), EPS)
    corr = ndimage.gaussian_filter(np.real(np.fft.ifft2(R)), 1.0, mode="wrap")
    corr = np.fft.fftshift(corr)
    h, w = corr.shape
    cy, cx = h // 2, w // 2
    if max_shift is not None:
        m = int(np.ceil(max_shift))
        mask = np.full(corr.shape, False)
        mask[max(cy - m, 0):cy + m + 1, max(cx - m, 0):cx + m + 1] = True
        search = np.where(mask, corr, -np.inf)
    else:
        search = corr
    py, px = np.unravel_index(np.argmax(search), corr.shape)

    def vertex(c_m, c_0, c_p):
        den = c_m - 2 * c_0 + c_p
        return 0.0 if abs(den) < EPS else float(np.clip(0.5 * (c_m - c_p) / den, -0.5, 0.5))

    dy = vertex(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    dx = vertex(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    strength = (corr[py, px] - corr.mean()) / (corr.std() + EPS)
    return np.array([py - cy + dy, px - cx + dx]), float(strength)


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ── Transform ────────────────────────────────────────────────────────────────

class Transform:
    """g(p) = R(theta) (p + d(p) + shift - center) + center, with d the
    nonrigid field (zero when there is none)."""

    def __init__(self, dims, theta=0.0, shift=(0.0, 0.0), grid_y=None, grid_x=None, dy=None, dx=None):
        self.dims = tuple(int(v) for v in dims)
        self.theta = float(theta)
        self.shift = np.asarray(shift, float)
        self.center = (np.asarray(self.dims, float) - 1) / 2
        self.grid_y, self.grid_x, self.dy, self.dx = grid_y, grid_x, dy, dx
        self._splines = None
        if dy is not None:
            k_y, k_x = min(3, len(grid_y) - 1), min(3, len(grid_x) - 1)
            self._splines = (RectBivariateSpline(grid_y, grid_x, dy, kx=k_y, ky=k_x),
                             RectBivariateSpline(grid_y, grid_x, dx, kx=k_y, ky=k_x))

    @property
    def is_nonrigid(self):
        return self._splines is not None

    def rigid_only(self):
        return Transform(self.dims, self.theta, self.shift)

    def field(self, pts):
        """Nonrigid displacement d at reference points (N, 2); constant beyond
        the outermost block centres."""
        pts = np.atleast_2d(pts)
        if self._splines is None:
            return np.zeros_like(pts, dtype=float)
        y = np.clip(pts[:, 0], self.grid_y[0], self.grid_y[-1])
        x = np.clip(pts[:, 1], self.grid_x[0], self.grid_x[-1])
        return np.stack([self._splines[0].ev(y, x), self._splines[1].ev(y, x)], axis=1)

    def pull(self, pts):
        """Reference points -> session points."""
        pts = np.atleast_2d(np.asarray(pts, float))
        q = pts + self.field(pts) + self.shift - self.center
        return q @ _rot(self.theta).T + self.center

    def push(self, pts, n_iter=30):
        """Session points -> reference points (inverse of pull)."""
        u = np.atleast_2d(np.asarray(pts, float))
        v = (u - self.center) @ _rot(-self.theta).T + self.center - self.shift   # rigid inverse
        p = v.copy()
        if self._splines is not None:
            for _ in range(n_iter):
                p = v - self.field(p)
        return p

    def warp_image(self, img, order=1, cval=np.nan):
        """Session image -> reference space."""
        H, W = self.dims
        rr, cc = np.mgrid[:H, :W]
        src = self.pull(np.stack([rr.ravel(), cc.ravel()], axis=1))
        out = ndimage.map_coordinates(np.asarray(img, float), [src[:, 0], src[:, 1]],
                                      order=order, mode="constant", cval=cval)
        return out.reshape(H, W)

    def max_field(self):
        return 0.0 if self.dy is None else float(np.hypot(self.dy, self.dx).max())

    def to_dict(self):
        d = dict(dims=np.asarray(self.dims), theta=self.theta, shift=self.shift)
        if self.dy is not None:
            d.update(grid_y=self.grid_y, grid_x=self.grid_x, dy=self.dy, dx=self.dx)
        return d

    @classmethod
    def from_dict(cls, d):
        get = (lambda k: d[k] if k in d else None)
        return cls(tuple(d["dims"]), float(d["theta"]), d["shift"], get("grid_y"), get("grid_x"), get("dy"), get("dx"))

    def save(self, path, **extra):
        np.savez(path, **self.to_dict(), **extra)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls.from_dict({k: z[k] for k in z.files})


# ── Registration ─────────────────────────────────────────────────────────────

def rotate_pull(img, theta):
    """img sampled at R(theta)(q - c) + c (a rotation about the image centre)."""
    return Transform(img.shape, theta).warp_image(img, order=1, cval=0.0)


def rigid_register(ref_e, mov_e, max_shift, max_angle_deg, angle_step_deg):
    """Coarse angle grid, then a finer one around the best angle."""
    win = tukey2d(ref_e.shape)
    b = ref_e * win

    def at(theta):
        s, strength = phase_corr(rotate_pull(mov_e, theta) * win, b, max_shift)
        return strength, theta, s

    coarse = np.deg2rad(np.arange(-max_angle_deg, max_angle_deg + 1e-9, angle_step_deg)) if max_angle_deg > 0 \
        else np.zeros(1)
    best = max((at(t) for t in coarse), key=lambda r: r[0])
    if max_angle_deg > 0:
        step = np.deg2rad(angle_step_deg)
        fine = best[1] + np.linspace(-step, step, 9)
        best = max([best] + [at(t) for t in fine], key=lambda r: r[0])
    strength, theta, s = best
    return Transform(ref_e.shape, theta, s), strength


def block_starts(n, block, step):
    starts = list(range(0, max(n - block, 0) + 1, step))
    if starts[-1] != max(n - block, 0):
        starts.append(max(n - block, 0))
    return np.array(starts)


def _block_shifts(ref_e, warped, ys, xs, block, max_shift, min_strength):
    win = tukey2d((block, block))
    sy = np.zeros((len(ys), len(xs)))
    sx = np.zeros_like(sy)
    ok = np.zeros(sy.shape, bool)
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            a = warped[y0:y0 + block, x0:x0 + block]
            b = ref_e[y0:y0 + block, x0:x0 + block]
            s, strength = phase_corr(a * win, b * win, max_shift)
            sy[i, j], sx[i, j] = s
            ok[i, j] = strength >= min_strength and np.hypot(*s) <= max_shift
    return sy, sx, ok


def _clean_field(sy, sx, ok, outlier_px, smooth_blocks):
    """Fill unreliable blocks with the median of the reliable ones; replace
    blocks that disagree with their 3x3 neighbourhood median by more than
    outlier_px (a plain median filter would flatten real warps); then light
    Gaussian smoothing."""
    sy, sx = sy.copy(), sx.copy()
    sy[~ok], sx[~ok] = np.median(sy[ok]), np.median(sx[ok])
    my, mx = ndimage.median_filter(sy, 3, mode="nearest"), ndimage.median_filter(sx, 3, mode="nearest")
    out = np.hypot(sy - my, sx - mx) > outlier_px
    sy[out], sx[out] = my[out], mx[out]
    if smooth_blocks > 0:
        sy = ndimage.gaussian_filter(sy, smooth_blocks, mode="nearest")
        sx = ndimage.gaussian_filter(sx, smooth_blocks, mode="nearest")
    return sy, sx, int(out.sum())


def nonrigid_register(ref_e, rigid, mov_e, block, overlap, max_shift, min_strength, smooth_blocks,
                      n_iter=3, min_valid_frac=0.3, outlier_px=2.0):
    """Block-wise residual shifts on top of `rigid`, refined n_iter times (each
    pass measures the remaining shifts of the image warped with the current
    field and adds them). Returns (Transform, info)."""
    H, W = ref_e.shape
    block = min(block, H, W)
    step = max(int(round(block * (1 - overlap))), 1)
    ys, xs = block_starts(H, block, step), block_starts(W, block, step)
    gy, gx = ys + (block - 1) / 2, xs + (block - 1) / 2
    dy, dx = np.zeros((len(ys), len(xs))), np.zeros((len(ys), len(xs)))
    current = rigid
    info = dict(n_blocks=len(ys) * len(xs))
    for it in range(n_iter):
        warped = current.warp_image(mov_e, order=1, cval=0.0)
        sy, sx, ok = _block_shifts(ref_e, warped, ys, xs, block, max_shift, min_strength)
        if it == 0:
            info["n_valid_blocks"] = int(ok.sum())
            if ok.mean() < min_valid_frac:
                info["nonrigid"] = "too few reliable blocks"
                return rigid, info
        if ok.mean() < min_valid_frac:
            break
        sy, sx, n_out = _clean_field(sy, sx, ok, outlier_px, smooth_blocks if it == 0 else 0)
        if it == 0:
            info["n_outlier_blocks"] = n_out
        dy, dx = dy + sy, dx + sx
        current = Transform((H, W), rigid.theta, rigid.shift, gy, gx, dy, dx)
        info["last_update_px"] = float(np.hypot(sy, sx).max())
        if info["last_update_px"] < 0.05:
            break
    info["nonrigid"] = "ok"
    info["n_iter"] = it + 1
    return current, info


# ── QC ───────────────────────────────────────────────────────────────────────

def image_corr(ref_e, warped_e, border):
    H, W = ref_e.shape
    inner = np.zeros((H, W), bool)
    inner[border:H - border, border:W - border] = True
    m = inner & np.isfinite(warped_e)
    if m.sum() < 100:
        return np.nan
    return float(np.corrcoef(ref_e[m], warped_e[m])[0, 1])


def mutual_nearest(ref_pts, pts, radius):
    """Mutual nearest-neighbour pairs within radius: (i_ref, i_pts, distance)."""
    if len(ref_pts) == 0 or len(pts) == 0:
        return np.array([], int), np.array([], int), np.array([])
    d1, j = cKDTree(pts).query(ref_pts)
    _, i_back = cKDTree(ref_pts).query(pts)
    i = np.arange(len(ref_pts))
    keep = (i_back[j] == i) & (d1 <= radius)
    return i[keep], j[keep], d1[keep]


def centroid_qc(ref_pts, sess_pts, transform, radius):
    """Session centroids pushed into reference space, matched to the reference's."""
    _, _, d = mutual_nearest(ref_pts, transform.push(sess_pts) if len(sess_pts) else sess_pts, radius)
    n = min(len(ref_pts), len(sess_pts))
    return dict(n_pairs=int(len(d)), frac_matched=float(len(d) / n) if n else np.nan,
                median_dist=float(np.median(d)) if len(d) else np.nan)
