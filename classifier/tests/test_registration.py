"""Registration on synthetic images with a known rotation, shift and smooth warp.

  python -m pytest classifier/tests        (or: python classifier/tests/test_registration.py)"""
import os
import sys
import tempfile

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cmclf import registration as reg

H = W = 512
BORDER = 60


def scene(seed=0):
    """Band-limited 'FOV': blobs about a cell wide on a slow gradient."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W))
    ys, xs = rng.integers(0, H, 600), rng.integers(0, W, 600)
    img[ys, xs] = rng.lognormal(0, 0.5, 600)
    img = ndimage.gaussian_filter(img, 4) * 200
    rr, cc = np.mgrid[:H, :W]
    return img + 0.5 * np.exp(-((rr - 200) ** 2 + (cc - 300) ** 2) / (2 * 180 ** 2))


def h_true(u, theta_deg=1.5, t=(7.3, -12.6), amp=2.5):
    """Session -> reference map: rotation + shift + smooth sinusoidal warp."""
    c = (np.array([H, W]) - 1) / 2
    R = reg._rot(np.deg2rad(theta_deg))
    v = (u - c) @ R.T + c + np.asarray(t)
    warp = amp * np.stack([np.sin(2 * np.pi * u[:, 1] / 400), np.cos(2 * np.pi * u[:, 0] / 350)], axis=1)
    return v + warp


def make_session(ref, **kw):
    rr, cc = np.mgrid[:H, :W]
    u = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(float)
    src = h_true(u, **kw)
    mov = ndimage.map_coordinates(ref, [src[:, 0], src[:, 1]], order=3, mode="nearest").reshape(H, W)
    return mov + np.random.default_rng(1).normal(0, 0.02, (H, W))


def interior_points(n=400, seed=2):
    rng = np.random.default_rng(seed)
    return rng.uniform(BORDER, H - BORDER, (n, 2))


def register(ref, mov):
    ref_e, mov_e = reg.enhance(ref, 12), reg.enhance(mov, 12)
    rigid, _ = reg.rigid_register(ref_e, mov_e, max_shift=80, max_angle_deg=12, angle_step_deg=1.0)
    nonrigid, info = reg.nonrigid_register(ref_e, rigid, mov_e, block=128, overlap=0.5, max_shift=12,
                                           min_strength=6, smooth_blocks=0.5)
    return rigid, nonrigid, info


def test_phase_corr_sign():
    b = ndimage.gaussian_filter(np.random.default_rng(0).normal(size=(128, 128)), 2)
    a = np.roll(b, (5, -3), axis=(0, 1))                 # a(p) = b(p - (5, -3))
    s, strength = reg.phase_corr(a, b)
    assert np.allclose(s, (5, -3), atol=0.1), s          # a(p + s) = b(p)
    assert strength > 10


def test_rigid_only():
    ref = scene()
    mov = make_session(ref, amp=0.0)
    rigid, _, _ = register(ref, mov)
    u = interior_points()
    err = np.hypot(*(rigid.push(u) - h_true(u, amp=0.0)).T)
    # theta belongs to the pull map (reference -> session): the inverse rotation
    assert abs(np.rad2deg(rigid.theta) + 1.5) < 0.1, np.rad2deg(rigid.theta)
    assert np.median(err) < 0.3 and err.max() < 0.8, (np.median(err), err.max())


def test_large_rotation():
    """Day-to-day rotations of ~7 deg and 30+ px shifts occur in the OLL data."""
    ref = scene()
    mov = make_session(ref, theta_deg=-7.0, t=(-20.6, 33.9), amp=0.0)
    rigid, _, _ = register(ref, mov)
    u = interior_points()
    err = np.hypot(*(rigid.push(u) - h_true(u, theta_deg=-7.0, t=(-20.6, 33.9), amp=0.0)).T)
    assert abs(np.rad2deg(rigid.theta) - 7.0) < 0.1, np.rad2deg(rigid.theta)
    assert np.median(err) < 0.3, np.median(err)


def test_nonrigid_recovers_warp():
    ref = scene()
    mov = make_session(ref)
    rigid, nonrigid, info = register(ref, mov)
    assert info["nonrigid"] == "ok", info
    u = interior_points()
    err_rigid = np.hypot(*(rigid.push(u) - h_true(u)).T)
    err = np.hypot(*(nonrigid.push(u) - h_true(u)).T)
    assert np.median(err) < 0.5 and np.percentile(err, 95) < 1.0, (np.median(err), np.percentile(err, 95))
    assert np.median(err) < np.median(err_rigid)
    # pull and push are inverses
    assert np.abs(nonrigid.pull(nonrigid.push(u)) - u).max() < 0.05
    # warped session image matches the reference
    ref_e, mov_e = reg.enhance(ref, 12), reg.enhance(mov, 12)
    assert reg.image_corr(ref_e, nonrigid.warp_image(mov_e), BORDER) > 0.9


def test_centroid_qc_and_roundtrip():
    ref = scene()
    mov = make_session(ref)
    _, nonrigid, _ = register(ref, mov)
    u = interior_points()                                   # session centroids
    q = reg.centroid_qc(h_true(u), u, nonrigid, radius=10)
    assert q["frac_close"] > 0.95 and q["median_dist"] < 0.5 and q["match_excess"] > 0.8, q
    ident = reg.centroid_qc(h_true(u), u, reg.Transform((H, W)), radius=10)
    assert ident["match_excess"] < q["match_excess"] - 0.5, ident
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.npz")
        nonrigid.save(path, ref_pid="x")
        back = reg.Transform.load(path)
    assert np.allclose(back.push(u), nonrigid.push(u))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
