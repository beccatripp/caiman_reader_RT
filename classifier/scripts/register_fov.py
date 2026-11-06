#!/usr/bin/env python
"""Register every session of one mouse x plane to a reference session
(xgboost-context, stage 1). Images: each session's ops.npy meanImg, contrast
enhanced (cmclf/registration.py). Nothing downstream uses a transform until its
QC has been looked at.

Groups are the (mouse, plane) pairs under data_root; --list prints them with
their array index. Output, <output_root>/registration/<run>/<mouse>_<plane>/:
  <pid>.npz         transform (reference -> session pull map) + mode, ref_pid
  qc.csv            per session: rotation, shift, largest nonrigid displacement;
                    image correlation and accepted-cell centroid matching for no
                    registration / rigid / nonrigid; the mode kept; pass flags
  summary.json      reference choice, parameters, git commit
  overlays_<k>.png  reference (magenta) vs registered session (green): full FOV
                    and a centre zoom; aligned structure looks white
  qc_metrics.png    QC against session date
  stack.png         reference vs the average of all passing registered sessions
An existing group folder is never overwritten: pick a new --run.

  python register_fov.py --list
  python register_fov.py --mouse RB06 --plane plane0
  sbatch --array=0-11 slurm/register.sbatch        (one group per array task)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cmclf import io, jobs
from cmclf import registration as reg
from cmclf.features import footprint_geometry


def groups(cfg):
    return sorted({(p["mouse"], p["plane"]) for p in io.discover_planes(cfg["data_root"])})


def load_session(p, cfg, rcfg):
    plane = io.load_plane(p["dir"])
    if plane.mean_img is None:
        return None
    geom = footprint_geometry(plane, cfg["features"]["mask_threshold"])
    cent = np.stack([geom["cy"], geom["cx"]], axis=1)[plane.idx]
    return dict(pid=p["pid"], session=p["session"], dims=plane.dims, img=plane.mean_img,
                enh=reg.enhance(plane.mean_img, rcfg["enhance_sigma_px"]), cent=cent)


def choose_reference(sess, rcfg):
    """Among the middle half of the sessions by date, the one whose image
    correlates best (median phase-correlation peak) with all the others."""
    n = len(sess)
    lo, hi = n // 4, max(n - n // 4, n // 4 + 1)
    win = reg.tukey2d(sess[0]["enh"].shape)
    scores = {}
    for i in range(lo, hi):
        b = sess[i]["enh"] * win
        s = [reg.phase_corr(sess[j]["enh"] * win, b, rcfg["rigid_max_shift_px"])[1] for j in range(n) if j != i]
        scores[sess[i]["pid"]] = float(np.median(s)) if s else 0.0
    return max(scores, key=scores.get), scores


def evaluate(ref, s, t, rcfg):
    q = reg.centroid_qc(ref["cent"], s["cent"], t, rcfg["match_radius_px"])
    return dict(corr=reg.image_corr(ref["enh"], t.warp_image(s["enh"]), rcfg["qc_border_px"]), **q)


def register_session(ref, s, rcfg):
    ident = reg.Transform(ref["dims"])
    if s["pid"] == ref["pid"]:
        q = evaluate(ref, s, ident, rcfg)
        return ident, dict(mode="reference", nonrigid_status="", n_valid_blocks=np.nan,
                           **{f"{k}_{m}": v for m in ("raw", "rigid", "nonrigid", "final") for k, v in q.items()})
    rigid, strength = reg.rigid_register(ref["enh"], s["enh"], rcfg["rigid_max_shift_px"],
                                         rcfg["max_angle_deg"], rcfg["angle_step_deg"])
    nonrigid, info = reg.nonrigid_register(ref["enh"], rigid, s["enh"], rcfg["block_px"], rcfg["block_overlap"],
                                           rcfg["nonrigid_max_shift_px"], rcfg["min_block_strength"],
                                           rcfg["field_smooth_blocks"], n_iter=rcfg["nonrigid_iterations"])
    q = {"raw": evaluate(ref, s, ident, rcfg), "rigid": evaluate(ref, s, rigid, rcfg)}
    q["nonrigid"] = evaluate(ref, s, nonrigid, rcfg) if nonrigid.is_nonrigid else dict(q["rigid"])
    # Keep the nonrigid field only if it doesn't make the cell matching or the
    # image correlation worse than rigid alone
    better = (nonrigid.is_nonrigid
              and not (q["nonrigid"]["median_dist"] > q["rigid"]["median_dist"] + 0.25)
              and q["nonrigid"]["corr"] >= q["rigid"]["corr"] - 0.01)
    final, mode = (nonrigid, "nonrigid") if better else (rigid, "rigid")
    q["final"] = q["nonrigid"] if better else q["rigid"]
    row = dict(mode=mode, nonrigid_status=info.get("nonrigid", ""), n_valid_blocks=info.get("n_valid_blocks"),
               rigid_peak_strength=strength,
               **{f"{k}_{m}": v for m, d in q.items() for k, v in d.items()})
    return final, row


# ── Figures ──────────────────────────────────────────────────────────────────

def to_unit(e):
    return np.clip((np.nan_to_num(e) + 1.5) / 4.5, 0, 1)


def overlay(ref_e, warped_e):
    r, g = to_unit(ref_e), to_unit(warped_e)
    return np.stack([r, g, r], axis=-1)


def plot_overlays(ref, sess, transforms, qc, out_dir, zoom):
    per_page = 8
    H, W = ref["dims"]
    zoom = min(zoom, H, W)
    y0, x0 = (H - zoom) // 2, (W - zoom) // 2
    for k in range(0, len(sess), per_page):
        chunk = list(range(k, min(k + per_page, len(sess))))
        fig, axes = plt.subplots(len(chunk), 2, figsize=(8, 4 * len(chunk)), squeeze=False)
        for row, i in enumerate(chunk):
            s, r = sess[i], qc.iloc[i]
            rgb = overlay(ref["enh"], transforms[i].warp_image(s["enh"]))
            axes[row, 0].imshow(rgb)
            axes[row, 0].add_patch(plt.Rectangle((x0, y0), zoom, zoom, fill=False, ec="yellow", lw=0.8))
            axes[row, 1].imshow(rgb[y0:y0 + zoom, x0:x0 + zoom])
            flag = "" if r["pass"] else "  FAIL"
            axes[row, 0].set_title(f"{s['session']}  {r['mode']}{flag}", fontsize=9,
                                   color="red" if not r["pass"] else "black")
            axes[row, 1].set_title(f"corr {r['corr_final']:.2f}  cells {r['median_dist_final']:.1f} px  "
                                   f"{100 * r['frac_matched_final']:.0f}% matched", fontsize=8)
            for ax in axes[row]:
                ax.set_axis_off()
        fig.suptitle(f"reference {ref['session']} (magenta) vs session (green)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"overlays_{k // per_page + 1:02d}.png"), dpi=90)
        plt.close(fig)


def plot_qc(qc, rcfg, out_dir, title):
    x = np.arange(len(qc))
    fig, axes = plt.subplots(3, 1, figsize=(max(8, 0.3 * len(qc)), 9), sharex=True)
    for m, st in (("raw", ":"), ("rigid", "--"), ("final", "-")):
        axes[0].plot(x, qc[f"corr_{m}"], st, marker=".", label=m)
        axes[1].plot(x, qc[f"median_dist_{m}"], st, marker=".", label=m)
        axes[2].plot(x, qc[f"frac_matched_{m}"], st, marker=".", label=m)
    p = rcfg["pass"]
    axes[0].axhline(p["min_img_corr"], color="r", lw=0.8)
    axes[1].axhline(p["max_median_dist_px"], color="r", lw=0.8)
    axes[2].axhline(p["min_frac_matched"], color="r", lw=0.8)
    axes[0].set_ylabel("image corr")
    axes[1].set_ylabel("median cell distance (px)")
    axes[2].set_ylabel("fraction of cells matched")
    for ax in axes:
        for i in np.flatnonzero(~qc["pass"].to_numpy()):
            ax.axvspan(i - 0.4, i + 0.4, color="r", alpha=0.12)
        ref_i = np.flatnonzero(qc["mode"].to_numpy() == "reference")
        for i in ref_i:
            ax.axvline(i, color="k", lw=0.6)
    axes[0].legend(fontsize=8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(qc["session"], rotation=90, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "qc_metrics.png"), dpi=90)
    plt.close(fig)


def plot_stack(ref, sess, transforms, qc, out_dir, zoom):
    keep = [i for i in range(len(sess)) if qc.iloc[i]["pass"]]
    stack = np.nanmean([transforms[i].warp_image(sess[i]["enh"]) for i in keep], axis=0)
    H, W = ref["dims"]
    zoom = min(zoom, H, W)
    y0, x0 = (H - zoom) // 2, (W - zoom) // 2
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for col, (img, name) in enumerate(((ref["enh"], f"reference {ref['session']}"),
                                       (stack, f"average of {len(keep)} passing sessions"))):
        axes[0, col].imshow(to_unit(img), cmap="gray")
        axes[0, col].set_title(name, fontsize=9)
        axes[1, col].imshow(to_unit(img)[y0:y0 + zoom, x0:x0 + zoom], cmap="gray")
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stack.png"), dpi=90)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the groups and their array index")
    ap.add_argument("--mouse")
    ap.add_argument("--plane")
    ap.add_argument("--task-id", type=int, help="group index (default $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--run", default="v1", help="output subfolder under registration/")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = io.load_config(a.config)
    rcfg = cfg["registration"]
    allg = groups(cfg)
    if a.list:
        for i, (m, p) in enumerate(allg):
            print(f"{i:3d}  {m} {p}")
        return
    jobs.guard_login_node()
    if a.mouse and a.plane:
        mouse, plane = a.mouse, a.plane
    else:
        t = jobs.task_id(a.task_id)
        if t is None:
            sys.exit("give --mouse and --plane, or --task-id / an array task")
        mouse, plane = allg[t]
    out_dir = os.path.join(cfg["output_root"], "registration", a.run, f"{mouse}_{plane}")
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; use a new --run to register again")

    t0 = time.time()
    planes = sorted((p for p in io.discover_planes(cfg["data_root"]) if p["mouse"] == mouse and p["plane"] == plane),
                    key=lambda p: p["session"])
    print(f"{mouse} {plane}: {len(planes)} sessions")
    sess, skipped = [], []
    for p in planes:
        try:
            s = load_session(p, cfg, rcfg)
        except Exception as e:  # a broken session must not stop the group
            s = None
            print(f"  {p['pid']}: could not load ({e})")
        if s is None:
            skipped.append(p["pid"])
            continue
        sess.append(s)
    dims = pd.Series([s["dims"] for s in sess]).mode()[0]
    for s in [s for s in sess if s["dims"] != dims]:
        print(f"  {s['pid']}: dims {s['dims']} != {dims}, skipped")
        skipped.append(s["pid"])
    sess = [s for s in sess if s["dims"] == dims]
    if len(sess) < 2:
        sys.exit("fewer than 2 usable sessions")
    print(f"  loaded {len(sess)} sessions in {time.time() - t0:.0f} s; skipped {skipped}")

    ref_pid, ref_scores = choose_reference(sess, rcfg)
    ref = next(s for s in sess if s["pid"] == ref_pid)
    print(f"  reference {ref_pid}")

    rows, transforms = [], []
    p = rcfg["pass"]
    for s in sess:
        t, row = register_session(ref, s, rcfg)
        row = dict(pid=s["pid"], session=s["session"], n_cells=len(s["cent"]), theta_deg=np.rad2deg(t.theta),
                   shift_y=t.shift[0], shift_x=t.shift[1], max_field_px=t.max_field(), **row)
        row["pass"] = bool(row["corr_final"] >= p["min_img_corr"]
                           and row["median_dist_final"] <= p["max_median_dist_px"]
                           and row["frac_matched_final"] >= p["min_frac_matched"])
        rows.append(row)
        transforms.append(t)
        print(f"  {s['pid']}: {row['mode']:9s} rot {row['theta_deg']:+.2f} deg  shift ({row['shift_y']:+.1f}, "
              f"{row['shift_x']:+.1f})  corr {row['corr_raw']:.2f}->{row['corr_final']:.2f}  cells "
              f"{row['median_dist_raw']:.1f}->{row['median_dist_final']:.1f} px, "
              f"{100 * row['frac_matched_final']:.0f}% matched  {'ok' if row['pass'] else 'FAIL'}")
    qc = pd.DataFrame(rows)

    os.makedirs(out_dir)
    for s, t, r in zip(sess, transforms, rows):
        t.save(os.path.join(out_dir, f"{s['pid']}.npz"), ref_pid=ref_pid, mode=r["mode"], pid=s["pid"])
    qc.to_csv(os.path.join(out_dir, "qc.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(dict(mouse=mouse, plane=plane, run=a.run, reference=ref_pid, reference_scores=ref_scores,
                       n_sessions=len(sess), n_pass=int(qc["pass"].sum()), skipped=skipped,
                       failed=qc.loc[~qc["pass"], "pid"].tolist(), params=rcfg, git_commit=io.git_commit()),
                  f, indent=2)
    zoom = rcfg["qc_zoom_px"]
    plot_overlays(ref, sess, transforms, qc, out_dir, zoom)
    plot_qc(qc, rcfg, out_dir, f"{mouse} {plane}: registration to {ref['session']}")
    plot_stack(ref, sess, transforms, qc, out_dir, zoom)
    print(f"  {int(qc['pass'].sum())}/{len(qc)} sessions pass; {time.time() - t0:.0f} s -> {out_dir}")


if __name__ == "__main__":
    main()
