#!/usr/bin/env python
"""Cross-session registration for one mouse x plane (xgboost-context, stage 1).
Images: each session's ops.npy meanImg, contrast enhanced (cmclf/registration.py).
Nothing downstream uses a transform until its QC has been looked at.

Two kinds of registration:
- Pairs (what the context features use): each session registered directly to
  the next `neighbor_sessions` sessions by date. Nearby days look alike, while
  the FOV drifts and rotates (7 deg or more) over weeks, so a single distant
  reference registers poorly.
- Reference (overview only): every session to one reference session, for the
  sharpness of the average image in stack.png.

QC per registration, for no registration / rigid / nonrigid: correlation of the
enhanced images (border excluded), and how many CaImAn-accepted centroids land
within 3 px of one in the other session *beyond chance* (match_excess: the same
count after a 40 px offset is subtracted; most accepted components aren't cells,
so the raw fraction is low even when aligned). Nonrigid is kept only if it makes
neither worse than rigid.

Output, <output_root>/registration/<run>/<mouse>_<plane>/:
  pairs/<pidA>__<pidB>.npz  transform, A-space -> B-space pull map (see Transform)
  pairs.csv                 one row per pair: rotation, shift, QC, mode, pass
  ref/<pid>.npz, ref.csv    the same for the reference registrations
  summary.json              reference, parameters, git commit, failing pairs
  pairs_overlays_<k>.png    consecutive sessions: earlier (magenta) vs later
                            (green) registered to it; full FOV + centre zoom
  pairs_qc.png              pair QC against date, consecutive and longer gaps
  stack.png                 reference vs the average of the passing sessions
An existing group folder is never overwritten: pick a new --run.

  python register_fov.py --list
  python register_fov.py --mouse RB06 --plane plane0 --run v2
  sbatch --array=0-11 slurm/register.sbatch --run v2      (one group per task)
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

MODES = ("raw", "rigid", "nonrigid", "final")


def groups(cfg):
    return sorted({(p["mouse"], p["plane"]) for p in io.discover_planes(cfg["data_root"])})


def load_session(p, cfg, rcfg):
    plane = io.load_plane(p["dir"])
    if plane.mean_img is None:
        return None
    geom = footprint_geometry(plane, cfg["features"]["mask_threshold"])
    cent = np.stack([geom["cy"], geom["cx"]], axis=1)[plane.idx]
    return dict(pid=p["pid"], session=p["session"], dims=plane.dims,
                enh=reg.enhance(plane.mean_img, rcfg["enhance_sigma_px"]), cent=cent)


def choose_reference(sess, rcfg):
    """Among the middle half of the sessions by date, the one whose image
    correlates best (median phase-correlation peak, rigid with rotation) with
    the sessions around it."""
    n = len(sess)
    lo, hi = n // 4, max(n - n // 4, n // 4 + 1)
    scores = {}
    for i in range(lo, hi):
        others = [j for j in range(max(0, i - 3), min(n, i + 4)) if j != i]
        s = [reg.rigid_register(sess[i]["enh"], sess[j]["enh"], rcfg["rigid_max_shift_px"],
                                rcfg["max_angle_deg"], 2 * rcfg["angle_step_deg"])[1] for j in others]
        scores[sess[i]["pid"]] = float(np.median(s)) if s else 0.0
    return max(scores, key=scores.get), scores


def evaluate(a, b, t, rcfg):
    """a is the space registered into; t pulls a-space points into b."""
    q = reg.centroid_qc(a["cent"], b["cent"], t, rcfg["match_radius_px"], rcfg["match_close_px"])
    return dict(corr=reg.image_corr(a["enh"], t.warp_image(b["enh"]), rcfg["qc_border_px"]), **q)


def register_pair(a, b, rcfg):
    """Register b onto a. Returns (Transform a -> b, QC row)."""
    ident = reg.Transform(a["dims"])
    rigid, strength = reg.rigid_register(a["enh"], b["enh"], rcfg["rigid_max_shift_px"],
                                         rcfg["max_angle_deg"], rcfg["angle_step_deg"])
    nonrigid, info = reg.nonrigid_register(a["enh"], rigid, b["enh"], rcfg["block_px"], rcfg["block_overlap"],
                                           rcfg["nonrigid_max_shift_px"], rcfg["min_block_strength"],
                                           rcfg["field_smooth_blocks"], n_iter=rcfg["nonrigid_iterations"])
    q = {"raw": evaluate(a, b, ident, rcfg), "rigid": evaluate(a, b, rigid, rcfg)}
    q["nonrigid"] = evaluate(a, b, nonrigid, rcfg) if nonrigid.is_nonrigid else dict(q["rigid"])
    better = (nonrigid.is_nonrigid
              and q["nonrigid"]["match_excess"] >= q["rigid"]["match_excess"] - 0.01
              and q["nonrigid"]["corr"] >= q["rigid"]["corr"] - 0.01)
    final = nonrigid if better else rigid
    q["final"] = q["nonrigid"] if better else q["rigid"]
    p = rcfg["pass"]
    row = dict(theta_deg=np.rad2deg(final.theta), shift_y=final.shift[0], shift_x=final.shift[1],
               max_field_px=final.max_field(), mode="nonrigid" if better else "rigid",
               nonrigid_status=info.get("nonrigid", ""), n_valid_blocks=info.get("n_valid_blocks"),
               rigid_peak_strength=strength, **{f"{k}_{m}": v for m in MODES for k, v in q[m].items()})
    row["pass"] = bool(row["corr_final"] >= p["min_img_corr"] and row["match_excess_final"] >= p["min_match_excess"])
    return final, row


def describe(row):
    return (f"{row['mode']:8s} rot {row['theta_deg']:+6.2f} deg  shift ({row['shift_y']:+6.1f}, {row['shift_x']:+6.1f})  "
            f"corr {row['corr_raw']:.2f}->{row['corr_final']:.2f}  cells within 3 px "
            f"{100 * row['frac_close_final']:.0f}% (chance {100 * row['null_close_final']:.0f}%)  "
            f"{'ok' if row['pass'] else 'FAIL'}")


# ── Figures ──────────────────────────────────────────────────────────────────

def to_unit(e):
    return np.clip((np.nan_to_num(e) + 1.5) / 4.5, 0, 1)


def plot_pair_overlays(sess, consecutive, out_dir, zoom):
    """consecutive: list of (i, j, transform, row) for j = i + 1."""
    per_page = 8
    H, W = sess[0]["dims"]
    zoom = min(zoom, H, W)
    y0, x0 = (H - zoom) // 2, (W - zoom) // 2
    for k in range(0, len(consecutive), per_page):
        chunk = consecutive[k:k + per_page]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(8, 4 * len(chunk)), squeeze=False)
        for row_i, (i, j, t, r) in enumerate(chunk):
            a = to_unit(sess[i]["enh"])
            rgb = np.stack([a, to_unit(t.warp_image(sess[j]["enh"])), a], axis=-1)
            axes[row_i, 0].imshow(rgb)
            axes[row_i, 0].add_patch(plt.Rectangle((x0, y0), zoom, zoom, fill=False, ec="yellow", lw=0.8))
            axes[row_i, 1].imshow(rgb[y0:y0 + zoom, x0:x0 + zoom])
            axes[row_i, 0].set_title(f"{sess[i]['session']} (magenta) vs {sess[j]['session']} (green)"
                                     f"{'' if r['pass'] else '  FAIL'}", fontsize=8,
                                     color="black" if r["pass"] else "red")
            axes[row_i, 1].set_title(f"{r['mode']}, rot {r['theta_deg']:+.1f} deg, corr {r['corr_final']:.2f}, "
                                     f"cells {100 * r['frac_close_final']:.0f}% vs {100 * r['null_close_final']:.0f}%",
                                     fontsize=8)
            for ax in axes[row_i]:
                ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"pairs_overlays_{k // per_page + 1:02d}.png"), dpi=90)
        plt.close(fig)


def plot_pairs_qc(pairs, sessions, rcfg, out_dir, title):
    idx = {s: i for i, s in enumerate(sessions)}
    pairs = pairs.assign(i=pairs["session_a"].map(idx), gap=pairs["session_b"].map(idx) - pairs["session_a"].map(idx))
    fig, axes = plt.subplots(3, 1, figsize=(max(8, 0.3 * len(sessions)), 9), sharex=True)
    for gap, g in pairs.groupby("gap"):
        axes[0].plot(g["i"], g["corr_final"], marker=".", label=f"+{gap} session{'s' if gap > 1 else ''}")
        axes[1].plot(g["i"], g["match_excess_final"], marker=".")
        axes[2].plot(g["i"], g["theta_deg"], marker=".")
    consecutive = pairs[pairs["gap"] == 1]
    axes[0].plot(consecutive["i"], consecutive["corr_raw"], ":", color="gray", label="+1, unregistered")
    axes[0].axhline(rcfg["pass"]["min_img_corr"], color="r", lw=0.8)
    axes[1].axhline(rcfg["pass"]["min_match_excess"], color="r", lw=0.8)
    axes[0].set_ylabel("image corr")
    axes[1].set_ylabel("cells within 3 px\nminus chance")
    axes[2].set_ylabel("rotation (deg)")
    axes[0].legend(fontsize=7)
    for i in sorted(set(pairs.loc[~pairs["pass"], "i"])):
        for ax in axes:
            ax.axvspan(i - 0.4, i + 0.4, color="r", alpha=0.08)
    axes[2].set_xticks(range(len(sessions)))
    axes[2].set_xticklabels(sessions, rotation=90, fontsize=7)
    fig.suptitle(title + "  (x: the earlier session of each pair)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pairs_qc.png"), dpi=90)
    plt.close(fig)


def plot_stack(ref, sess, transforms, refqc, out_dir, zoom):
    keep = [i for i in range(len(sess)) if refqc.iloc[i]["pass"]]
    H, W = ref["dims"]
    zoom = min(zoom, H, W)
    y0, x0 = (H - zoom) // 2, (W - zoom) // 2
    stack = (np.nanmean([transforms[i].warp_image(sess[i]["enh"]) for i in keep], axis=0) if keep
             else np.zeros((H, W)))
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for col, (img, name) in enumerate(((ref["enh"], f"reference {ref['session']}"),
                                       (stack, f"average of {len(keep)} sessions passing vs the reference"))):
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
    ap.add_argument("--run", default="v2", help="output subfolder under registration/")
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

    # Pairs: each session onto the next k
    k = rcfg["neighbor_sessions"]
    pair_rows, consecutive, pair_t = [], [], []
    for i in range(len(sess)):
        for j in range(i + 1, min(i + 1 + k, len(sess))):
            t, row = register_pair(sess[i], sess[j], rcfg)
            row = dict(pid_a=sess[i]["pid"], pid_b=sess[j]["pid"], session_a=sess[i]["session"],
                       session_b=sess[j]["session"], gap=j - i, n_cells_a=len(sess[i]["cent"]),
                       n_cells_b=len(sess[j]["cent"]), **row)
            pair_rows.append(row)
            pair_t.append(t)
            if j == i + 1:
                consecutive.append((i, j, t, row))
            print(f"  {sess[i]['session']} <- {sess[j]['session']}: {describe(row)}")
    pairs = pd.DataFrame(pair_rows)

    # Reference registration, for the overview
    ref_pid, ref_scores = choose_reference(sess, rcfg)
    ref = next(s for s in sess if s["pid"] == ref_pid)
    print(f"  reference {ref_pid}")
    ref_rows, ref_t = [], []
    for s in sess:
        if s["pid"] == ref_pid:
            t = reg.Transform(dims)
            q = evaluate(ref, s, t, rcfg)
            row = dict(theta_deg=0.0, shift_y=0.0, shift_x=0.0, max_field_px=0.0, mode="reference", pass_=True,
                       **{f"{k_}_{m}": v for m in MODES for k_, v in q.items()})
            row["pass"] = row.pop("pass_")
        else:
            t, row = register_pair(ref, s, rcfg)
        ref_rows.append(dict(pid=s["pid"], session=s["session"], **row))
        ref_t.append(t)
    refqc = pd.DataFrame(ref_rows)

    os.makedirs(os.path.join(out_dir, "pairs"))
    os.makedirs(os.path.join(out_dir, "ref"))
    for r, t in zip(pair_rows, pair_t):
        t.save(os.path.join(out_dir, "pairs", f"{r['pid_a']}__{r['pid_b']}.npz"), pid_a=r["pid_a"], pid_b=r["pid_b"],
               mode=r["mode"])
    for r, t in zip(ref_rows, ref_t):
        t.save(os.path.join(out_dir, "ref", f"{r['pid']}.npz"), ref_pid=ref_pid, pid=r["pid"], mode=r["mode"])
    pairs.to_csv(os.path.join(out_dir, "pairs.csv"), index=False)
    refqc.to_csv(os.path.join(out_dir, "ref.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(dict(mouse=mouse, plane=plane, run=a.run, n_sessions=len(sess), skipped=skipped,
                       n_pairs=len(pairs), n_pairs_pass=int(pairs["pass"].sum()),
                       consecutive_pass=f"{int(pairs.loc[pairs['gap'] == 1, 'pass'].sum())}/"
                                        f"{int((pairs['gap'] == 1).sum())}",
                       failed_pairs=pairs.loc[~pairs["pass"], ["session_a", "session_b"]].values.tolist(),
                       reference=ref_pid, reference_scores=ref_scores, n_ref_pass=int(refqc["pass"].sum()),
                       params=rcfg, git_commit=io.git_commit()), f, indent=2, default=str)
    zoom = rcfg["qc_zoom_px"]
    plot_pair_overlays(sess, consecutive, out_dir, zoom)
    plot_pairs_qc(pairs, [s["session"] for s in sess], rcfg, out_dir, f"{mouse} {plane}: pairwise registration")
    plot_stack(ref, sess, ref_t, refqc, out_dir, zoom)
    print(f"  pairs: {int(pairs['pass'].sum())}/{len(pairs)} pass "
          f"(consecutive {int(pairs.loc[pairs['gap'] == 1, 'pass'].sum())}/{int((pairs['gap'] == 1).sum())}); "
          f"vs reference {int(refqc['pass'].sum())}/{len(refqc)}; {time.time() - t0:.0f} s -> {out_dir}")


if __name__ == "__main__":
    main()
