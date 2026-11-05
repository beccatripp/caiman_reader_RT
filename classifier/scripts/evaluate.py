#!/usr/bin/env python
"""Held-out evaluation.

  --locked-test   score models/<version>/dev on the locked test planes. Runs once
                  per version (refuses if a result exists, unless --force).
  --compare       compare results_quality_model.csv with a manual
                  results_quality.csv on the given planes (the prospective blind
                  check), against the pass bar in config.yaml.

  python evaluate.py --version v20261105 --locked-test
  python evaluate.py --version v20261105 --compare --plane-list lists/blind_check.txt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd

from cmclf import io, jobs, model, report


def locked_test(cfg, version, force):
    out = os.path.join(cfg["output_root"], "reports", version, "locked_test")
    if os.path.exists(os.path.join(out, "metrics.json")) and not force:
        sys.exit(f"Locked test already scored for {version} ({out}). It is meant to be run once; "
                 f"use --force only if you know why.")
    est, meta = model.load_model(os.path.join(cfg["output_root"], "models", version, "dev"))
    test = io.test_planes(cfg)
    assert not set(meta["train_planes"]) & {p["pid"] for p in test}, "test planes were in the dev training set"
    data, summary = model.load_table(cfg, test, with_movie=meta["feature_set"] == "full_movie")
    p, pred = model.predict(est, meta, data)
    preds = data[["pid", "mouse", "plane", "roi_id", "y"]].assign(p=p, pred=pred, threshold=meta["threshold"])
    os.makedirs(out, exist_ok=True)
    pooled = report.metrics(preds["y"], preds["pred"], preds["p"])
    row = report.summarize(preds, version)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(dict(pooled=pooled, summary=row, threshold=meta["threshold"],
                       planes=[p["pid"] for p in test]), f, indent=2, default=float)
    preds.to_csv(os.path.join(out, "predictions.csv"), index=False)
    report.write_table(report.per_group(preds, "pid"), os.path.join(out, "per_plane.csv"))
    report.write_table(summary.set_index("pid"), os.path.join(out, "label_summary.csv"))
    report.write_table(report.disagreements(preds).set_index("pid"), os.path.join(out, "disagreements.csv"))
    report.plot_diagnostics(preds, os.path.join(out, "diagnostics.png"), f"{version}: locked test set")
    print(pd.Series(pooled).to_string())
    print(f"-> {out}")


def compare(cfg, version, planes, name):
    rows = []
    for pl in planes:
        mpath = os.path.join(pl["dir"], io.MODEL_CSV)
        hpath = os.path.join(pl["dir"], io.MANUAL_CSV)
        if not (os.path.exists(mpath) and os.path.exists(hpath)):
            print(f"{pl['pid']}: needs both {io.MODEL_CSV} and {io.MANUAL_CSV}, skipping")
            continue
        m = pd.DataFrame(io.read_labels(mpath), columns=["roi_id", "model"])
        h = pd.DataFrame(io.read_labels(hpath), columns=["roi_id", "manual"])
        d = m.merge(h, on="roi_id", how="inner")
        n_unreviewed = int((~d["manual"].isin(["A", "R"])).sum())
        d = d[d["manual"].isin(["A", "R"])]
        ppath = os.path.join(cfg["output_root"], "predictions", version, f"{pl['pid']}.csv")
        if os.path.exists(ppath):
            d = d.merge(pd.read_csv(ppath)[["roi_id", "p_accept"]], on="roi_id", how="left")
        else:
            d["p_accept"] = np.nan
        d = d.assign(pid=pl["pid"], y=(d["manual"] == "A").astype(int), pred=(d["model"] == "A").astype(int))
        print(f"{pl['pid']}: {len(d)} cells compared, {n_unreviewed} unreviewed in the manual file")
        rows.append(d)
    if not rows:
        sys.exit("Nothing to compare")
    d = pd.concat(rows, ignore_index=True)
    has_p = d["p_accept"].notna().all()
    d["p"] = d["p_accept"]
    pooled = report.metrics(d["y"], d["pred"], d["p"] if has_p else None)
    bar = cfg["prospective_pass"]
    passed = pooled["balanced_accuracy"] >= bar["balanced_accuracy"] and pooled["kappa"] >= bar["kappa"]
    out = os.path.join(cfg["output_root"], "reports", version, f"compare_{name}")
    os.makedirs(out, exist_ok=True)
    per_plane = pd.DataFrame({pid: report.metrics(g["y"], g["pred"]) for pid, g in d.groupby("pid")}).T
    report.write_table(per_plane, os.path.join(out, "per_plane.csv"))
    d.to_csv(os.path.join(out, "cells.csv"), index=False)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(dict(pooled=pooled, pass_bar=bar, passed=bool(passed)), f, indent=2, default=float)
    print(per_plane[["n", "accuracy", "balanced_accuracy", "kappa", "n_accept_manual", "n_accept_model"]].round(3).to_string())
    print(pd.Series(pooled).to_string())
    print(f"Pass bar: balanced accuracy >= {bar['balanced_accuracy']}, kappa >= {bar['kappa']} -> "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--locked-test", action="store_true")
    mode.add_argument("--compare", action="store_true")
    ap.add_argument("--plane-list")
    ap.add_argument("--plane", nargs="+")
    ap.add_argument("--name", default="prospective", help="output folder suffix for --compare")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    jobs.guard_login_node()
    cfg = io.load_config(a.config)
    if a.locked_test:
        locked_test(cfg, a.version, a.force)
    else:
        planes = io.read_plane_list(a.plane_list) if a.plane_list else [io.plane_from_dir(d) for d in a.plane or []]
        compare(cfg, a.version, planes, a.name)


if __name__ == "__main__":
    main()
