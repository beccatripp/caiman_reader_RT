#!/usr/bin/env python
"""Which label file to use for every segmented plane, for analysis pipelines.
Read-only and light (lists files, reads nothing large).

One row per plane with results.hdf5 under data_root. `classifier` is:
  manual             training plane listed as Relaxed Params: results_quality.csv
  xgboost-useredits  training plane listed as xgboost, corrected in caiman-reader:
                     newest results_quality_model_xgboost_useredits[-vNN].csv
  xgboost            training plane listed as xgboost, cleared without edits:
                     results_quality_model_xgboost.csv
  <tag>-useredits    not a training plane, corrected in caiman-reader:
                     newest results_quality_model_<tag>_useredits[-vNN].csv
  <tag>              not a training plane: results_quality_model_<tag>.csv
  none               no label file yet (apply.py has not run on it)
with <tag> the config's model_tag (xgboost-02). label_file is the file name in
plane_dir (relative to data_root). Training planes follow io.resolve_labels, the
same rule train.py used.

  python classifier/scripts/label_sources.py --config classifier/config_xgboost-02.yaml \\
         --out classifier/label_sources/xgboost-02.csv
"""
import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cmclf import io

SOURCE_NAMES = {"manual": "manual", "useredits:xgboost": "xgboost-useredits", "model:xgboost": "xgboost"}


def choose(plane, trained, tag):
    """(classifier, label file path or '')."""
    if plane["pid"] in trained:
        path, source = io.resolve_labels(trained[plane["pid"]])
        return SOURCE_NAMES.get(source, source), path
    model_csv = os.path.join(plane["dir"], io.model_csv_name(tag))
    edits = io.user_edits_versions(model_csv)
    if edits:
        return f"{tag}-useredits", edits[-1]
    if os.path.exists(model_csv):
        return tag, model_csv
    return "none", ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--out", required=True, help="CSV to write (refuses to overwrite unless --force)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = io.load_config(a.config)
    tag = cfg["model_tag"]
    trained = {p["pid"]: p for p in io.training_planes(cfg)}
    test_ids = {p["pid"] for p in io.test_planes(cfg)}

    rows = []
    for p in io.discover_planes(cfg["data_root"]):
        classifier, path = choose(p, trained, tag)
        rows.append(dict(pid=p["pid"], date=p["session"].replace("-OLL", ""), mouse=p["mouse"], plane=p["plane"],
                         classifier=classifier, label_file=os.path.basename(path),
                         plane_dir=os.path.relpath(p["dir"], cfg["data_root"]).replace(os.sep, "/"),
                         training_plane=p["pid"] in trained, locked_test=p["pid"] in test_ids))
    missing = sorted(set(trained) - {r["pid"] for r in rows})
    if missing:
        sys.exit(f"training planes without results.hdf5: {missing}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w" if a.force else "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    counts = collections.Counter(r["classifier"] for r in rows)
    print(f"{len(rows)} planes: " + ", ".join(f"{n} {c}" for c, n in counts.most_common()))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
