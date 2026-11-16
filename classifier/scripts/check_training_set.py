#!/usr/bin/env python
"""Check the training planes' label files before training. Read-only and light
(reads small CSVs, only checks that results.hdf5 exists), so it can run anywhere.

For each training plane: the label file train.py will use, A / R / blank / F
counts and, for model-seeded planes, how many labels differ from the model file.
Flags:
  NO_HDF5          no results.hdf5 in the plane folder
  NO_LABELS        the label file for its label source is missing
  ALSO_MANUAL      seeded plane that also has a manual results_quality.csv
  ALSO_EDITS       manual plane that also has a _useredits file
  EDIT_VERSIONS    several _useredits versions; the newest is used
  UNCHANGED        edits file identical to the model file (saved without changes)
  BLANKS           unreviewed (blank) labels; they are left out of training
  ROWS_DIFFER      label file has a different set of ROIs from the model file

  python check_training_set.py --config classifier/config_xgboost-02.yaml
  python check_training_set.py --config ... --out $OUT/reports/xgboost-02_training_check.csv
"""
import argparse
import collections
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd

from cmclf import io


def check(p):
    fold = p["dir"]
    source = p.get("label_source", "manual")
    row = dict(pid=p["pid"], label_source=source, label_file="", n_A=0, n_R=0, n_blank=0, n_F=0,
               n_changed_from_model="", flags=[])
    if not os.path.exists(os.path.join(fold, io.HDF5_NAME)):
        row["flags"].append("NO_HDF5")
    try:
        path = io.label_path(p)
    except FileNotFoundError:
        row["flags"].append("NO_LABELS")
        return row
    row["label_file"] = os.path.basename(path)
    labels = io.read_labels(path)
    counts = collections.Counter(v for _, v in labels)
    row.update(n_A=counts["A"], n_R=counts["R"], n_blank=counts[""], n_F=counts["F"])
    if counts[""]:
        row["flags"].append("BLANKS")

    edits = glob.glob(os.path.join(glob.escape(fold), io.MODEL_CSV_PREFIX + "*" + io.USER_EDITS_SUFFIX + "*.csv"))
    if source == "manual":
        if edits:
            row["flags"].append("ALSO_EDITS")
        return row
    if os.path.exists(os.path.join(fold, io.MANUAL_CSV)):
        row["flags"].append("ALSO_MANUAL")
    tag = source.split(":", 1)[1]
    model_csv = os.path.join(fold, io.model_csv_name(tag))
    if len(io.user_edits_versions(model_csv)) > 1:
        row["flags"].append("EDIT_VERSIONS")
    if os.path.exists(model_csv):
        model = dict(io.read_labels(model_csv))
        mine = dict(labels)
        if set(model) != set(mine):
            row["flags"].append("ROWS_DIFFER")
        row["n_changed_from_model"] = sum(model.get(k) != v for k, v in mine.items())
        if row["n_changed_from_model"] == 0:
            row["flags"].append("UNCHANGED")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--out", help="also write the table as CSV (refuses to overwrite)")
    a = ap.parse_args()
    cfg = io.load_config(a.config)
    planes = io.training_planes(cfg)
    test_ids = {p["pid"] for p in io.test_planes(cfg)}

    df = pd.DataFrame([check(p) for p in planes])
    df["test"] = df["pid"].isin(test_ids)
    df["flags"] = df["flags"].apply(" ".join)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    print(df.to_string(index=False))

    print(f"\n{len(df)} training planes ({len(test_ids)} locked test): "
          + ", ".join(f"{n} {s}" for s, n in df["label_source"].value_counts().items()))
    print(f"Labels: {df['n_A'].sum()} A, {df['n_R'].sum()} R, {df['n_blank'].sum()} blank, {df['n_F'].sum()} F")
    seeded = df[df["n_changed_from_model"] != ""]
    if len(seeded):
        ch = seeded["n_changed_from_model"].astype(int)
        print(f"Seeded planes: {ch.sum()} labels changed from the model "
              f"(median {ch.median():.0f} per plane, {int((ch == 0).sum())} planes unchanged)")
    flags = collections.Counter(f for fl in df["flags"] for f in fl.split())
    print("Flags: " + (", ".join(f"{f} {n}" for f, n in sorted(flags.items())) or "none"))
    blocking = df[df["flags"].str.contains("NO_HDF5|NO_LABELS")]
    if a.out:
        df.to_csv(a.out, index=False, mode="x")
        print(f"-> {a.out}")
    if len(blocking):
        sys.exit(f"{len(blocking)} planes can't be used for training: {list(blocking['pid'])}")


if __name__ == "__main__":
    main()
