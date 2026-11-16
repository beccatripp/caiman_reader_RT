#!/usr/bin/env python
"""Write a plane list (one plane folder per line) for the array jobs.

  --training / --dev / --test    the planes in config.yaml
  --untrained                    every plane with results.hdf5 that is NOT a training
                                 plane of the config, whatever label files it has (those
                                 with labels are listed; apply them with --include-labeled)
  (default)                      every plane with results.hdf5 and NO manual
                                 results_quality.csv (use --include-labeled to keep those)

  python make_plane_list.py --training --out lists/training.txt
  python make_plane_list.py --out lists/unlabeled.txt
  python make_plane_list.py --random 3 --seed 1 --out lists/blind_check.txt
  python make_plane_list.py --config classifier/config_xgboost-02.yaml --untrained --out lists/xgboost-02_untrained.txt
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from cmclf import io


def labelled_files(fold):
    """A manual results_quality.csv and any caiman-reader _useredits files."""
    found = glob.glob(os.path.join(glob.escape(fold), io.MODEL_CSV_PREFIX + "*" + io.USER_EDITS_SUFFIX + "*.csv"))
    manual = os.path.join(fold, io.MANUAL_CSV)
    return found + ([manual] if os.path.exists(manual) else [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--training", action="store_true")
    src.add_argument("--dev", action="store_true")
    src.add_argument("--test", action="store_true")
    src.add_argument("--untrained", action="store_true", help="all planes not in the config's training planes")
    ap.add_argument("--mouse", nargs="+")
    ap.add_argument("--session", nargs="+", help="e.g. 26-03-12-OLL")
    ap.add_argument("--plane", nargs="+", help="e.g. plane0")
    ap.add_argument("--include-labeled", action="store_true", help="keep planes with a manual results_quality.csv")
    ap.add_argument("--random", type=int, help="pick N planes at random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = io.load_config(a.config)

    if a.training:
        planes = io.training_planes(cfg)
    elif a.dev:
        planes = io.dev_planes(cfg)
    elif a.test:
        planes = io.test_planes(cfg)
    elif a.untrained:
        trained = {p["pid"] for p in io.training_planes(cfg)}
        planes = [p for p in io.discover_planes(cfg["data_root"]) if p["pid"] not in trained]
        print(f"{len(trained)} training planes left out")
    else:
        planes = io.discover_planes(cfg["data_root"])
        if not a.include_labeled:
            planes = [p for p in planes if not os.path.exists(os.path.join(p["dir"], io.MANUAL_CSV))]
    if a.mouse:
        planes = [p for p in planes if p["mouse"] in a.mouse]
    if a.session:
        planes = [p for p in planes if p["session"] in a.session]
    if a.plane:
        planes = [p for p in planes if p["plane"] in a.plane]
    missing = [p["dir"] for p in planes if not os.path.exists(os.path.join(p["dir"], io.HDF5_NAME))]
    if missing:
        sys.exit("No results.hdf5 in:\n  " + "\n  ".join(missing))
    if a.random:
        rng = np.random.default_rng(a.seed)
        planes = [planes[i] for i in sorted(rng.choice(len(planes), size=min(a.random, len(planes)), replace=False))]

    if a.untrained:
        labelled = [(p["pid"], sorted(os.path.basename(f) for f in labelled_files(p["dir"]))) for p in planes]
        labelled = [(pid, files) for pid, files in labelled if files]
        if labelled:
            print(f"{len(labelled)} listed planes already have labels (not training planes; "
                  f"apply.py needs --include-labeled for them, and never changes these files):")
            for pid, files in labelled:
                print(f"  {pid}: {', '.join(files)}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "x") as f:
        for p in planes:
            f.write(p["dir"] + "\n")
    print(f"{len(planes)} planes -> {a.out}")
    print(f"Array range for sbatch: --array=0-{len(planes) - 1} (with CHUNK=1)")


if __name__ == "__main__":
    main()
