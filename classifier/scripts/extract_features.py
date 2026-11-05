#!/usr/bin/env python
"""Extract per-ROI features for planes and cache them in
<output_root>/features/<pid>.parquet (with --movie: features_movie/).

Also writes <pid>.checks.json: Dice of our footprint masks with roi_masks.npy
and correlation of our dF/F with traces_dff.npy (expect ~1 for both).

  python extract_features.py --training                       # all 46 labelled planes
  python extract_features.py --plane-list L.txt --task-id 3   # one array task
  python extract_features.py --plane /path/to/session/RB01/plane0
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cmclf import io, jobs
from cmclf.features import extract_plane_features, plane_checks
from cmclf.model import feature_path


def write_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--training", action="store_true", help="the training planes in config.yaml")
    src.add_argument("--plane-list")
    src.add_argument("--plane")
    ap.add_argument("--task-id", type=int, help="array task (default $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--chunk", type=int, default=1, help="planes per array task")
    ap.add_argument("--movie", action="store_true", help="G9 movie features instead of the core features")
    ap.add_argument("--overwrite", action="store_true", help="recompute cached features")
    ap.add_argument("--config")
    a = ap.parse_args()
    jobs.guard_login_node()
    cfg = io.load_config(a.config)

    if a.training:
        planes = io.training_planes(cfg)
    elif a.plane_list:
        planes = io.read_plane_list(a.plane_list)
    else:
        planes = [io.plane_from_dir(a.plane)]
    planes = io.task_slice(planes, jobs.task_id(a.task_id) if not a.plane else None, a.chunk)

    failed = []
    for p in planes:
        out = feature_path(cfg, p["pid"], movie=a.movie)
        if os.path.exists(out) and not a.overwrite:
            print(f"{p['pid']}: cached, skipping ({out})")
            continue
        t0 = time.time()
        try:
            plane = io.load_plane(p["dir"])
            if a.movie:
                from cmclf.movie_features import extract_movie_features
                df = extract_movie_features(plane, cfg["movie_features"], cfg["features"])
                write_parquet(df, out)
            else:
                df, extra = extract_plane_features(plane, cfg["features"])
                checks = plane_checks(plane, extra["geom"], extra["dff"])
                write_parquet(df, out)
                with open(out.replace(".parquet", ".checks.json"), "w") as f:
                    json.dump(checks, f, indent=2)
                print(f"  checks: {checks}")
            print(f"{p['pid']}: {len(df)} ROIs, {time.time() - t0:.0f} s -> {out}")
        except Exception:
            traceback.print_exc()
            failed.append(p["pid"])
    if failed:
        sys.exit(f"Failed: {failed}")


if __name__ == "__main__":
    main()
