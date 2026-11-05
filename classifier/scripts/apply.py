#!/usr/bin/env python
"""Apply a deployment model and write results_quality_model.csv to plane folders.

The file has the same format as caiman-reader's results_quality.csv (no header,
roi_id,A|R, one row per estimates/idx_components entry in that order).
Probabilities go to <output_root>/predictions/<version>/<pid>.csv.

Safety:
- results_quality.csv is never written, renamed or deleted.
- An existing results_quality_model.csv is skipped. With --replace it is first
  renamed to results_quality_model_<version that wrote it>.csv (nothing deleted).
- Planes with a manual results_quality.csv are skipped unless --include-labeled.
- Every existing file in the plane folder is fingerprinted before and after; the
  run fails if anything other than the new file changed.

  python apply.py --version v20261105 --plane /path/26-03-02-OLL/RB01/plane0 --dry-run
  python apply.py --version v20261105 --plane-list lists/unlabeled.txt --task-id 0 --chunk 5
"""
import argparse
import hashlib
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd

from cmclf import io, jobs, model
from cmclf.features import extract_plane_features
from cmclf.movie_features import MOVIE_FEATURES

HASH_MAX_BYTES = 20 * 2 ** 20


def fingerprint(fold):
    fp = {}
    for name in os.listdir(fold):
        path = os.path.join(fold, name)
        if not os.path.isfile(path):
            continue
        st = os.stat(path)
        digest = None
        if st.st_size <= HASH_MAX_BYTES:
            with open(path, "rb") as f:
                digest = hashlib.sha1(f.read()).hexdigest()
        fp[name] = (st.st_size, st.st_mtime_ns, digest)
    return fp


def previous_version(cfg, pid):
    """Version whose predictions file for this plane is most recent."""
    root = os.path.join(cfg["output_root"], "predictions")
    hits = [(os.path.getmtime(os.path.join(root, v, f"{pid}.csv")), v)
            for v in (os.listdir(root) if os.path.isdir(root) else [])
            if os.path.exists(os.path.join(root, v, f"{pid}.csv"))]
    return max(hits)[1] if hits else "unknown"


def apply_plane(cfg, p, est, meta, a):
    fold = p["dir"]
    out = os.path.join(fold, io.MODEL_CSV)
    if os.path.exists(os.path.join(fold, io.MANUAL_CSV)) and not a.include_labeled:
        return "skipped: has manual labels (--include-labeled to apply anyway)"
    if os.path.exists(out) and not a.replace:
        return f"skipped: {io.MODEL_CSV} exists (--replace to rename it and write a new one)"

    plane = io.load_plane(fold)
    if len(plane.idx) == 0:
        return "skipped: no accepted components"
    df, _ = extract_plane_features(plane, cfg["features"])
    if any(f in MOVIE_FEATURES for f in meta["features"]):
        from cmclf.movie_features import extract_movie_features
        df = df.merge(extract_movie_features(plane, cfg["movie_features"], cfg["features"]), on="roi_id", how="left")
    df = df.sort_values("order_idx")
    assert list(df["roi_id"]) == list(plane.idx), "feature rows are not in idx_components order"
    p_acc, pred = model.predict(est, meta, df)
    labels = ["A" if v else "R" for v in pred]
    msg = f"{len(labels)} ROIs: {sum(pred)} A / {len(labels) - sum(pred)} R (threshold {meta['threshold']})"
    if a.dry_run:
        return "dry run, nothing written. " + msg

    before = fingerprint(fold)
    renamed = None
    if os.path.exists(out):
        old = previous_version(cfg, p["pid"])
        renamed = f"results_quality_model_{old}.csv"
        if os.path.exists(os.path.join(fold, renamed)):
            renamed = f"results_quality_model_{old}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        os.rename(out, os.path.join(fold, renamed))
    io.write_labels(out, zip(plane.idx.tolist(), labels))

    pred_dir = os.path.join(cfg["output_root"], "predictions", meta["version"])
    os.makedirs(pred_dir, exist_ok=True)
    pd.DataFrame({"roi_id": plane.idx, "p_accept": p_acc, "label": labels, "threshold": meta["threshold"],
                  "model_version": meta["version"]}).to_csv(os.path.join(pred_dir, f"{p['pid']}.csv"), index=False)

    # Verify: output format, and nothing else in the folder changed
    back = io.read_labels(out)
    assert [k for k, _ in back] == plane.idx.tolist() and {v for _, v in back} <= {"A", "R"}, "bad output file"
    after = fingerprint(fold)
    expected = dict(before)
    if renamed:
        expected[renamed] = expected.pop(io.MODEL_CSV)
    new = set(after) - set(expected)
    changed = [n for n in expected if n not in after or
               (after[n][0], after[n][2]) != (expected[n][0], expected[n][2]) or
               (n != renamed and after[n][1] != expected[n][1])]
    if new != {io.MODEL_CSV} or changed:
        raise RuntimeError(f"unexpected changes in {fold}: new={sorted(new)} changed={changed}")
    return f"wrote {io.MODEL_CSV}" + (f" (previous renamed to {renamed})" if renamed else "") + ". " + msg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, help="model version under <output_root>/models/")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--plane")
    src.add_argument("--plane-list")
    ap.add_argument("--task-id", type=int, help="array task (default $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--chunk", type=int, default=1, help="planes per array task")
    ap.add_argument("--include-labeled", action="store_true")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    jobs.guard_login_node()
    cfg = io.load_config(a.config)
    est, meta = model.load_model(os.path.join(cfg["output_root"], "models", a.version, "deploy"))
    planes = [io.plane_from_dir(a.plane)] if a.plane else io.task_slice(
        io.read_plane_list(a.plane_list), jobs.task_id(a.task_id), a.chunk)

    failed = []
    for p in planes:
        t0 = time.time()
        try:
            print(f"{p['pid']}: {apply_plane(cfg, p, est, meta, a)} [{time.time() - t0:.0f} s]")
        except Exception:
            traceback.print_exc()
            failed.append(p["pid"])
    if failed:
        sys.exit(f"Failed: {failed}")


if __name__ == "__main__":
    main()
