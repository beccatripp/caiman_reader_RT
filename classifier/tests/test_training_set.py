"""Training set from a CSV, label sources, and the --untrained plane list, on a
temporary data folder.

  python -m pytest classifier/tests        (or: python classifier/tests/test_training_set.py)"""
import os
import subprocess
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER = os.path.join(HERE, "..")
sys.path.insert(0, CLASSIFIER)
from cmclf import io

CSV_HEADER = "Mouse,Plane,Date,Status,Seg Type,Reviewed On\n"


def make_root(d):
    """Training: manual; seeded with two edits versions; seeded with only the model
    file; seeded with only a manual file; seeded with no label file.
    Untrained: partial manual labels; only a model file."""
    data = os.path.join(d, "data")

    def plane(session, mouse, pl, files):
        fold = os.path.join(data, f"{session}-OLL", mouse, pl)
        os.makedirs(fold)
        for name, text in dict(files, **{"results.hdf5": ""}).items():
            with open(os.path.join(fold, name), "w") as f:
                f.write(text)
        return fold

    plane("26-03-01", "RB01", "plane0", {"results_quality.csv": "1,A\n2,R\n"})
    plane("26-03-02", "RB01", "plane0", {"results_quality_model_xgboost.csv": "1,A\n2,R\n",
                                         "results_quality_model_xgboost_useredits.csv": "1,A\n2,A\n",
                                         "results_quality_model_xgboost_useredits-v02.csv": "1,R\n2,A\n",
                                         "results_quality_model_xgboost_useredits__20261106.csv": "1,A\n2,A\n"})
    plane("26-03-03", "RB01", "plane1", {"results_quality_model_xgboost.csv": "1,A\n2,R\n"})
    plane("26-03-04", "RB01", "plane0", {"results_quality.csv": "1,R\n2,R\n"})
    plane("26-03-05", "RB01", "plane0", {})
    plane("26-03-24", "RB07", "plane0", {"results_quality.csv": "1,A\n2,\n"})
    plane("26-03-25", "RB07", "plane0", {"results_quality_model_xgboost.csv": "1,A\n2,R\n"})

    with open(os.path.join(d, "set.csv"), "w") as f:
        f.write(CSV_HEADER + "RB01,0,26-03-01,\"Processed, Training Set\",Relaxed Params,26-10-26\n"
                "RB01,0,26-03-02,Processed,xgboost,26-11-13\nRB01,1,26-03-03,Processed,xgboost,26-11-13\n"
                "RB01,0,26-03-04,Processed,xgboost,26-11-13\nRB01,0,26-03-05,Processed,xgboost,26-11-13\n")
    cfg = yaml.safe_load(open(os.path.join(CLASSIFIER, "config_xgboost-02.yaml")))
    cfg.update(data_root=data, output_root=os.path.join(d, "out"), training_set="set.csv",
               test_planes={"RB01": {"plane0": ["26-03-01"]}})
    path = os.path.join(d, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_real_training_set():
    cfg = io.load_config(os.path.join(CLASSIFIER, "config_xgboost-02.yaml"))
    train, test, dev = io.training_planes(cfg), io.test_planes(cfg), io.dev_planes(cfg)
    assert (len(train), len(test), len(dev)) == (125, 6, 119)
    sources = [p["label_source"] for p in train]
    assert sources.count("manual") == 46 and sources.count("useredits:xgboost") == 79
    assert all(p["label_source"] == "manual" for p in test)
    # The old config is unchanged: 45 manual planes
    old = io.load_config()
    assert len(io.training_planes(old)) == 45 and old["model_tag"] == "xgboost"


def test_label_path():
    with tempfile.TemporaryDirectory() as d:
        cfg = io.load_config(make_root(d))
        planes = {p["pid"]: p for p in io.training_planes(cfg)}
        assert os.path.basename(io.label_path(planes["26-03-01-OLL_RB01_plane0"])) == "results_quality.csv"
        resolved = {}
        for pid in ["26-03-02-OLL_RB01_plane0", "26-03-03-OLL_RB01_plane1", "26-03-04-OLL_RB01_plane0"]:
            path, source = io.resolve_labels(planes[pid])
            resolved[pid] = (os.path.basename(path), source)
        assert resolved == {
            "26-03-02-OLL_RB01_plane0": ("results_quality_model_xgboost_useredits-v02.csv", "useredits:xgboost"),
            "26-03-03-OLL_RB01_plane1": ("results_quality_model_xgboost.csv", "model:xgboost"),
            "26-03-04-OLL_RB01_plane0": ("results_quality.csv", "manual")}, resolved
        try:
            io.label_path(planes["26-03-05-OLL_RB01_plane0"])
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass
        assert [p["pid"] for p in io.test_planes(cfg)] == ["26-03-01-OLL_RB01_plane0"]
        assert len(io.dev_planes(cfg)) == 4


def test_untrained_list_and_check():
    with tempfile.TemporaryDirectory() as d:
        cfg_path = make_root(d)
        out = os.path.join(d, "untrained.txt")
        r = subprocess.run([sys.executable, os.path.join(CLASSIFIER, "scripts", "make_plane_list.py"),
                            "--config", cfg_path, "--untrained", "--out", out], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        listed = sorted(io.plane_from_dir(line.strip())["pid"] for line in open(out))
        assert listed == ["26-03-24-OLL_RB07_plane0", "26-03-25-OLL_RB07_plane0"], listed
        assert "26-03-24-OLL_RB07_plane0: results_quality.csv" in r.stdout, r.stdout

        r = subprocess.run([sys.executable, os.path.join(CLASSIFIER, "scripts", "check_training_set.py"),
                            "--config", cfg_path], capture_output=True, text=True)
        assert r.returncode != 0 and "26-03-05-OLL_RB01_plane0" in r.stderr, r.stderr   # no label file
        assert "26-03-03-OLL_RB01_plane1" not in r.stderr, r.stderr                    # model file used
        for flag in ["EDIT_VERSIONS", "NO_LABELS", "NO_EDITS 2", "model:xgboost"]:
            assert flag in r.stdout, (flag, r.stdout)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
