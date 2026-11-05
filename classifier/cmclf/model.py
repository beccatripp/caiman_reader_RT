"""Feature tables, nested cross-validation, threshold choice, final fits.

Every choice a model makes (hyperparameters, number of trees, decision
threshold) is taken from inner folds grouped by plane, so the outer held-out
plane (or animal) never influences it."""
import datetime
import itertools
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import CORE_FEATURES, FEATURE_GROUPS, REPLICA_FEATURES
from .io import MANUAL_CSV, git_commit, read_labels
from .movie_features import MOVIE_FEATURES

FEATURE_SETS = {
    "snr_only": ["snr"],
    "replica": REPLICA_FEATURES,
    "full": CORE_FEATURES,
    "full_movie": CORE_FEATURES + MOVIE_FEATURES,
}

# (name, kind, feature set)
MODEL_SPECS = {
    "always_reject": ("constant", "snr_only"),
    "snr_logistic": ("logistic", "snr_only"),
    "logistic_all": ("logistic", "full"),
    "replica": ("xgb_fixed", "replica"),
    "full": ("xgb_tuned", "full"),
    "full_movie": ("xgb_tuned", "full_movie"),
}

# The previous student's deployed settings (maze-runners S03), threshold 0.5
REPLICA_PARAMS = dict(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
                      colsample_bytree=0.8, random_state=42)

THRESHOLDS = np.round(np.arange(0.02, 0.981, 0.01), 2)


# ── Feature tables ───────────────────────────────────────────────────────────

def feature_path(cfg, pid, movie=False):
    return os.path.join(cfg["output_root"], "features_movie" if movie else "features", f"{pid}.parquet")


def load_table(cfg, planes, with_movie=False, labels=True):
    """Concatenated features for the planes; with labels, only A/R rows are kept
    (blank and F are excluded) and a per-plane label summary is returned."""
    frames, summary = [], []
    for p in planes:
        path = feature_path(cfg, p["pid"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run scripts/extract_features.py first")
        df = pd.read_parquet(path)
        if with_movie:
            mpath = feature_path(cfg, p["pid"], movie=True)
            if not os.path.exists(mpath):
                raise FileNotFoundError(f"{mpath} missing: run scripts/extract_features.py --movie first")
            df = df.merge(pd.read_parquet(mpath), on="roi_id", how="left")
        if labels:
            lab = dict(read_labels(os.path.join(p["dir"], MANUAL_CSV)))
            df["label"] = df["roi_id"].map(lab).fillna("missing").replace("", "blank")
            vc = df["label"].value_counts()
            summary.append(dict(pid=p["pid"], n_rows=len(df), n_A=int(vc.get("A", 0)), n_R=int(vc.get("R", 0)),
                                n_blank=int(vc.get("blank", 0)), n_F=int(vc.get("F", 0)),
                                n_missing=int(vc.get("missing", 0)),
                                n_csv_not_in_idx=len(set(lab) - set(df["roi_id"]))))
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    if not labels:
        return data
    data = data[data["label"].isin(["A", "R"])].copy()
    data["y"] = (data["label"] == "A").astype(int)
    return data.reset_index(drop=True), pd.DataFrame(summary)


def matrix(df, features):
    return df[features].astype(float).to_numpy()


# ── Estimators ───────────────────────────────────────────────────────────────

THRESHOLD_OBJECTIVES = {"kappa": cohen_kappa_score, "balanced_accuracy": balanced_accuracy_score}


def best_threshold(y, p, objective="kappa"):
    """Threshold maximising the objective (ties: closest to 0.5). Cohen's kappa
    keeps the accept rate near the manual one; balanced accuracy over-accepts
    when only ~25% of cells are accepted."""
    score = THRESHOLD_OBJECTIVES[objective]
    scores = np.array([score(y, p >= t) for t in THRESHOLDS])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    return float(THRESHOLDS[best[np.argmin(np.abs(THRESHOLDS[best] - 0.5))]])


def make_xgb(mcfg, params, y, n_estimators=None, early_stop=False, seed=0):
    spw = params["scale_pos_weight"]
    if spw == "balanced":
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
    return xgb.XGBClassifier(
        n_estimators=n_estimators or mcfg["max_estimators"], max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"], scale_pos_weight=float(spw),
        learning_rate=mcfg["learning_rate"], subsample=mcfg["subsample"],
        colsample_bytree=mcfg["colsample_bytree"], tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=50 if early_stop else None, n_jobs=1, random_state=seed, verbosity=0)


def make_logistic():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=5000))


def inner_splits(groups, n_folds):
    n = min(n_folds, len(np.unique(groups)))
    return list(GroupKFold(n_splits=n).split(np.zeros(len(groups)), groups=groups))


def tune_xgb(X, y, groups, mcfg, seed=0):
    """Grid over mcfg['grid'] with inner GroupKFold by plane. Trees: argmin of
    the mean inner validation log-loss curve. Threshold: optimum of
    mcfg['threshold_objective'] on the inner out-of-fold predictions at that
    many trees."""
    splits = inner_splits(groups, mcfg["inner_folds"])
    grid = mcfg["grid"]
    keys = list(grid)
    best = None
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        curves, models = [], []
        for tr, va in splits:
            m = make_xgb(mcfg, params, y[tr], early_stop=True, seed=seed)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            curves.append(np.asarray(m.evals_result()["validation_0"]["logloss"]))
            models.append(m)
        L = max(len(c) for c in curves)
        mean_curve = np.mean([np.pad(c, (0, L - len(c)), mode="edge") for c in curves], axis=0)
        n_trees = int(np.argmin(mean_curve)) + 1
        score = float(mean_curve[n_trees - 1])
        if best is None or score < best["inner_logloss"]:
            best = dict(params=params, n_estimators=n_trees, inner_logloss=score, models=models)
    oof = np.zeros(len(y))
    for (tr, va), m in zip(splits, best.pop("models")):
        n = min(best["n_estimators"], m.get_booster().num_boosted_rounds())
        oof[va] = m.predict_proba(X[va], iteration_range=(0, n))[:, 1]
    best["threshold"] = best_threshold(y, oof, mcfg.get("threshold_objective", "kappa"))
    return best


def fit_model(kind, X, y, groups, mcfg, seed=0):
    """Fit one model and choose its threshold from inner folds only.
    Returns (predict_proba function, info dict, fitted estimator or None)."""
    if kind == "constant":
        return (lambda Xn: np.zeros(len(Xn))), dict(threshold=0.5), None
    if kind == "xgb_fixed":
        m = xgb.XGBClassifier(**REPLICA_PARAMS, eval_metric="logloss", n_jobs=1, verbosity=0)
        m.fit(X, y)
        return (lambda Xn: m.predict_proba(Xn)[:, 1]), dict(threshold=0.5), m
    if kind == "logistic":
        oof = np.zeros(len(y))
        for tr, va in inner_splits(groups, mcfg["inner_folds"]):
            oof[va] = make_logistic().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        m = make_logistic().fit(X, y)
        threshold = best_threshold(y, oof, mcfg.get("threshold_objective", "kappa"))
        return (lambda Xn: m.predict_proba(Xn)[:, 1]), dict(threshold=threshold), m
    if kind == "xgb_tuned":
        info = tune_xgb(X, y, groups, mcfg, seed)
        m = make_xgb(mcfg, info["params"], y, n_estimators=info["n_estimators"], seed=seed)
        m.fit(X, y)
        return (lambda Xn: m.predict_proba(Xn)[:, 1]), info, m
    raise ValueError(kind)


# ── Cross-validation ─────────────────────────────────────────────────────────

def _outer_fold(data, features, kind, outer_col, held_out, mcfg, seed):
    te = (data[outer_col] == held_out).to_numpy()
    tr = ~te
    X = matrix(data, features)
    y = data["y"].to_numpy()
    predict, info, _ = fit_model(kind, X[tr], y[tr], data["pid"].to_numpy()[tr], mcfg, seed)
    p = predict(X[te])
    out = data.loc[te, ["pid", "mouse", "plane", "roi_id", "y"]].copy()
    out["p"] = p
    out["threshold"] = info["threshold"]
    out["pred"] = (p >= info["threshold"]).astype(int)
    out["fold"] = held_out
    out["params"] = json.dumps({k: v for k, v in info.items() if k in ("params", "n_estimators")})
    return out


def nested_cv(data, features, kind, outer_col, mcfg, seed=0, n_jobs=1):
    """Outer leave-one-<outer_col>-out; out-of-fold predictions for every row."""
    folds = sorted(data[outer_col].unique())
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_outer_fold)(data, features, kind, outer_col, g, mcfg, seed) for g in folds)
    return pd.concat(parts, ignore_index=True)


def ablation_sets(with_movie=False):
    base = FEATURE_SETS["full_movie" if with_movie else "full"]
    groups = dict(FEATURE_GROUPS)
    if with_movie:
        groups["G9_movie"] = MOVIE_FEATURES
    return {f"minus_{g}": [c for c in base if c not in cols] for g, cols in groups.items()}


# ── Saving / loading ─────────────────────────────────────────────────────────

def save_model(model_dir, estimator, meta):
    os.makedirs(model_dir, exist_ok=True)
    estimator.save_model(os.path.join(model_dir, "model.json"))
    meta = dict(meta, created=datetime.datetime.now().isoformat(timespec="seconds"),
                git_commit=git_commit(), xgboost_version=xgb.__version__)
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_model(model_dir):
    with open(os.path.join(model_dir, "metadata.json")) as f:
        meta = json.load(f)
    m = xgb.XGBClassifier()
    m.load_model(os.path.join(model_dir, "model.json"))
    return m, meta


def predict(estimator, meta, df):
    p = estimator.predict_proba(matrix(df, meta["features"]))[:, 1]
    return p, (p >= meta["threshold"]).astype(int)


def shap_importance(estimator, X, features):
    """Mean |SHAP| per feature (XGBoost's built-in TreeSHAP)."""
    contrib = estimator.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=features).sort_values(ascending=False)
