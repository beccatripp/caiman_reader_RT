"""Metrics, summary tables and plots for cross-validation and test sets."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def metrics(y, pred, p=None):
    y, pred = np.asarray(y), np.asarray(pred)
    both = len(np.unique(y)) > 1
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(n=len(y), n_accept_manual=int(y.sum()), n_accept_model=int(pred.sum()),
               accuracy=accuracy_score(y, pred), majority_baseline=max(y.mean(), 1 - y.mean()),
               balanced_accuracy=balanced_accuracy_score(y, pred) if both else np.nan,
               accept_precision=precision_score(y, pred, zero_division=0),
               accept_recall=recall_score(y, pred, zero_division=0),
               accept_f1=f1_score(y, pred, zero_division=0),
               kappa=cohen_kappa_score(y, pred) if both else np.nan,
               mcc=matthews_corrcoef(y, pred) if both else np.nan,
               tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
    if p is not None:
        p = np.asarray(p)
        out.update(roc_auc=roc_auc_score(y, p) if both else np.nan,
                   pr_auc=average_precision_score(y, p) if both else np.nan,
                   pr_auc_chance=y.mean(), brier=brier_score_loss(y, p))
    return out


def per_group(pred_df, col):
    rows = []
    for g, d in pred_df.groupby(col):
        m = metrics(d["y"], d["pred"], d["p"] if "p" in d else None)
        m[col] = g
        m["count_error"] = m["n_accept_model"] - m["n_accept_manual"]
        m["count_error_pct"] = 100 * m["count_error"] / max(m["n_accept_manual"], 1)
        rows.append(m)
    return pd.DataFrame(rows).set_index(col)


def summarize(pred_df, name):
    """Pooled metrics plus per-plane mean and SD."""
    pooled = metrics(pred_df["y"], pred_df["pred"], pred_df.get("p"))
    pp = per_group(pred_df, "pid")
    row = {"model": name, **pooled}
    for c in ("balanced_accuracy", "roc_auc", "pr_auc", "kappa", "accept_f1"):
        if c in pp:
            row[f"plane_{c}_mean"] = pp[c].mean()
            row[f"plane_{c}_sd"] = pp[c].std()
    row["plane_abs_count_error_mean"] = pp["count_error"].abs().mean()
    row["plane_abs_count_error_pct_mean"] = pp["count_error_pct"].abs().mean()
    return row


def disagreements(pred_df, data_root=None, n=100):
    """Most confident model-vs-manual disagreements, for checking in the reader."""
    d = pred_df[pred_df["pred"] != pred_df["y"]].copy()
    d["confidence"] = np.abs(d["p"] - d["threshold"])
    d["manual"] = np.where(d["y"] == 1, "A", "R")
    d["model"] = np.where(d["pred"] == 1, "A", "R")
    cols = ["pid", "roi_id", "manual", "model", "p", "threshold", "confidence"]
    return d.sort_values("confidence", ascending=False)[cols].head(n)


def plot_curves(preds, out_png):
    """ROC and precision-recall curves for several models' predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, d in preds.items():
        if d["p"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(d["y"], d["p"])
        prec, rec, _ = precision_recall_curve(d["y"], d["p"])
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(d['y'], d['p']):.3f})")
        axes[1].plot(rec, prec, label=f"{name} ({average_precision_score(d['y'], d['p']):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    any_d = next(iter(preds.values()))
    axes[1].axhline(any_d["y"].mean(), color="k", ls="--", lw=0.8, label="chance")
    axes[0].set(xlabel="False accept rate", ylabel="True accept rate", title="ROC (AUC)")
    axes[1].set(xlabel="Accept recall", ylabel="Accept precision", title="Precision-recall (AP)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_diagnostics(pred_df, out_png, title):
    """Calibration, P(accept) by true label, per-plane accept counts."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    frac, mean_p = calibration_curve(pred_df["y"], pred_df["p"], n_bins=10, strategy="quantile")
    axes[0].plot(mean_p, frac, "o-")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="Predicted P(accept)", ylabel="Fraction accepted", title="Calibration")
    for lbl, color, name in ((0, "#c0392b", "manual R"), (1, "#27ae60", "manual A")):
        axes[1].hist(pred_df.loc[pred_df["y"] == lbl, "p"], bins=40, alpha=0.6, color=color,
                     density=True, label=name)
    axes[1].axvline(pred_df["threshold"].median(), color="k", ls="--", lw=1, label="threshold (median)")
    axes[1].set(xlabel="P(accept)", title="Predicted probability")
    axes[1].legend(fontsize=8)
    pp = per_group(pred_df, "pid")
    axes[2].scatter(pp["n_accept_manual"], pp["n_accept_model"])
    lim = [0, max(pp["n_accept_manual"].max(), pp["n_accept_model"].max()) * 1.05]
    axes[2].plot(lim, lim, "k--", lw=0.8)
    axes[2].set(xlabel="Manual accepts per plane", ylabel="Model accepts per plane", title="Accepts per plane")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_importance(imp, groups, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(imp) * 0.18)))
    imp.sort_values().plot.barh(ax=axes[0], color="#2c7fb8")
    axes[0].set(xlabel="mean |SHAP|", title="Feature importance")
    axes[0].tick_params(axis="y", labelsize=7)
    g = pd.Series({k: imp.reindex(v).sum() for k, v in groups.items()}).sort_values()
    g.plot.barh(ax=axes[1], color="#41b6c4")
    axes[1].set(xlabel="sum of mean |SHAP|", title="By feature group")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def write_table(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, float_format="%.4f")
