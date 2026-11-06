#!/usr/bin/env python
"""Cross-validate, ablate and fit the classifier. Locked test planes are never
loaded except to refit the deployment model at the very end (stage final).

Stages (default: all):
  cv        nested leave-one-plane-out CV on the dev planes for every model:
            always_reject, snr_logistic, logistic_all, replica, full (+ full_movie)
  loao      nested leave-one-animal-out CV for full and replica
  ablation  full model minus each feature group (nested leave-one-plane-out)
  final     tune + fit on the dev planes -> models/<version>/dev (scored once by
            evaluate.py --locked-test); refit with the same settings on all
            training planes -> models/<version>/deploy (used by apply.py)

Outputs: <output_root>/reports/<version>/ and <output_root>/models/<version>/
"""
import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd

from cmclf import io, jobs, model, report
from cmclf.features import FEATURE_GROUPS
from cmclf.movie_features import MOVIE_FEATURES


def md_table(df, floatfmt="{:.3f}"):
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for i, row in df.iterrows():
        vals = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join([str(i)] + vals) + " |")
    return "\n".join(lines)


SUMMARY_COLS = ["balanced_accuracy", "roc_auc", "pr_auc", "pr_auc_chance", "kappa", "mcc", "accuracy",
                "majority_baseline", "accept_precision", "accept_recall", "accept_f1", "brier",
                "plane_balanced_accuracy_mean", "plane_balanced_accuracy_sd", "plane_kappa_mean",
                "plane_abs_count_error_mean", "plane_abs_count_error_pct_mean"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=f"v{datetime.date.today():%Y%m%d}")
    ap.add_argument("--stages", default="cv,loao,ablation,final")
    ap.add_argument("--with-movie", action="store_true", help="also evaluate full_movie (needs G9 features)")
    ap.add_argument("--final-set", choices=["full", "full_movie"], default="full")
    ap.add_argument("--n-jobs", type=int, help="parallel outer folds (default $SLURM_CPUS_PER_TASK)")
    ap.add_argument("--quick", action="store_true", help="one hyperparameter combo, 3 inner folds (debugging)")
    ap.add_argument("--config")
    a = ap.parse_args()
    jobs.guard_login_node()
    cfg = io.load_config(a.config)
    mcfg = dict(cfg["model"])
    if a.quick:
        mcfg["grid"] = {k: v[:1] for k, v in mcfg["grid"].items()}
        mcfg["inner_folds"] = 3
    if a.final_set == "full_movie":
        a.with_movie = True
    seed, n_jobs = cfg["random_seed"], jobs.n_cpus(a.n_jobs)
    stages = a.stages.split(",")

    rep = os.path.join(cfg["output_root"], "reports", a.version)
    mdir = os.path.join(cfg["output_root"], "models", a.version)
    os.makedirs(rep, exist_ok=True)
    test_ids = {p["pid"] for p in io.test_planes(cfg)}
    dev = io.dev_planes(cfg)
    data, label_summary = model.load_table(cfg, dev, with_movie=a.with_movie)
    assert not set(data["pid"]) & test_ids, "locked test planes leaked into the dev data"
    report.write_table(label_summary.set_index("pid"), os.path.join(rep, "label_summary_dev.csv"))
    print(f"Dev: {data['pid'].nunique()} planes, {len(data)} cells, {data['y'].sum()} accepted "
          f"({100 * data['y'].mean():.1f}%); excluded: {label_summary[['n_blank', 'n_F', 'n_missing']].sum().to_dict()}")
    print(f"Locked test planes (not loaded): {sorted(test_ids)}")
    with open(os.path.join(rep, "run_args.json"), "w") as f:
        json.dump(dict(vars(a), model_cfg=mcfg, git_commit=io.git_commit()), f, indent=2)

    cv_dir = os.path.join(rep, "cv")
    os.makedirs(cv_dir, exist_ok=True)

    def run_cv(name, kind, features, outer="pid"):
        t0 = time.time()
        oof = model.nested_cv(data, features, kind, outer, mcfg, seed, n_jobs)
        oof.to_parquet(os.path.join(cv_dir, f"{name}_{outer}_oof.parquet"), index=False)
        print(f"  {name} ({outer}-out): {time.time() - t0:.0f} s")
        return oof

    if "cv" in stages:
        print("Stage cv: nested leave-one-plane-out")
        rows, preds = [], {}
        for name, (kind, fs) in model.MODEL_SPECS.items():
            if fs == "full_movie" and not a.with_movie:
                continue
            preds[name] = run_cv(name, kind, model.FEATURE_SETS[fs])
            rows.append(report.summarize(preds[name], name))
            report.write_table(report.per_group(preds[name], "pid"), os.path.join(cv_dir, f"per_plane_{name}.csv"))
        summary = pd.DataFrame(rows).set_index("model")
        report.write_table(summary, os.path.join(rep, "cv_summary.csv"))
        report.plot_curves(preds, os.path.join(rep, "cv_roc_pr.png"))
        best = a.final_set if a.final_set in preds else "full"
        report.plot_diagnostics(preds[best], os.path.join(rep, f"cv_diagnostics_{best}.png"),
                                f"{best}: nested leave-one-plane-out CV")
        report.write_table(report.per_group(preds[best], "mouse"), os.path.join(rep, f"cv_per_animal_{best}.csv"))
        report.write_table(report.disagreements(preds[best]).set_index("pid"),
                           os.path.join(rep, f"cv_disagreements_{best}.csv"))
        print(summary[SUMMARY_COLS[:6]].round(3).to_string())

    if "loao" in stages:
        print("Stage loao: nested leave-one-animal-out")
        rows = []
        for name in ("replica", a.final_set):
            kind, fs = model.MODEL_SPECS[name]
            oof = run_cv(name, kind, model.FEATURE_SETS[fs], outer="mouse")
            rows.append(report.summarize(oof, name))
            report.write_table(report.per_group(oof, "mouse"), os.path.join(rep, f"loao_per_animal_{name}.csv"))
        report.write_table(pd.DataFrame(rows).set_index("model"), os.path.join(rep, "loao_summary.csv"))

    if "ablation" in stages:
        print("Stage ablation: drop one feature group at a time")
        base_path = os.path.join(cv_dir, f"{a.final_set}_pid_oof.parquet")
        if os.path.exists(base_path):
            base = report.summarize(pd.read_parquet(base_path), a.final_set)
        else:
            base = report.summarize(run_cv(a.final_set, "xgb_tuned", model.FEATURE_SETS[a.final_set]), a.final_set)
        rows = [base]
        for name, feats in model.ablation_sets(with_movie=a.final_set == "full_movie").items():
            rows.append(report.summarize(run_cv(name, "xgb_tuned", feats), name))
        abl = pd.DataFrame(rows).set_index("model")
        for c in ("balanced_accuracy", "roc_auc", "pr_auc", "kappa"):
            abl[f"delta_{c}"] = abl[c] - abl.loc[a.final_set, c]
        report.write_table(abl, os.path.join(rep, "ablation_summary.csv"))
        print(abl[[c for c in abl if c.startswith("delta_")]].round(4).to_string())

    if "final" in stages:
        print(f"Stage final: {a.final_set}")
        feats = model.FEATURE_SETS[a.final_set]
        X, y, g = model.matrix(data, feats), data["y"].to_numpy(), data["pid"].to_numpy()
        _, info, est = model.fit_model("xgb_tuned", X, y, g, mcfg, seed)
        cv_row = None
        cv_path = os.path.join(rep, "cv_summary.csv")
        if os.path.exists(cv_path):
            cv = pd.read_csv(cv_path, index_col=0)
            cv_row = cv.loc[a.final_set].to_dict() if a.final_set in cv.index else None
        meta = dict(version=a.version, tag=io.model_tag(cfg), feature_set=a.final_set, features=feats,
                    threshold=info["threshold"], params=info["params"], n_estimators=info["n_estimators"],
                    inner_logloss=info["inner_logloss"], test_planes=sorted(test_ids),
                    n_cells=len(data), n_accept=int(y.sum()), cv_summary=cv_row, model_cfg=mcfg)
        model.save_model(os.path.join(mdir, "dev"), est, dict(meta, stage="dev", train_planes=sorted(set(g))))
        imp = model.shap_importance(est, X, feats)
        report.write_table(imp.rename("mean_abs_shap").to_frame(), os.path.join(rep, "feature_importance_dev.csv"))
        groups = dict(FEATURE_GROUPS, **({"G9_movie": MOVIE_FEATURES} if a.final_set == "full_movie" else {}))
        report.plot_importance(imp, groups, os.path.join(rep, "feature_importance_dev.png"))
        print(f"  dev model: {info}")

        # Deployment model: same settings, all training planes (incl. locked test)
        alldata, all_summary = model.load_table(cfg, io.training_planes(cfg), with_movie=a.with_movie)
        report.write_table(all_summary.set_index("pid"), os.path.join(rep, "label_summary_all.csv"))
        Xa, ya = model.matrix(alldata, feats), alldata["y"].to_numpy()
        dep = model.make_xgb(mcfg, info["params"], ya, n_estimators=info["n_estimators"], seed=seed)
        dep.fit(Xa, ya)
        model.save_model(os.path.join(mdir, "deploy"), dep,
                         dict(meta, stage="deploy", n_cells=len(alldata), n_accept=int(ya.sum()),
                              train_planes=sorted(alldata["pid"].unique())))
        print(f"  models -> {mdir}/{{dev,deploy}}")

    write_markdown(rep, a.version)


def write_markdown(rep, version):
    parts = [f"# Auto-QC classifier report {version}\n"]
    for title, fname, cols in (("Nested leave-one-plane-out CV (dev planes)", "cv_summary.csv", SUMMARY_COLS),
                               ("Leave-one-animal-out CV", "loao_summary.csv", SUMMARY_COLS),
                               ("Ablation (full model minus one group)", "ablation_summary.csv", None)):
        path = os.path.join(rep, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            if cols is None:
                cols = ["balanced_accuracy", "roc_auc", "pr_auc", "kappa"] + [c for c in df if c.startswith("delta_")]
            parts.append(f"## {title}\n\n" + md_table(df[[c for c in cols if c in df]]) + "\n")
    imp = os.path.join(rep, "feature_importance_dev.csv")
    if os.path.exists(imp):
        parts.append("## Top 20 features (mean |SHAP|, dev model)\n\n" + md_table(pd.read_csv(imp, index_col=0).head(20)) + "\n")
    with open(os.path.join(rep, "report.md"), "w") as f:
        f.write("\n".join(parts))
    print(f"Report: {os.path.join(rep, 'report.md')}")


if __name__ == "__main__":
    main()
