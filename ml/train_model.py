#!/usr/bin/env python3
"""
train_model.py   (Phase B - Steps B5, B6)

GOAL
  Teach a model to predict a link's utilization (%) HORIZON steps into the future
  (1 step = 5 minutes) from the last 30 minutes of that link's history.

WHAT THE SCRIPT DOES, IN ORDER
  1. loads link_timeseries.csv and builds one training row per (link, moment)
     using features.py (the same code the live controller will use)
  2. splits the data BY TIME: early part = training, late part = test.
     (Random shuffling would let the model "see the future" and fake good results.)
  3. computes two simple baselines that any model must beat:
        persistence   : "the future value equals the value now"
        moving_avg_3  : "the future value equals the average of the last 3 values"
  4. trains and tunes four models with time-ordered cross-validation:
        ridge (linear), decision_tree, random_forest, hist_gb (gradient boosting)
  5. picks the best model using ONLY the training data (cross-validation),
     then reports every model on the untouched test period
  6. saves tables, charts and the model file

USAGE   (from ~/sdn_project/ml)
  python3 train_model.py --horizon 1 --quick        # fast first run (about 1 minute)
  python3 train_model.py --horizon 1                # full tuning
  python3 train_model.py --horizon 3 --make-default # 15 minutes ahead, also saves model.pkl
  python3 train_model.py --horizon 3 --target level --tag level   # experiment, saved separately

OUTPUTS (in ml/outputs/h<H>[_tag]/ , H = horizon in steps)
  metrics.csv, per_link_metrics.csv, predictions_test.csv, feature_importance.csv,
  metrics.json, fig_*.png      and the model file ml/model_h<H>.pkl
"""
import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as F  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REQUIRED = ["timestamp", "link_id", "bandwidth_mbps", "propagation_delay_ms", "utilization_pct"]


# ----------------------------------------------------------------------------
# 1. Data and features
# ----------------------------------------------------------------------------
def load_data(csv_path):
    if not Path(csv_path).exists():
        sys.exit("ERROR: %s not found. Run Step 13 of Part 2 first." % csv_path)
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit("ERROR: CSV is missing columns: %s" % missing)
    df["dt"] = pd.to_datetime(df["timestamp"], format="%Y%m%d-%H%M")
    return df.sort_values(["link_id", "dt"]).reset_index(drop=True)


def build_dataset(df, horizon, use_time):
    """One row per (link, origin time). Target = utilization `horizon` steps later."""
    names = F.feature_names(use_time)
    rows = []
    span_needed = (F.WINDOW - 1 + horizon) * F.STEP_MINUTES
    for link_id, g in df.groupby("link_id", sort=False):
        util = g["utilization_pct"].to_numpy(dtype=float)
        dts = g["dt"].to_numpy(dtype="datetime64[ns]")
        hours = g["dt"].dt.hour.to_numpy()
        minutes = g["dt"].dt.minute.to_numpy()
        bw = float(g["bandwidth_mbps"].iloc[0])
        delay = float(g["propagation_delay_ms"].iloc[0])
        for i in range(F.WINDOW - 1, len(g) - horizon):
            # skip if the timeline has a gap inside the window or before the target
            if (dts[i + horizon] - dts[i - F.WINDOW + 1]) != np.timedelta64(span_needed, "m"):
                continue
            feat = F.row_features(util[i - F.WINDOW + 1:i + 1], bw, delay,
                                  minute_of_day=int(hours[i]) * 60 + int(minutes[i]),
                                  use_time=use_time)
            row = F.to_vector(feat, names)
            rows.append(row + [link_id, dts[i], dts[i + horizon], util[i + horizon]])
    cols = names + ["link_id", "origin_dt", "target_dt", "target"]
    data = pd.DataFrame(rows, columns=cols)
    return data.sort_values(["origin_dt", "link_id"]).reset_index(drop=True), names


def chronological_split(data, all_times, train_frac):
    """Train: rows whose TARGET lies before the cutoff. Test: rows whose ORIGIN is at/after it.
    This leaves no overlap: nothing the model trains on comes from the test period."""
    cutoff = int(len(all_times) * train_frac)
    origin_rank = np.searchsorted(all_times, data["origin_dt"].to_numpy(dtype="datetime64[ns]"))
    target_rank = np.searchsorted(all_times, data["target_dt"].to_numpy(dtype="datetime64[ns]"))
    train_mask = target_rank < cutoff
    test_mask = origin_rank >= cutoff
    return train_mask, test_mask, cutoff


# ----------------------------------------------------------------------------
# 2. Metrics
# ----------------------------------------------------------------------------
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    denom = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - p) ** 2) / denom) if denom > 0 else float("nan")


def prf(y_true_bool, y_pred_bool):
    tp = int(np.sum(y_true_bool & y_pred_bool))
    fp = int(np.sum(~y_true_bool & y_pred_bool))
    fn = int(np.sum(y_true_bool & ~y_pred_bool))
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and (prec + rec) > 0) else float("nan")
    return prec, rec, f1


def full_metrics(y, p, cong_thr, high_thr):
    y, p = np.asarray(y), np.asarray(p)
    hi = y > high_thr
    prec, rec, f1 = prf(y > cong_thr, p > cong_thr)
    return {
        "rmse": rmse(y, p), "mae": mae(y, p), "r2": r2(y, p),
        "rmse_high_load": rmse(y[hi], p[hi]) if hi.sum() > 0 else float("nan"),
        "n_high_load": int(hi.sum()),
        "cong_precision": prec, "cong_recall": rec, "cong_f1": f1,
    }


# ----------------------------------------------------------------------------
# 3. Models and tuning grids
# ----------------------------------------------------------------------------
def get_models(quick, seed):
    if quick:
        grids = {
            "ridge": {"ridge__alpha": [1.0, 10.0]},
            "decision_tree": {"max_depth": [4, 8, None], "min_samples_leaf": [5, 20]},
            "random_forest": {"n_estimators": [100], "max_depth": [8, None], "min_samples_leaf": [5]},
            "hist_gb": {"max_depth": [4], "learning_rate": [0.1]},
        }
    else:
        grids = {
            "ridge": {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]},
            "decision_tree": {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 5, 10, 20]},
            "random_forest": {"n_estimators": [150], "max_depth": [10, None], "min_samples_leaf": [3, 10]},
            "hist_gb": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1], "max_iter": [150]},
        }
    estimators = {
        "ridge": Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]),
        "decision_tree": DecisionTreeRegressor(random_state=seed),
        "random_forest": RandomForestRegressor(random_state=seed, n_jobs=-1),
        "hist_gb": HistGradientBoostingRegressor(random_state=seed, early_stopping=False),
    }
    return estimators, grids


def n_combos(grid):
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


# ----------------------------------------------------------------------------
# 4. Charts
# ----------------------------------------------------------------------------
def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_charts(out_dir, results_df, preds, best_name, imp_df, focus_link, horizon_min):
    # model comparison
    order = results_df.sort_values("rmse")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2f5496" if m == best_name else ("#a6a6a6" if m in ("persistence", "moving_avg_3") else "#8faadc")
              for m in order["model"]]
    bars = ax.bar(order["model"], order["rmse"], color=colors)
    for b, v in zip(bars, order["rmse"]):
        ax.text(b.get_x() + b.get_width() / 2, v, "%.2f" % v, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("RMSE on test period (utilization %-points, lower is better)")
    ax.set_title("Model comparison, forecasting %d minutes ahead (grey = naive baselines, dark = chosen)" % horizon_min, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=20)
    save_fig(fig, out_dir / "fig_model_comparison.png")

    # focus link over time
    g = preds[preds["link_id"] == focus_link].sort_values("target_dt")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(g["target_dt"], g["actual"], color="black", linewidth=1.2, label="actual")
    ax.plot(g["target_dt"], g["persistence"], color="#a6a6a6", linewidth=1, label="persistence baseline")
    ax.plot(g["target_dt"], g[best_name], color="#c00000", linewidth=1, label="best model (%s)" % best_name)
    ax.set_title("Test period, link %s: actual vs predicted utilization (%d min ahead)" % (focus_link, horizon_min))
    ax.set_ylabel("utilization (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, out_dir / "fig_focus_link_prediction.png")

    # scatter
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(preds["actual"], preds[best_name], s=5, alpha=0.4)
    lim = [0, max(preds["actual"].max(), preds[best_name].max()) * 1.05]
    ax.plot(lim, lim, color="red", linewidth=1)
    ax.set_xlabel("actual utilization (%)")
    ax.set_ylabel("predicted utilization (%)")
    ax.set_title("Predicted vs actual (test)  [red = perfect]", fontsize=10)
    save_fig(fig, out_dir / "fig_predicted_vs_actual.png")

    # feature importance
    top = imp_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], color="#2f5496")
    ax.set_xlabel("increase in RMSE when the feature is shuffled")
    ax.set_title("Which features matter most (permutation importance, test)", fontsize=10)
    save_fig(fig, out_dir / "fig_feature_importance.png")

    # error by link
    pl = []
    for lid, gg in preds.groupby("link_id"):
        pl.append((lid, rmse(gg["actual"], gg["persistence"]), rmse(gg["actual"], gg[best_name])))
    pl = pd.DataFrame(pl, columns=["link_id", "persistence", "model"]).sort_values("persistence")
    idx = np.arange(len(pl))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(idx - 0.2, pl["persistence"], height=0.4, color="#a6a6a6", label="persistence")
    ax.barh(idx + 0.2, pl["model"], height=0.4, color="#c00000", label="best model")
    ax.set_yticks(idx)
    ax.set_yticklabels(pl["link_id"], fontsize=8)
    ax.set_xlabel("RMSE (utilization %-points)")
    ax.set_title("Test error per link", fontsize=10)
    ax.legend()
    save_fig(fig, out_dir / "fig_error_by_link.png")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train link-utilization forecasting models")
    ap.add_argument("--csv", default=str(HERE.parent / "data" / "link_timeseries.csv"))
    ap.add_argument("--horizon", type=int, default=1, help="steps ahead to predict (1 step = 5 min)")
    ap.add_argument("--train-frac", type=float, default=0.7, help="fraction of the timeline used for training")
    ap.add_argument("--cv-splits", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="small tuning grids, much faster")
    ap.add_argument("--target", choices=["delta", "level"], default="delta",
                    help="delta: model predicts the CHANGE (future - now), recommended. "
                         "level: model predicts the future value directly")
    ap.add_argument("--no-time-features", action="store_true", help="drop hour_sin/hour_cos")
    ap.add_argument("--congestion-threshold", type=float, default=50.0,
                    help="utilization %% above which a link counts as congested")
    ap.add_argument("--high-load-threshold", type=float, default=20.0,
                    help="RMSE is also reported on rows whose true utilization is above this %%")
    ap.add_argument("--focus-link", default=None, help="link drawn in the time chart (default: busiest)")
    ap.add_argument("--use-model", choices=["auto", "ridge", "decision_tree", "random_forest", "hist_gb"],
                    default="auto", help="auto = best by cross-validation; or force one model to be saved")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="label for an experiment, e.g. level -> outputs/h3_level, model_h3_level.pkl")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--make-default", action="store_true", help="also copy the model to ml/model.pkl")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    use_time = not args.no_time_features
    horizon_min = args.horizon * F.STEP_MINUTES
    suffix = ("_" + args.tag) if args.tag else ""
    out_dir = Path(args.out_dir) if args.out_dir else HERE / "outputs" / ("h%d%s" % (args.horizon, suffix))
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 70)
    print("PHASE B: training  |  horizon = %d step(s) = %d minutes  |  target = %s  |  quick = %s"
          % (args.horizon, horizon_min, args.target, args.quick))
    print("=" * 70)

    # ---- 1. data
    df = load_data(args.csv)
    all_times = np.sort(df["dt"].drop_duplicates().to_numpy(dtype="datetime64[ns]"))
    print("[1/6] loaded %d rows, %d links, %d timestamps" % (len(df), df["link_id"].nunique(), len(all_times)))
    if df["link_id"].nunique() != 15 or len(all_times) < 200:
        print("[WARN] expected 15 links and several hundred timestamps; check Step 13 of Part 2")
    data, names = build_dataset(df, args.horizon, use_time)
    print("[2/6] built %d training examples with %d features each" % (len(data), len(names)))
    if len(data) < 500:
        sys.exit("ERROR: too few examples (%d). Use more traffic files (Step 12 of Part 2)." % len(data))

    # ---- 2. split
    train_mask, test_mask, cutoff = chronological_split(data, all_times, args.train_frac)
    Xtr, ytr = data.loc[train_mask, names].to_numpy(float), data.loc[train_mask, "target"].to_numpy(float)
    Xte, yte = data.loc[test_mask, names].to_numpy(float), data.loc[test_mask, "target"].to_numpy(float)
    te = data.loc[test_mask].reset_index(drop=True)
    print("[3/6] chronological split at timestamp #%d of %d" % (cutoff, len(all_times)))
    print("      train: targets before %s   (%d rows)" % (pd.Timestamp(all_times[cutoff]), len(ytr)))
    print("      test : origins from    %s   (%d rows)" % (pd.Timestamp(all_times[cutoff]), len(yte)))
    if len(yte) < 100:
        sys.exit("ERROR: test set too small. Use a smaller --train-frac or more data.")

    thr, hthr = args.congestion_threshold, args.high_load_threshold
    idx_now, idx_ma3 = names.index("util_now"), names.index("mean3")
    base_tr, base_te = Xtr[:, idx_now], Xte[:, idx_now]
    # delta mode: the model learns (future - now); prediction = now + learned change.
    # The error in % points is identical either way, but delta mode starts from the
    # persistence baseline and only learns corrections, which suits links of very different size.
    ytr_fit = ytr - base_tr if args.target == "delta" else ytr
    yte_fit = yte - base_te if args.target == "delta" else yte
    preds = pd.DataFrame({
        "origin_dt": te["origin_dt"], "target_dt": te["target_dt"], "link_id": te["link_id"],
        "actual": yte, "persistence": Xte[:, idx_now], "moving_avg_3": Xte[:, idx_ma3],
    })
    results = {
        "persistence": full_metrics(yte, preds["persistence"], thr, hthr),
        "moving_avg_3": full_metrics(yte, preds["moving_avg_3"], thr, hthr),
    }

    # ---- 3. tuning and training
    print("[4/6] tuning models (time-ordered cross-validation, %d folds)" % args.cv_splits)
    estimators, grids = get_models(args.quick, args.seed)
    cv = TimeSeriesSplit(n_splits=args.cv_splits)
    fitted, cv_rmse, best_params = {}, {}, {}
    for name in ["ridge", "decision_tree", "random_forest", "hist_gb"]:
        t0 = time.time()
        gs = GridSearchCV(estimators[name], grids[name], cv=cv,
                          scoring="neg_root_mean_squared_error", n_jobs=1, refit=True)
        gs.fit(Xtr, ytr_fit)
        fitted[name] = gs.best_estimator_
        cv_rmse[name] = float(-gs.best_score_)
        best_params[name] = {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                             for k, v in gs.best_params_.items()}
        p = fitted[name].predict(Xte)
        if args.target == "delta":
            p = p + base_te
        preds[name] = p
        results[name] = full_metrics(yte, p, thr, hthr)
        print("      %-14s %3d settings tried | CV rmse %7.3f | TEST rmse %7.3f | %4.0f s"
              % (name, n_combos(grids[name]), cv_rmse[name], results[name]["rmse"], time.time() - t0))

    # ---- 4. choose the best using ONLY training-data cross-validation
    best_name = min(cv_rmse, key=cv_rmse.get)
    if args.use_model != "auto":
        best_name = args.use_model
    best_model = fitted[best_name]
    base_rmse = results["persistence"]["rmse"]
    improvement = 100.0 * (base_rmse - results[best_name]["rmse"]) / base_rmse if base_rmse > 0 else float("nan")
    how = "forced by --use-model" if args.use_model != "auto" else "chosen by cross-validation on training data"
    print("[5/6] model to save (%s): %s  %s" % (how, best_name, best_params[best_name]))

    # ---- results table
    table = pd.DataFrame([dict(model=m, **v) for m, v in results.items()])
    table.insert(1, "cv_rmse", [cv_rmse.get(m, np.nan) for m in table["model"]])
    table = table.sort_values("rmse").reset_index(drop=True)
    table.to_csv(out_dir / "metrics.csv", index=False)
    print()
    print("TEST RESULTS (last part of the timeline, never used for training)")
    show = table[["model", "cv_rmse", "rmse", "mae", "r2", "rmse_high_load", "cong_precision", "cong_recall"]]
    print(show.to_string(index=False, float_format=lambda v: "%.3f" % v))
    print("(rmse/mae in utilization %%-points; lower is better. rmse_high_load = only rows with true utilization > %.0f%%.)" % hthr)

    # ---- per link
    pl = []
    for lid, g in preds.groupby("link_id"):
        pl.append({"link_id": lid, "rmse_persistence": rmse(g["actual"], g["persistence"]),
                   "rmse_model": rmse(g["actual"], g[best_name]), "mae_model": mae(g["actual"], g[best_name]),
                   "mean_actual": g["actual"].mean()})
    pl = pd.DataFrame(pl)
    pl["improvement_pct"] = 100.0 * (pl["rmse_persistence"] - pl["rmse_model"]) / pl["rmse_persistence"]
    pl = pl.sort_values("mean_actual", ascending=False)
    pl.to_csv(out_dir / "per_link_metrics.csv", index=False)
    print()
    print("PER-LINK TEST ERROR of the best model vs persistence")
    print(pl.to_string(index=False, float_format=lambda v: "%.3f" % v))

    # ---- importance, files
    print()
    print("[6/6] computing feature importance and saving files ...")
    pi = permutation_importance(best_model, Xte, yte_fit, n_repeats=5, random_state=args.seed,
                                scoring="neg_root_mean_squared_error", n_jobs=1)
    imp = pd.DataFrame({"feature": names, "importance": pi.importances_mean, "std": pi.importances_std})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(out_dir / "feature_importance.csv", index=False)
    preds.to_csv(out_dir / "predictions_test.csv", index=False)

    focus = args.focus_link or str(pl.iloc[0]["link_id"])
    if not args.no_plots:
        make_charts(out_dir, table, preds, best_name, imp, focus, horizon_min)

    bundle = {
        "model": best_model, "model_name": best_name, "feature_names": names,
        "window": F.WINDOW, "horizon": args.horizon, "horizon_minutes": horizon_min,
        "use_time_features": use_time, "step_minutes": F.STEP_MINUTES,
        "target_mode": args.target,
        "best_params": best_params[best_name], "test_metrics": results[best_name],
        "trained_until": str(pd.Timestamp(all_times[cutoff])),
        "sklearn_version": sklearn.__version__,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    model_path = HERE / ("model_h%d%s.pkl" % (args.horizon, suffix))
    joblib.dump(bundle, model_path)
    if args.make_default:
        shutil.copyfile(model_path, HERE / "model.pkl")

    summary = {
        "horizon_steps": args.horizon, "horizon_minutes": horizon_min, "use_time_features": use_time,
        "target_mode": args.target, "train_frac": args.train_frac, "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "test_starts": str(pd.Timestamp(all_times[cutoff])),
        "best_model": best_name, "best_params": best_params[best_name],
        "cv_rmse": cv_rmse, "test": results,
        "improvement_vs_persistence_pct": float(improvement),
        "congestion_threshold": thr, "focus_link": focus,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---- verdict
    print()
    print("=" * 70)
    print("SUMMARY  (horizon %d min)" % horizon_min)
    print("=" * 70)
    print("[PASS] %d training rows, %d test rows, %d features" % (len(ytr), len(yte), len(names)))
    print("[PASS] best model: %s | test RMSE %.3f | test MAE %.3f | R2 %.3f"
          % (best_name, results[best_name]["rmse"], results[best_name]["mae"], results[best_name]["r2"]))
    if improvement > 2:
        print("[PASS] beats the persistence baseline by %.1f%% (RMSE %.3f vs %.3f)" % (improvement, results[best_name]["rmse"], base_rmse))
    elif improvement > -2:
        print("[INFO] about equal to persistence (%.1f%%). Common at short horizons: try --horizon 3 or 6." % improvement)
    else:
        print("[WARN] WORSE than persistence by %.1f%%. Report it honestly and try --horizon 3 / 6 (see guide)." % (-improvement))
    print("[PASS] model saved: %s" % model_path)
    if args.make_default:
        print("[PASS] also saved as: %s" % (HERE / "model.pkl"))
    print("       tables and charts: %s" % out_dir)
    print("       total time: %.0f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
