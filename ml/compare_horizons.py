#!/usr/bin/env python3
"""
compare_horizons.py   (Phase B - Step B6)

WHAT IT DOES
  After you ran train_model.py for several horizons (for example 1, 3 and 6),
  this script reads their metrics.json files and prints ONE table:
  "how well does the best model do when predicting 5, 15 and 30 minutes ahead,
  compared with the simple persistence baseline?"
  It also saves a chart for your report.

USAGE   (from ~/sdn_project/ml)
  python3 compare_horizons.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main():
    files = [f for f in OUT.glob("h*/metrics.json") if re.match(r"^h\d+(_.+)?$", f.parent.name)]
    files = sorted(files, key=lambda p: (int(re.match(r"h(\d+)", p.parent.name).group(1)), p.parent.name))
    if not files:
        sys.exit("ERROR: no results found. Run train_model.py first (for example --horizon 1).")
    rows = []
    for f in files:
        with open(f) as fh:
            m = json.load(fh)
        best = m["best_model"]
        t = m["test"][best]
        base = m["test"]["persistence"]
        variant = re.sub(r"^h\d+_?", "", f.parent.name) or "main"
        rows.append({
            "variant": variant, "horizon_min": m["horizon_minutes"], "best_model": best,
            "model_rmse": t["rmse"], "persistence_rmse": base["rmse"],
            "improvement_%": m["improvement_vs_persistence_pct"],
            "model_mae": t["mae"], "persistence_mae": base["mae"],
            "model_r2": t["r2"], "cong_recall": t["cong_recall"], "cong_precision": t["cong_precision"],
        })
    all_df = pd.DataFrame(rows)
    all_df.to_csv(OUT / "horizon_comparison.csv", index=False)
    df = all_df[all_df["variant"] == "main"].reset_index(drop=True)
    if df.empty:
        sys.exit("ERROR: no main runs found (outputs/h1, h3, h6). Run train_model.py without --tag first.")
    print("=" * 78)
    print("HORIZON COMPARISON (test period; RMSE/MAE in utilization %-points, lower = better)")
    print("=" * 78)
    print(df.drop(columns=["variant"]).to_string(index=False, float_format=lambda v: "%.3f" % v))
    extra = all_df[all_df["variant"] != "main"]
    if len(extra):
        print()
        print("EXPERIMENT VARIANTS (made with --tag)")
        print(extra[["variant", "horizon_min", "best_model", "model_rmse", "persistence_rmse", "improvement_%"]]
              .to_string(index=False, float_format=lambda v: "%.3f" % v))
    print()
    print("improvement_% > 0 means the learned model beats 'copy the current value'.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = range(len(df))
    w = 0.38
    axes[0].bar([i - w / 2 for i in x], df["persistence_rmse"], w, color="#a6a6a6", label="persistence")
    axes[0].bar([i + w / 2 for i in x], df["model_rmse"], w, color="#2f5496", label="best model")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(["%d min" % h for h in df["horizon_min"]])
    axes[0].set_ylabel("test RMSE (%-points)")
    axes[0].set_title("Error by forecast horizon")
    axes[0].legend()
    axes[1].bar(list(x), df["improvement_%"], color=["#2f5496" if v > 0 else "#c00000" for v in df["improvement_%"]])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(["%d min" % h for h in df["horizon_min"]])
    axes[1].set_ylabel("improvement over persistence (%)")
    axes[1].set_title("Is the model better than copying 'now'?")
    fig.tight_layout()
    fig.savefig(OUT / "fig_horizon_comparison.png", dpi=130)
    plt.close(fig)

    pos = df[df["improvement_%"] > 2]
    print()
    if len(pos):
        if 15 in set(pos["horizon_min"]):
            pick = 15
            why = "15 minutes is a good compromise for routing: far enough to act early, short enough to stay accurate."
        else:
            pick = int(pos.sort_values("horizon_min").iloc[-1]["horizon_min"])
            why = "it is the longest horizon where the model clearly beats persistence."
        print("SUGGESTION: use the %d-minute horizon: %s" % (pick, why))
        print("Make it the default model:  python3 train_model.py --horizon %d --make-default" % (pick // 5))
    else:
        print("SUGGESTION: the model does not clearly beat persistence at any horizon you ran.")
        print("That is an honest finding for the report. For the controller you can still use")
        print("the model (it is not worse) or fall back to persistence. See the guide, Section 7.")
    print("\nSaved: %s and %s" % (OUT / "horizon_comparison.csv", OUT / "fig_horizon_comparison.png"))


if __name__ == "__main__":
    main()
