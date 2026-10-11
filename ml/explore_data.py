#!/usr/bin/env python3
"""
explore_data.py   (Phase B - Step B3)

WHAT IT DOES
  Looks at link_timeseries.csv BEFORE any machine learning, so you understand
  your data and can catch problems early:
    1. basic checks  (rows, links, time range, missing 5-minute snapshots, NaN)
    2. per-link statistics (mean, 95th percentile, max, % of time above 50% / 100%)
    3. predictability: correlation between "now" and "5 / 15 / 30 minutes later"
    4. charts saved as PNG files in ml/outputs/

USAGE   (from ~/sdn_project/ml)
  python3 explore_data.py
  python3 explore_data.py --csv ../data/link_timeseries.csv
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw to files, no screen needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = Path(__file__).resolve().parent
REQUIRED = ["timestamp", "link_id", "bandwidth_mbps", "propagation_delay_ms",
            "aggregated_load_mbps", "utilization_pct"]


def main():
    ap = argparse.ArgumentParser(description="Explore link_timeseries.csv")
    ap.add_argument("--csv", default=str(HERE.parent / "data" / "link_timeseries.csv"))
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit("ERROR: %s not found.\nHint: run Step 13 of Part 2 first (aggregate_link_load.py)." % csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit("ERROR: columns missing from the CSV: %s" % missing)
    df["dt"] = pd.to_datetime(df["timestamp"], format="%Y%m%d-%H%M")
    df = df.sort_values(["link_id", "dt"]).reset_index(drop=True)

    # ------------------------------------------------------------ 1. basic checks
    print("=" * 66)
    print("1. BASIC CHECKS")
    print("=" * 66)
    n_links = df["link_id"].nunique()
    times = np.sort(df["dt"].drop_duplicates().to_numpy(dtype="datetime64[ns]"))
    gaps = int((np.diff(times).astype("timedelta64[m]").astype(int) != 5).sum()) if len(times) > 1 else 0
    print("rows                  : %d" % len(df))
    print("links                 : %d" % n_links)
    print("distinct timestamps   : %d" % len(times))
    print("time range            : %s  ->  %s" % (pd.Timestamp(times[0]), pd.Timestamp(times[-1])))
    print("missing 5-min steps   : %d  (gaps in the timeline)" % gaps)
    print("empty (NaN) cells     : %d" % int(df[REQUIRED].isna().sum().sum()))
    print("negative utilization  : %d" % int((df["utilization_pct"] < 0).sum()))
    print("rows per link         : %s" % sorted(df.groupby("link_id").size().unique().tolist()))
    flags = []
    flags.append(("[PASS]" if n_links == 15 else "[WARN]") + " 15 links expected, found %d" % n_links)
    flags.append(("[PASS]" if len(times) >= 500 else "[WARN]") + " at least ~500 timestamps recommended, found %d" % len(times))
    flags.append(("[PASS]" if gaps == 0 else "[WARN]") + " timeline gaps: %d (training code skips windows that contain a gap)" % gaps)
    print()
    for f in flags:
        print(f)

    # ------------------------------------------------------------ 2. per-link stats
    print()
    print("=" * 66)
    print("2. PER-LINK UTILIZATION (%)  and  3. PREDICTABILITY")
    print("=" * 66)
    rows = []
    for link_id, g in df.groupby("link_id", sort=True):
        s = g["utilization_pct"].to_numpy(dtype=float)
        ac = {}
        for k in (1, 3, 6):
            ac[k] = float(np.corrcoef(s[:-k], s[k:])[0, 1]) if len(s) > k + 2 and np.std(s) > 0 else float("nan")
        rows.append({
            "link_id": link_id,
            "bw_mbps": g["bandwidth_mbps"].iloc[0],
            "mean": s.mean(), "p95": np.percentile(s, 95), "max": s.max(),
            "pct_time_>50": 100.0 * (s > 50).mean(), "pct_time_>100": 100.0 * (s > 100).mean(),
            "corr_5min": ac[1], "corr_15min": ac[3], "corr_30min": ac[6],
        })
    st = pd.DataFrame(rows).sort_values("mean", ascending=False)
    print(st.to_string(index=False, float_format=lambda v: "%.2f" % v))
    st.to_csv(out_dir / "data_summary.csv", index=False)
    print()
    print("How to read it:")
    print(" - 'max' above 100 means that link was offered more traffic than its capacity.")
    print(" - corr_5min close to 1.00 means 'the next value is almost the same as now'.")
    print("   Then the simple 'copy the current value' baseline is hard to beat for 5 minutes.")
    print(" - corr_30min lower than corr_5min means longer forecasts are harder, and that is")
    print("   where a learned model has the best chance to help.")
    hot = st.iloc[0]["link_id"]
    print()
    print("Busiest link (will be the focus link in charts): %s" % hot)

    # ------------------------------------------------------------ 4. charts
    fig, axes = plt.subplots(5, 3, figsize=(14, 12), sharex=True)
    for ax, (link_id, g) in zip(axes.ravel(), df.groupby("link_id", sort=True)):
        ax.plot(g["dt"], g["utilization_pct"], linewidth=0.8)
        ax.set_title(link_id, fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
    for ax in axes[-1]:
        for lab in ax.get_xticklabels():
            lab.set_rotation(30)
    fig.suptitle("Link utilization (%) over time, every 5 minutes", fontsize=12)
    fig.tight_layout()
    f1 = out_dir / "fig_explore_utilization.png"
    fig.savefig(f1, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    order = st.sort_values("mean")
    ax.barh(order["link_id"], order["max"], color="#f4b183", label="max")
    ax.barh(order["link_id"], order["mean"], color="#2f5496", label="mean")
    ax.axvline(100, color="red", linestyle="--", linewidth=1, label="100% capacity")
    ax.set_xlabel("utilization (%)")
    ax.set_title("Mean and maximum utilization per link")
    ax.legend()
    fig.tight_layout()
    f2 = out_dir / "fig_explore_link_summary.png"
    fig.savefig(f2, dpi=120)
    plt.close(fig)

    print()
    print("Saved:")
    print("  %s" % (out_dir / "data_summary.csv"))
    print("  %s" % f1)
    print("  %s" % f2)
    print("\nOK: next step is  python3 train_model.py --horizon 1 --quick")


if __name__ == "__main__":
    main()
