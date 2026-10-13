#!/usr/bin/env python3
"""
compare_policies.py   (Phase D - Step D7)

WHAT IT DOES
  Reads the live results of the three runs
      results/phase_d_static.csv, phase_d_reactive.csv, phase_d_forecast.csv
  (made by phase_d_tests.py) and prints ONE comparison: mean / 95th percentile /
  maximum round-trip time, packet loss and number of path changes per watched
  host pair and over all pairs, plus the improvement of the forecast policy.
  It saves results/phase_d_comparison.csv and two charts for the report.

  Missing policies are simply skipped. Runs are only comparable if you used the
  SAME replay window (ML_START, ML_STEPS) and the same pairs for every policy.

USAGE   (from ~/sdn_project/controller, no sudo)
  python3 compare_policies.py
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ORDER = ["static", "reactive", "forecast"]
COLORS = {"static": "#7f7f7f", "reactive": "#8faadc", "forecast": "#c00000"}


def pctl(values, q):
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE.parent / "results"))
    args = ap.parse_args()
    d = Path(args.dir)
    data = {p: load(d / ("phase_d_%s.csv" % p)) for p in ORDER if (d / ("phase_d_%s.csv" % p)).exists()}
    if not data:
        sys.exit("ERROR: no phase_d_<policy>.csv found in %s. Run phase_d_tests.py first." % d)
    pairs = sorted({r["pair"] for rows in data.values() for r in rows})
    steps = {p: len({r["step"] for r in rows}) for p, rows in data.items()}
    if len(set(steps.values())) > 1:
        print("[WARN] the runs have different numbers of steps %s: they may not be comparable" % steps)

    stats = {}
    for p, rows in data.items():
        for pr in pairs + ["ALL"]:
            sel = [r for r in rows if pr == "ALL" or r["pair"] == pr]
            rtt = [float(r["rtt_ms"]) for r in sel if r["rtt_ms"] != ""]
            loss = [float(r["loss_pct"]) for r in sel]
            seq = [r["path"] for r in sel]
            if pr == "ALL":
                changes = sum(1 for q in pairs for x, y in zip([r["path"] for r in rows if r["pair"] == q], [r["path"] for r in rows if r["pair"] == q][1:]) if x != y)
            else:
                changes = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
            stats[(p, pr)] = {"mean": sum(rtt) / len(rtt) if rtt else float("nan"), "p95": pctl(rtt, 95),
                              "max": max(rtt) if rtt else float("nan"), "loss": sum(loss) / len(loss) if loss else float("nan"),
                              "changes": changes, "n": len(rtt)}

    print("=" * 86)
    print("LIVE COMPARISON (round-trip time of pings, milliseconds; lower is better)")
    print("=" * 86)
    print("%-8s %-10s %9s %9s %9s %8s %13s" % ("pair", "policy", "mean", "p95", "max", "loss %", "path changes"))
    for pr in pairs + ["ALL"]:
        for p in data:
            s = stats[(p, pr)]
            print("%-8s %-10s %9.1f %9.1f %9.1f %8.1f %13d" % (pr, p, s["mean"], s["p95"], s["max"], s["loss"], s["changes"]))
        print()

    if "forecast" in data:
        for base in ("static", "reactive"):
            if base in data:
                for key, label in (("mean", "mean RTT"), ("p95", "95th percentile RTT")):
                    b, f_ = stats[(base, "ALL")][key], stats[("forecast", "ALL")][key]
                    print("forecast vs %-8s %-20s %7.1f -> %7.1f ms  (%+.1f%%)" % (base, label, b, f_, 100.0 * (f_ - b) / b))
        print("(negative percentages mean the forecast policy is faster)")

    out_csv = d / "phase_d_comparison.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "policy", "mean_rtt_ms", "p95_rtt_ms", "max_rtt_ms", "loss_pct", "path_changes", "samples"])
        for pr in pairs + ["ALL"]:
            for p in data:
                s = stats[(p, pr)]
                w.writerow([pr, p, "%.2f" % s["mean"], "%.2f" % s["p95"], "%.2f" % s["max"], "%.2f" % s["loss"], s["changes"], s["n"]])

    # chart 1: mean and p95 per pair
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = list(range(len(pairs)))
    w = 0.8 / max(1, len(data))
    for ax, key, title in ((axes[0], "mean", "Mean RTT"), (axes[1], "p95", "95th percentile RTT")):
        for n, p in enumerate(data):
            ax.bar([i + n * w for i in x], [stats[(p, pr)][key] for pr in pairs], w, color=COLORS[p], label=p)
        ax.set_xticks([i + w * (len(data) - 1) / 2 for i in x])
        ax.set_xticklabels(pairs)
        ax.set_ylabel("ms")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(d / "fig_phase_d_rtt.png", dpi=130)
    plt.close(fig)

    # chart 2: timeline of the pair with the largest spread between policies
    def spread(pr):
        vals = [stats[(p, pr)]["max"] for p in data]
        return max(vals) - min(vals)
    focus = max(pairs, key=spread)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for p, rows in data.items():
        sel = [r for r in rows if r["pair"] == focus and r["rtt_ms"] != ""]
        ax.plot([int(r["step"]) for r in sel], [float(r["rtt_ms"]) for r in sel], marker="o", markersize=3, color=COLORS[p], label=p)
    ax.set_xlabel("replay step (one step = 5 minutes of data)")
    ax.set_ylabel("round-trip time (ms)")
    ax.set_title("Live RTT for %s under the three policies" % focus)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "fig_phase_d_timeline.png", dpi=130)
    plt.close(fig)
    print("\nSaved: %s, %s, %s" % (out_csv, d / "fig_phase_d_rtt.png", d / "fig_phase_d_timeline.png"))


if __name__ == "__main__":
    main()
