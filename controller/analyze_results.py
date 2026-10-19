#!/usr/bin/env python3
"""
analyze_results.py   (Phase E - Step E7)

WHAT IT DOES
  Reads every live run made by phase_e_runner.py
      results/phase_e_<scenario>_<policy>.csv   (+ the *_events.csv of failure runs)
  and produces the final evidence for the report, for each scenario:

    - mean / 95th percentile / maximum round-trip time of the AFFECTED pairs
      (all watched pairs except the control pair), and the control pair's mean
    - packet loss, number of path changes, mean TCP throughput, reroute outage
    - the paired difference forecast - static and forecast - reactive in mean RTT
      with a 95% moving-block bootstrap interval (steps are correlated, so blocks
      of 3 steps are resampled)
    - a plain-language verdict ("clearly lower", "not clearly different", ...)

  Only steps measured under ALL policies of a scenario are compared (fair pairing).

USAGE   (from ~/sdn_project/controller, no sudo)
  python3 analyze_results.py
  python3 analyze_results.py --control h3-h9

OUTPUTS (in ../results/)
  phase_e_summary.csv, phase_e_tables.md, fig_phase_e_summary.png,
  fig_phase_e_throughput.png (if iperf data exist), fig_phase_e_failure.png (if failure runs exist)
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
POLICIES = ["static", "reactive", "forecast"]
COLORS = {"static": "#7f7f7f", "reactive": "#8faadc", "forecast": "#c00000"}


def block_bootstrap_ci(diff, block=3, n=4000, seed=1):
    diff = np.asarray(diff, dtype=float)
    m = len(diff)
    if m < block + 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(m / float(block)))
    means = np.empty(n)
    for i in range(n):
        st = rng.integers(0, m - block + 1, size=nb)
        means[i] = np.concatenate([diff[s:s + block] for s in st])[:m].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(lo, hi):
    if lo != lo:
        return "too few steps"
    return "clearly lower" if hi < 0 else ("clearly higher" if lo > 0 else "not clearly different")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pctl(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Final analysis of the Phase E runs")
    ap.add_argument("--dir", default=str(HERE.parent / "results"))
    ap.add_argument("--control", default="h3-h9")
    args = ap.parse_args()
    d = Path(args.dir)

    runs = defaultdict(dict)        # scenario -> policy -> rows
    events = defaultdict(dict)      # scenario -> policy -> event rows
    for f in sorted(d.glob("phase_e_*.csv")):
        m = re.match(r"phase_e_(.+)_(static|reactive|forecast)(_events)?\.csv$", f.name)
        if not m:
            continue
        sc, pol, ev = m.group(1), m.group(2), m.group(3)
        (events if ev else runs)[sc][pol] = read_csv(f)
    if not runs:
        sys.exit("ERROR: no phase_e_<scenario>_<policy>.csv found in %s. Run phase_e_runner.py first." % d)

    summary_rows, md = [], []
    fig_data = {}
    for sc in sorted(runs):
        pols = [p for p in POLICIES if p in runs[sc]]
        steps = None
        for p in pols:
            s = set(int(r["step"]) for r in runs[sc][p])
            steps = s if steps is None else steps & s
        steps = sorted(steps)
        print("=" * 92)
        print("SCENARIO '%s': policies %s | %d steps measured under all of them" % (sc, ", ".join(pols), len(steps)))
        print("=" * 92)
        if len(pols) < 2 or len(steps) < 5:
            print("[WARN] need at least two policies and five common steps; skipped")
            continue
        per = {}
        for p in pols:
            rows = [r for r in runs[sc][p] if int(r["step"]) in steps]
            aff = [r for r in rows if r["pair"] != args.control and r["rtt_ms"] != ""]
            ctl = [float(r["rtt_ms"]) for r in rows if r["pair"] == args.control and r["rtt_ms"] != ""]
            rtt = [float(r["rtt_ms"]) for r in aff]
            tp = [float(r["tput_mbps"]) for r in rows if r["pair"] != args.control and r["tput_mbps"] != ""]
            changes = 0
            for pr in sorted(set(r["pair"] for r in rows if r["pair"] != args.control)):
                seq = [r["path"] for r in sorted((r for r in rows if r["pair"] == pr), key=lambda r: int(r["step"]))]
                changes += sum(1 for a, b in zip(seq, seq[1:]) if a != b)
            outage = [float(e["outage_ms"]) for e in events.get(sc, {}).get(p, []) if e["event"] == "link_down"]
            by_step = defaultdict(list)
            for r in aff:
                by_step[int(r["step"])].append(float(r["rtt_ms"]))
            per[p] = {"mean": float(np.mean(rtt)), "p95": pctl(rtt, 95), "max": max(rtt), "ctl": float(np.mean(ctl)) if ctl else float("nan"),
                      "loss": float(np.mean([float(r["loss_pct"]) for r in rows])), "changes": changes,
                      "tput": float(np.mean(tp)) if tp else float("nan"), "outage": outage[0] if outage else float("nan"),
                      "step_mean": np.array([np.mean(by_step[s]) if by_step[s] else np.nan for s in steps])}
        print("%-10s %9s %9s %9s %12s %8s %9s %11s %11s" % ("policy", "mean ms", "p95 ms", "max ms", "control ms", "loss %", "changes", "Mbps", "outage ms"))
        for p in pols:
            x = per[p]
            print("%-10s %9.1f %9.1f %9.1f %12.1f %8.1f %9d %11s %11s" % (
                p, x["mean"], x["p95"], x["max"], x["ctl"], x["loss"], x["changes"],
                "%.1f" % x["tput"] if x["tput"] == x["tput"] else "-", "%.0f" % x["outage"] if x["outage"] == x["outage"] else "-"))
            summary_rows.append([sc, p, "%.2f" % x["mean"], "%.2f" % x["p95"], "%.2f" % x["max"], "%.2f" % x["ctl"], "%.2f" % x["loss"],
                                 x["changes"], "%.2f" % x["tput"] if x["tput"] == x["tput"] else "", "%.0f" % x["outage"] if x["outage"] == x["outage"] else ""])
        md.append("### Scenario: %s\n\n| Policy | Mean RTT (ms) | p95 RTT (ms) | Max RTT (ms) | Control pair (ms) | Loss (%%) | Path changes | Throughput (Mbps) | Reroute outage (ms) |\n|---|---|---|---|---|---|---|---|---|" % sc)
        for p in pols:
            x = per[p]
            md.append("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %d | %s | %s |" % (
                p, x["mean"], x["p95"], x["max"], x["ctl"], x["loss"], x["changes"],
                "%.1f" % x["tput"] if x["tput"] == x["tput"] else "-", "%.0f" % x["outage"] if x["outage"] == x["outage"] else "-"))
        md.append("")

        if "forecast" in pols:
            print()
            print("PAIRED DIFFERENCE in mean RTT (affected pairs), 95% block-bootstrap interval:")
            md.append("| Comparison | Difference in mean RTT (ms) | 95% interval | Verdict |\n|---|---|---|---|")
            for other in ("static", "reactive"):
                if other not in pols:
                    continue
                diff = per["forecast"]["step_mean"] - per[other]["step_mean"]
                diff = diff[~np.isnan(diff)]
                lo, hi = block_bootstrap_ci(diff)
                v = verdict(lo, hi)
                print("   forecast - %-8s = %+7.2f ms   [%+7.2f, %+7.2f]   forecast is %s" % (other, diff.mean(), lo, hi, v))
                md.append("| forecast - %s | %+.2f | [%+.2f, %+.2f] | forecast is %s |" % (other, diff.mean(), lo, hi, v))
            md.append("")
            if per["forecast"]["tput"] == per["forecast"]["tput"] and "static" in pols and per["static"]["tput"] == per["static"]["tput"]:
                print("   mean throughput: static %.1f | %s | forecast %.1f Mbps" % (
                    per["static"]["tput"], ("reactive %.1f" % per["reactive"]["tput"]) if "reactive" in per else "reactive -", per["forecast"]["tput"]))
        print()
        fig_data[sc] = (pols, per, steps)

    # ------------------------------------------------------------------ files
    with open(d / "phase_e_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "policy", "mean_rtt_ms", "p95_rtt_ms", "max_rtt_ms", "control_rtt_ms", "loss_pct", "path_changes", "throughput_mbps", "reroute_outage_ms"])
        w.writerows(summary_rows)
    with open(d / "phase_e_tables.md", "w") as f:
        f.write("\n".join(md) + "\n")

    scs = sorted(fig_data)
    if scs:
        fig, axes = plt.subplots(1, len(scs), figsize=(5.2 * len(scs), 4.5), squeeze=False)
        for ax, sc in zip(axes[0], scs):
            pols, per, _ = fig_data[sc]
            x = np.arange(len(pols))
            ax.bar(x - 0.2, [per[p]["mean"] for p in pols], 0.4, color=[COLORS[p] for p in pols], label="mean")
            ax.bar(x + 0.2, [per[p]["p95"] for p in pols], 0.4, color=[COLORS[p] for p in pols], alpha=0.5, label="p95")
            ax.set_xticks(x)
            ax.set_xticklabels(pols)
            ax.set_ylabel("round-trip time (ms)")
            ax.set_title("%s\n(solid = mean, light = p95)" % sc, fontsize=9)
        fig.tight_layout()
        fig.savefig(d / "fig_phase_e_summary.png", dpi=130)
        plt.close(fig)

    tp_sc = [sc for sc in scs if any(fig_data[sc][1][p]["tput"] == fig_data[sc][1][p]["tput"] for p in fig_data[sc][0])]
    if tp_sc:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sc = tp_sc[0]
        pols, per, _ = fig_data[sc]
        ax.bar(pols, [per[p]["tput"] if per[p]["tput"] == per[p]["tput"] else 0 for p in pols], color=[COLORS[p] for p in pols])
        ax.set_ylabel("mean TCP throughput of the iperf pairs (Mbps)")
        ax.set_title("Throughput under replayed load (%s)" % sc)
        fig.tight_layout()
        fig.savefig(d / "fig_phase_e_throughput.png", dpi=130)
        plt.close(fig)

    fl = [sc for sc in scs if events.get(sc)]
    if fl:
        sc = fl[0]
        pols = fig_data[sc][0]
        pair = next(iter(events[sc].values()))[0]["pair"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 1]})
        down_steps = set()
        for p in pols:
            sel = [r for r in runs[sc][p] if r["pair"] == pair and r["rtt_ms"] != ""]
            axes[0].plot([int(r["step"]) for r in sel], [float(r["rtt_ms"]) for r in sel], marker="o", markersize=3, color=COLORS[p], label=p)
            down_steps |= set(int(r["step"]) for r in runs[sc][p] if r["links_down"] != "")
        if down_steps:
            axes[0].axvspan(min(down_steps), max(down_steps), color="#f4cccc", alpha=0.5, label="link down")
        axes[0].set_xlabel("replay step")
        axes[0].set_ylabel("round-trip time (ms)")
        axes[0].set_title("Failure scenario '%s': RTT of %s" % (sc, pair))
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        outs = [(p, float(next((e["outage_ms"] for e in events[sc].get(p, []) if e["event"] == "link_down"), "nan"))) for p in pols]
        axes[1].bar([o[0] for o in outs], [0 if o[1] != o[1] else o[1] for o in outs], color=[COLORS[o[0]] for o in outs])
        axes[1].set_ylabel("reroute outage (ms)")
        axes[1].set_title("Outage after the failure")
        fig.tight_layout()
        fig.savefig(d / "fig_phase_e_failure.png", dpi=130)
        plt.close(fig)

    print("Saved: %s, %s and the charts (fig_phase_e_*.png) in %s" % (d / "phase_e_summary.csv", d / "phase_e_tables.md", d))


if __name__ == "__main__":
    main()
