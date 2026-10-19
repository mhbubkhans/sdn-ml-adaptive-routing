#!/usr/bin/env python3
"""
multi_window_eval.py   (Phase E - Step E3)

WHAT IT DOES
  One live run covers only one replay window. This script (offline, no Mininet, no
  sudo) checks whether your result is a coincidence: it replays the WHOLE test
  period once and then reports, for the N biggest traffic surges on the busiest
  link, how the policies compare inside each surge window, and over the whole
  test period. It also gives a confidence interval for the difference between
  policies.

  Policies: static (delay only), reactive (utilization now), forecast (ML),
  oracle (true future load, an upper bound). Judged as in offline_eval.py:
  modelled path latency against the conditions at t + LEAD, affected pairs only.

  CONFIDENCE INTERVAL: moving-block bootstrap over the steps (blocks of 3 steps,
  because neighbouring steps are correlated), 4000 resamples.

USAGE   (from ~/sdn_project/controller)
  python3 multi_window_eval.py --k 6
  python3 multi_window_eval.py --k 6 --windows 8 --window-steps 30

OUTPUTS (in ../results/)
  phase_e_multi_window.csv, fig_phase_e_windows.png
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ml"))
import routing_core as RC   # noqa: E402
import load_model as LM     # noqa: E402
import offline_eval as OE   # noqa: E402
from predictor import UtilizationPredictor   # noqa: E402

POLICIES = ["static", "reactive", "forecast", "oracle"]
COLORS = {"static": "#7f7f7f", "reactive": "#8faadc", "forecast": "#c00000", "oracle": "#70ad47"}


def block_bootstrap_ci(diff, block=3, n=4000, seed=1):
    """95% CI of the mean of a (correlated) series by moving-block bootstrap."""
    diff = np.asarray(diff, dtype=float)
    m = len(diff)
    if m < block + 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(m / float(block)))
    means = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, m - block + 1, size=n_blocks)
        sample = np.concatenate([diff[s:s + block] for s in starts])[:m]
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser(description="Offline evaluation over several surge windows")
    ap.add_argument("--csv", default=str(HERE.parent / "data" / "link_timeseries.csv"))
    ap.add_argument("--model", default=str(HERE.parent / "ml" / "model.pkl"))
    ap.add_argument("--map", default=str(HERE.parent / "topology" / "topology_map.json"))
    ap.add_argument("--cost-model", choices=["quadratic", "queue"], default="quadratic")
    ap.add_argument("--k", type=float, default=6.0)
    ap.add_argument("--lead", type=int, default=None)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--window-steps", type=int, default=40)
    ap.add_argument("--out-dir", default=str(HERE.parent / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topo = RC.load_topology(args.map, "delay")
    bundle = joblib.load(args.model)
    lead = args.lead if args.lead is not None else bundle["horizon"]
    timeline, util = LM.load_timeline(args.csv)
    i0 = timeline.index(LM.pretty_ts_to_compact(bundle["trained_until"]))
    order = sorted(topo.hosts, key=lambda x: int(x[1:]))
    pairs = [(a, b) for a in order for b in order if a != b]

    maxu = {l: max(util[t][l] for t in timeline[i0:]) for l in topo.links}
    meanu = {l: sum(util[t][l] for t in timeline[i0:]) / float(len(timeline) - i0) for l in topo.links}
    spiking = [l for l, m in maxu.items() if m > 50]
    static_paths = OE.all_paths(topo, None)
    affected = np.array([n for n, pr in enumerate(pairs)
                         if any(l in spiking for l in topo.path_links(static_paths[pr]))], dtype=int)
    hot = max(meanu, key=meanu.get)
    if len(affected) == 0:
        sys.exit("ERROR: no link exceeds 50% in the test period; nothing to compare")
    print("model %s | judged at +%d step(s) | %s cost, k=%g | affected pairs %d | busiest link %s"
          % (bundle["model_name"], lead, args.cost_model, args.k, len(affected), hot))
    sys.stdout.write("replaying the whole test period ... ")
    sys.stdout.flush()
    pred = UtilizationPredictor(args.model, args.map)
    _, series, _ = OE.simulate(topo, pred, timeline, util, i0, lead, args.cost_model, args.k, POLICIES, pairs)
    print("done")

    # per-step mean latency over the affected pairs
    step_lat = {p: np.array(series[p])[:, affected].mean(axis=1) for p in POLICIES}
    n_steps = len(step_lat["static"])
    hot_series = np.array([util[timeline[i0 + s]][hot] for s in range(n_steps)])

    # choose surge windows: greedy on the peaks of the busiest link, no overlap
    W = min(args.window_steps, n_steps)
    half = W // 2
    taken, windows = [], []
    for s in np.argsort(-hot_series):
        if len(windows) >= args.windows:
            break
        if all(abs(int(s) - t) >= W for t in taken):
            lo = int(min(max(s - half, 0), n_steps - W))
            windows.append((lo, lo + W, int(s)))
            taken.append(int(s))
    windows.sort()

    rows = []
    print()
    print("%-9s %-13s %8s %9s %9s %9s %9s %12s %12s" % ("window", "start", "peak %", "static", "reactive", "forecast", "oracle", "fc vs static", "fc vs react."))
    for n, (lo, hi, pk) in enumerate(windows, start=1):
        m = {p: float(step_lat[p][lo:hi].mean()) for p in POLICIES}
        vs_s = 100.0 * (m["forecast"] - m["static"]) / m["static"]
        vs_r = 100.0 * (m["forecast"] - m["reactive"]) / m["reactive"]
        ts = timeline[i0 + lo]
        print("%-9s %-13s %8.0f %9.2f %9.2f %9.2f %9.2f %11.1f%% %11.1f%%" % (
            "W%d" % n, ts, hot_series[pk], m["static"], m["reactive"], m["forecast"], m["oracle"], vs_s, vs_r))
        rows.append(["W%d" % n, ts, "%.0f" % hot_series[pk]] + ["%.3f" % m[p] for p in POLICIES] + ["%.2f" % vs_s, "%.2f" % vs_r])
    allm = {p: float(step_lat[p].mean()) for p in POLICIES}
    print("%-9s %-13s %8s %9.2f %9.2f %9.2f %9.2f %11.1f%% %11.1f%%" % (
        "ALL", "test period", "-", allm["static"], allm["reactive"], allm["forecast"], allm["oracle"],
        100 * (allm["forecast"] - allm["static"]) / allm["static"], 100 * (allm["forecast"] - allm["reactive"]) / allm["reactive"]))
    rows.append(["ALL", "test period", ""] + ["%.3f" % allm[p] for p in POLICIES] +
                ["%.2f" % (100 * (allm["forecast"] - allm["static"]) / allm["static"]),
                 "%.2f" % (100 * (allm["forecast"] - allm["reactive"]) / allm["reactive"])])
    print("(modelled one-way latency in ms, affected pairs; negative percentages mean the forecast policy is faster)")

    print()
    print("DIFFERENCE IN MEAN LATENCY over the whole test period, with 95% block-bootstrap interval (ms):")
    verdicts = {}
    for other in ("static", "reactive"):
        d = step_lat["forecast"] - step_lat[other]
        lo_ci, hi_ci = block_bootstrap_ci(d)
        sig = "clearly lower" if hi_ci < 0 else ("clearly higher" if lo_ci > 0 else "not clearly different")
        verdicts[other] = sig
        print("   forecast - %-8s = %+.3f ms   [%+.3f, %+.3f]   -> forecast is %s" % (other, d.mean(), lo_ci, hi_ci, sig))
    wins_s = sum(1 for r in rows[:-1] if float(r[-2]) < 0)
    wins_r = sum(1 for r in rows[:-1] if float(r[-1]) < 0)
    print("windows where forecast beats static: %d of %d | beats reactive: %d of %d" % (wins_s, len(windows), wins_r, len(windows)))

    with open(out_dir / "phase_e_multi_window.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "start", "peak_util_pct"] + ["%s_ms" % p for p in POLICIES] + ["forecast_vs_static_pct", "forecast_vs_reactive_pct"])
        w.writerows(rows)

    labels = [r[0] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bw = 0.2
    for i, p in enumerate(POLICIES):
        ax.bar(x + (i - 1.5) * bw, [float(r[3 + i]) for r in rows], bw, color=COLORS[p], label=p)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean modelled latency, affected pairs (ms)")
    ax.set_title("Policies in each surge window and over the whole test period (k=%g, %s)" % (args.k, args.cost_model))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_phase_e_windows.png", dpi=130)
    plt.close(fig)
    print("\nSaved: %s and %s" % (out_dir / "phase_e_multi_window.csv", out_dir / "fig_phase_e_windows.png"))


if __name__ == "__main__":
    main()
