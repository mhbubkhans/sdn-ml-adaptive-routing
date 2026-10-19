#!/usr/bin/env python3
"""
ablation_eval.py   (Phase E - Step E4)

WHAT IT DOES
  An ablation study answers: "which design choices matter?". Offline, without
  Mininet and without sudo, it repeats the policy comparison while changing ONE
  thing at a time:

      forecast horizon of the model   : model_h1.pkl (5 min), model_h3.pkl (15 min), model_h6.pkl (30 min)
      cost function                   : quadratic with several k, and the queue-delay cost

  Every setting is judged against the SAME conditions (default: 15 minutes after the
  decision), so the numbers are comparable. The static, reactive and oracle policies
  do not depend on the model; they are shown for reference at each k.

REQUIRES  the models of Phase B:  ml/model_h1.pkl, ml/model_h3.pkl, ml/model_h6.pkl
          (python3 train_model.py --horizon 1 / 3 / 6). Missing ones are skipped.

USAGE   (from ~/sdn_project/controller)
  python3 ablation_eval.py
  python3 ablation_eval.py --k-list 4,6,8 --lead 3

OUTPUTS (in ../results/)
  phase_e_ablation.csv, fig_phase_e_ablation.png
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


def main():
    ap = argparse.ArgumentParser(description="Offline ablation study")
    ap.add_argument("--csv", default=str(HERE.parent / "data" / "link_timeseries.csv"))
    ap.add_argument("--ml-dir", default=str(HERE.parent / "ml"))
    ap.add_argument("--map", default=str(HERE.parent / "topology" / "topology_map.json"))
    ap.add_argument("--k-list", default="4,6,8,12")
    ap.add_argument("--lead", type=int, default=3, help="steps after the decision at which it is judged (3 = 15 min)")
    ap.add_argument("--out-dir", default=str(HERE.parent / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topo = RC.load_topology(args.map, "delay")
    timeline, util = LM.load_timeline(args.csv)

    models = []
    for h in (1, 3, 6):
        mp = Path(args.ml_dir) / ("model_h%d.pkl" % h)
        if mp.exists():
            b = joblib.load(str(mp))
            models.append((h, str(mp), b))
        else:
            print("[INFO] %s not found, horizon %d skipped (python3 train_model.py --horizon %d)" % (mp.name, h, h))
    if not models:
        sys.exit("ERROR: no model_h*.pkl found in %s" % args.ml_dir)

    test_start = max(LM.pretty_ts_to_compact(b["trained_until"]) for _, _, b in models)
    i0 = timeline.index(test_start)
    order = sorted(topo.hosts, key=lambda x: int(x[1:]))
    pairs = [(a, b) for a in order for b in order if a != b]
    maxu = {l: max(util[t][l] for t in timeline[i0:]) for l in topo.links}
    spiking = [l for l, m in maxu.items() if m > 50]
    static_paths = OE.all_paths(topo, None)
    affected = np.array([n for n, pr in enumerate(pairs)
                         if any(l in spiking for l in topo.path_links(static_paths[pr]))], dtype=int)
    if len(affected) == 0:
        sys.exit("ERROR: no link exceeds 50% in the test period; nothing to compare")
    ks = [float(x) for x in args.k_list.split(",")]
    settings = [("quadratic", k) for k in ks] + [("queue", 0.0)]
    print("test period from %s | judged at +%d step(s) | affected pairs %d | %d models x %d cost settings"
          % (test_start, args.lead, len(affected), len(models), len(settings)))

    rows = []
    print()
    print("%-9s %-18s %9s %9s %9s %9s %14s %11s" % ("horizon", "cost", "static", "reactive", "forecast", "oracle", "changes/100", "rules/step"))
    for h, mp, b in models:
        for cm, k in settings:
            pred = UtilizationPredictor(mp, args.map)
            res, _, _ = OE.simulate(topo, pred, timeline, util, i0, args.lead, cm, k, POLICIES, pairs)
            s = OE.summarize(res, pairs, affected, POLICIES)
            m = {p: s[(p, "affected")]["mean_latency_ms"] for p in POLICIES}
            f = s[("forecast", "affected")]
            label = ("quadratic k=%g" % k) if cm == "quadratic" else "queue delay"
            print("%-9s %-18s %9.2f %9.2f %9.2f %9.2f %14.2f %11.2f" % (
                "%d min" % b["horizon_minutes"], label, m["static"], m["reactive"], m["forecast"], m["oracle"],
                f["path_changes_per_100_pair_steps"], f["rules_changed_per_step"]))
            rows.append({"horizon_min": b["horizon_minutes"], "model": b["model_name"], "cost_model": cm, "k": k,
                         "static_ms": m["static"], "reactive_ms": m["reactive"], "forecast_ms": m["forecast"],
                         "oracle_ms": m["oracle"], "path_changes_per_100": f["path_changes_per_100_pair_steps"],
                         "rules_per_step": f["rules_changed_per_step"]})

    best = min(rows, key=lambda r: r["forecast_ms"] + 0.1 * r["path_changes_per_100"])
    worst = max(rows, key=lambda r: r["forecast_ms"])
    print()
    print("(mean modelled one-way latency in ms on the affected pairs; lower is better)")
    print("BEST setting  : horizon %d min, %s, forecast %.2f ms (static %.2f, reactive %.2f, oracle %.2f)" % (
        best["horizon_min"], ("quadratic k=%g" % best["k"]) if best["cost_model"] == "quadratic" else "queue delay",
        best["forecast_ms"], best["static_ms"], best["reactive_ms"], best["oracle_ms"]))
    print("WORST setting : horizon %d min, %s, forecast %.2f ms" % (
        worst["horizon_min"], ("quadratic k=%g" % worst["k"]) if worst["cost_model"] == "quadratic" else "queue delay", worst["forecast_ms"]))
    beats = sum(1 for r in rows if r["forecast_ms"] < r["static_ms"] * 0.99)
    beats_r = sum(1 for r in rows if r["forecast_ms"] < r["reactive_ms"] * 0.99)
    print("forecast beats static in %d of %d settings and reactive in %d of %d (1%% margin)" % (beats, len(rows), beats_r, len(rows)))

    with open(out_dir / "phase_e_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.4f" % v if isinstance(v, float) else v) for k, v in r.items()})

    fig, ax = plt.subplots(figsize=(11, 4.8))
    labels, vals, cols = [], [], []
    palette = {1: "#f4b183", 3: "#c00000", 6: "#8e2c2c"}
    for r in rows:
        labels.append("%dmin\n%s" % (r["horizon_min"], ("k=%g" % r["k"]) if r["cost_model"] == "quadratic" else "queue"))
        vals.append(r["forecast_ms"])
        cols.append(palette.get(r["horizon_min"] // 5, "#c00000"))
    ax.bar(range(len(vals)), vals, color=cols)
    ref = rows[0]
    ax.axhline(ref["static_ms"], color="#7f7f7f", linestyle="--", label="static")
    ax.axhline(min(r["oracle_ms"] for r in rows), color="#70ad47", linestyle="--", label="oracle (best)")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("forecast policy: mean modelled latency (ms)")
    ax.set_title("Ablation: forecast horizon x cost function (judged %d min after the decision)" % (args.lead * 5))
    lo, hi = min(vals + [min(r["oracle_ms"] for r in rows)]), max(vals + [ref["static_ms"]])
    ax.set_ylim(max(0, lo - 0.5 * (hi - lo) - 0.5), hi + 0.3 * (hi - lo) + 0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_phase_e_ablation.png", dpi=130)
    plt.close(fig)
    print("\nSaved: %s and %s" % (out_dir / "phase_e_ablation.csv", out_dir / "fig_phase_e_ablation.png"))


if __name__ == "__main__":
    main()
