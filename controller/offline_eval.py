#!/usr/bin/env python3
"""
offline_eval.py   (Phase D - Steps D3, D4)

WHAT IT DOES
  Replays the TEST period of your data step by step, exactly as the ML controller
  will, and compares four routing policies for all 132 host pairs WITHOUT Mininet:

    static    cost = link delay only                    (the Phase C baseline)
    reactive  cost uses the utilization measured NOW    (no ML)
    forecast  cost uses the ML forecast of utilization  (the project's idea)
    oracle    cost uses the TRUE future utilization     (upper bound, not realistic)

  A decision taken at moment t is judged against the real conditions at t + LEAD
  (by default the model's forecast horizon), using the load model of load_model.py:
  latency of a path = sum over its links of (delay + queue delay).

  It also
    - sweeps the cost parameter k to find a good value,
    - recommends a replay window containing the biggest traffic surge, for the demo.

USAGE   (from ~/sdn_project/controller, no sudo)
  python3 offline_eval.py                       # evaluate with defaults, sweep k
  python3 offline_eval.py --k 8                 # evaluate one k only (no sweep)
  python3 offline_eval.py --cost-model queue    # use the queue-delay cost instead
  python3 offline_eval.py --lead 1              # judge against the next 5 minutes

OUTPUTS (in ../results/)
  phase_d_offline_eval.csv, fig_offline_policies.png, fig_offline_k_sweep.png,
  fig_offline_timeline.png
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
from predictor import UtilizationPredictor   # noqa: E402
import features as F        # noqa: E402

POLICIES = ["static", "reactive", "forecast", "oracle"]


def all_paths(topo, weights):
    """{(src_host, dst_host): [switch names]} using one Dijkstra per destination switch."""
    out = {}
    for b, hb in topo.hosts.items():
        root = topo.dpid_of[hb["switch"]]
        _, nxt = topo._dijkstra(root, set(), weights)
        for a, ha in topo.hosts.items():
            if a == b:
                continue
            cur = topo.dpid_of[ha["switch"]]
            p = [topo.name_of[cur]]
            while cur != root:
                cur = nxt[cur]
                p.append(topo.name_of[cur])
            out[(a, b)] = p
    return out


def weights_for(topo, source_util, model, k):
    return {lid: LM.link_cost(l["delay_ms"], source_util[lid], model, k) for lid, l in topo.links.items()}


def simulate(topo, pred, timeline, util, i0, lead, model, k, policies, pairs):
    """Run the replay for one (model, k) setting. Returns per-policy statistics lists."""
    res = {p: {"lat": [], "maxu": [], "changes": 0, "rules": 0, "steps": 0,
               "prev_paths": None, "prev_flows": None} for p in policies}
    series = {p: [] for p in policies}            # latency timeline for the timeline chart
    fc_log = []
    for j in range(i0 - (F.WINDOW - 1), i0):       # warm-up history
        pred.update_all(util[timeline[j]], timeline[j])
    last = len(timeline) - 1 - lead
    for i in range(i0, last + 1):
        ts = timeline[i]
        pred.update_all(util[ts], ts)
        fc = pred.predict_all()
        fc_log.append((ts, fc))
        actual = util[timeline[i + lead]]
        wsrc = {"static": None,
                "reactive": weights_for(topo, util[ts], model, k),
                "forecast": weights_for(topo, fc, model, k),
                "oracle": weights_for(topo, actual, model, k)}
        for p in policies:
            r = res[p]
            paths = all_paths(topo, wsrc[p])
            flows = topo.compute_flows((), wsrc[p])
            if r["prev_flows"] is not None:
                keys = set(flows) | set(r["prev_flows"])
                r["rules"] += sum(1 for kk in keys if flows.get(kk) != r["prev_flows"].get(kk))
            if r["prev_paths"] is not None:
                r["changes"] += sum(1 for pr in pairs if paths[pr] != r["prev_paths"][pr])
            r["prev_paths"], r["prev_flows"] = paths, flows
            r["steps"] += 1
            lats = []
            for pr in pairs:
                pth = paths[pr]
                lat = LM.path_latency_ms(topo, pth, actual)
                r["lat"].append(lat)
                r["maxu"].append(LM.path_max_util(topo, pth, actual))
                lats.append(lat)
            series[p].append(lats)
    return res, series, fc_log


def summarize(res, pairs, affected_idx, policies):
    rows = {}
    npairs = len(pairs)
    for p in policies:
        r = res[p]
        lat = np.array(r["lat"]).reshape(-1, npairs)
        mu = np.array(r["maxu"]).reshape(-1, npairs)
        for subset, idx in (("all", np.arange(npairs)), ("affected", affected_idx)):
            if len(idx) == 0:
                continue
            L, U = lat[:, idx], mu[:, idx]
            rows[(p, subset)] = {
                "mean_latency_ms": float(L.mean()), "p95_latency_ms": float(np.percentile(L, 95)),
                "mean_path_max_util": float(U.mean()),
                "pct_exposed_80": float(100.0 * (U > 80).mean()), "pct_exposed_100": float(100.0 * (U > 100).mean()),
                "path_changes_per_100_pair_steps": float(100.0 * r["changes"] / max(1, (r["steps"] - 1) * npairs)),
                "rules_changed_per_step": float(r["rules"] / max(1, r["steps"] - 1)),
            }
    return rows


def main():
    ap = argparse.ArgumentParser(description="Offline comparison of routing policies")
    ap.add_argument("--csv", default=str(HERE.parent / "data" / "link_timeseries.csv"))
    ap.add_argument("--model", default=str(HERE.parent / "ml" / "model.pkl"))
    ap.add_argument("--map", default=str(HERE.parent / "topology" / "topology_map.json"))
    ap.add_argument("--cost-model", choices=["quadratic", "queue"], default="quadratic")
    ap.add_argument("--k", type=float, default=None, help="evaluate only this k (default: sweep)")
    ap.add_argument("--k-list", default="1,2,4,6,8,12,16")
    ap.add_argument("--lead", type=int, default=None, help="steps ahead the decision is judged (default: model horizon)")
    ap.add_argument("--window-steps", type=int, default=40)
    ap.add_argument("--out-dir", default=str(HERE.parent / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topo = RC.load_topology(args.map, "delay")
    bundle = joblib.load(args.model)
    horizon = bundle["horizon"]
    lead = args.lead if args.lead is not None else horizon
    timeline, util = LM.load_timeline(args.csv)
    test_start = LM.pretty_ts_to_compact(bundle["trained_until"])
    if test_start not in util:
        sys.exit("ERROR: test start %s is not in the CSV timeline" % test_start)
    i0 = timeline.index(test_start)
    if i0 < F.WINDOW:
        sys.exit("ERROR: not enough history before the test period")

    order = sorted(topo.hosts, key=lambda x: int(x[1:]))
    pairs = [(a, b) for a in order for b in order if a != b]
    n_steps = len(timeline) - 1 - lead - i0 + 1
    print("model: %s | horizon %d min | judged at +%d step(s) | test starts %s | %d steps | %d pairs"
          % (bundle["model_name"], bundle["horizon_minutes"], lead, test_start, n_steps, len(pairs)))

    # spiking links in the test period and the pairs whose static path uses them
    maxu = {lid: max(util[t][lid] for t in timeline[i0:]) for lid in topo.links}
    meanu = {lid: sum(util[t][lid] for t in timeline[i0:]) / float(len(timeline) - i0) for lid in topo.links}
    spiking = sorted([l for l, m in maxu.items() if m > 50], key=lambda l: -maxu[l])
    static_paths = all_paths(topo, None)
    affected_idx = np.array([n for n, pr in enumerate(pairs)
                             if any(l in spiking for l in topo.path_links(static_paths[pr]))], dtype=int)
    hot = max(meanu, key=meanu.get)
    print("spiking links (max > 50%%): %s" % (", ".join("%s (%.0f%%)" % (l, maxu[l]) for l in spiking) or "none"))
    print("affected pairs (static path crosses a spiking link): %d of %d | busiest link: %s" % (len(affected_idx), len(pairs), hot))
    if len(affected_idx) == 0:
        print("[WARN] no link exceeds 50%% in the test period; differences between policies will be small")

    ks = [args.k] if args.k is not None else [float(x) for x in args.k_list.split(",")]
    if args.cost_model == "queue":
        ks = [0.0]
    sweep, runs = {}, {}
    for k in ks:
        sys.stdout.write("simulating %s cost, k=%g ... " % (args.cost_model, k))
        sys.stdout.flush()
        pred_k = UtilizationPredictor(args.model, args.map)
        res, series, fc_log = simulate(topo, pred_k, timeline, util, i0, lead, args.cost_model, k, POLICIES, pairs)
        sweep[k] = summarize(res, pairs, affected_idx, POLICIES)
        runs[k] = (series, fc_log)
        print("done")

    subset = "affected" if len(affected_idx) else "all"

    def score(k):
        s = sweep[k][("forecast", subset)]
        return s["mean_latency_ms"] + 0.1 * s["path_changes_per_100_pair_steps"]
    best_k = min(ks, key=score)

    # ------------------------------------------------------------ report
    print()
    print("=" * 100)
    print("RESULTS for pairs: %s   (cost model: %s, k = %g)" % (subset, args.cost_model, best_k))
    print("=" * 100)
    print("%-10s %12s %12s %14s %12s %13s %16s %12s" % ("policy", "mean lat ms", "p95 lat ms", "mean max-util", "exposed>80%", "exposed>100%", "path changes/100", "rules/step"))
    for p in POLICIES:
        s = sweep[best_k][(p, subset)]
        print("%-10s %12.2f %12.2f %14.1f %11.1f%% %12.1f%% %16.2f %12.2f" % (
            p, s["mean_latency_ms"], s["p95_latency_ms"], s["mean_path_max_util"], s["pct_exposed_80"],
            s["pct_exposed_100"], s["path_changes_per_100_pair_steps"], s["rules_changed_per_step"]))
    stat = sweep[best_k][("static", subset)]["mean_latency_ms"]
    reac = sweep[best_k][("reactive", subset)]["mean_latency_ms"]
    fore = sweep[best_k][("forecast", subset)]["mean_latency_ms"]
    orac = sweep[best_k][("oracle", subset)]["mean_latency_ms"]
    print()
    print("mean latency: static %.2f | reactive %.2f | forecast %.2f | oracle %.2f ms" % (stat, reac, fore, orac))
    if stat > orac + 1e-9:
        print("forecast closes %.0f%% of the gap between static and the oracle; reactive closes %.0f%%"
              % (100 * (stat - fore) / (stat - orac), 100 * (stat - reac) / (stat - orac)))
    print("[%s] forecast policy %s the static baseline (%.2f vs %.2f ms)" % (
        "PASS" if fore < 0.99 * stat else "INFO", "beats" if fore < 0.99 * stat else "does not clearly beat (less than 1% better than)", fore, stat))
    print("[%s] forecast policy %s the reactive policy (%.2f vs %.2f ms)" % (
        "PASS" if fore < 0.99 * reac else "INFO", "beats" if fore < 0.99 * reac else "does not clearly beat (less than 1% better than)", fore, reac))

    if len(ks) > 1:
        print()
        print("K SWEEP (forecast policy, %s pairs):  k -> mean latency ms | path changes/100 | score" % subset)
        for k in ks:
            s = sweep[k][("forecast", subset)]
            print("   k=%-5g %8.2f %10.2f %10.2f %s" % (k, s["mean_latency_ms"], s["path_changes_per_100_pair_steps"], score(k), "  <- recommended" if k == best_k else ""))
        print("(score = mean latency + 0.1 x path changes per 100 pair-steps; lower is better)")

    # ------------------------------------------------------------ window recommendation
    hi = len(timeline) - 1 - lead
    pk = max(range(i0 + 3, hi), key=lambda i: util[timeline[i]][hot])
    start = max(i0 + F.WINDOW, pk - int(0.4 * args.window_steps))
    start = min(start, hi - args.window_steps)
    print()
    print("RECOMMENDED REPLAY WINDOW for the live demo (contains the biggest surge on %s, %.0f%% at %s):" % (hot, util[timeline[pk]][hot], timeline[pk]))
    print("   ML_START=%s  ML_STEPS=%d  ML_K=%g  ML_COST_MODEL=%s" % (timeline[start], args.window_steps, best_k, args.cost_model))

    # ------------------------------------------------------------ files
    csv_path = out_dir / "phase_d_offline_eval.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cost_model", "k", "policy", "pairs", "mean_latency_ms", "p95_latency_ms", "mean_path_max_util",
                    "pct_exposed_80", "pct_exposed_100", "path_changes_per_100_pair_steps", "rules_changed_per_step"])
        for k in ks:
            for p in POLICIES:
                for sub in ("all", "affected"):
                    if (p, sub) in sweep[k]:
                        s = sweep[k][(p, sub)]
                        w.writerow([args.cost_model, k, p, sub] + ["%.4f" % s[c] for c in (
                            "mean_latency_ms", "p95_latency_ms", "mean_path_max_util", "pct_exposed_80", "pct_exposed_100",
                            "path_changes_per_100_pair_steps", "rules_changed_per_step")])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = [sweep[best_k][(p, subset)]["mean_latency_ms"] for p in POLICIES]
    bars = ax.bar(POLICIES, vals, color=["#a6a6a6", "#8faadc", "#c00000", "#70ad47"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, "%.2f" % v, ha="center", va="bottom")
    ax.set_ylabel("mean modelled path latency (ms)")
    ax.set_title("Routing policies on %s pairs (k=%g, %s)" % (subset, best_k, args.cost_model))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_offline_policies.png", dpi=130)
    plt.close(fig)

    if len(ks) > 1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for p, c in (("reactive", "#8faadc"), ("forecast", "#c00000"), ("oracle", "#70ad47")):
            ax.plot(ks, [sweep[k][(p, subset)]["mean_latency_ms"] for k in ks], marker="o", color=c, label=p)
        ax.axhline(sweep[ks[0]][("static", subset)]["mean_latency_ms"], color="#7f7f7f", linestyle="--", label="static")
        ax.set_xlabel("cost parameter k")
        ax.set_ylabel("mean modelled path latency (ms)")
        ax.set_title("Effect of k (%s pairs)" % subset)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_offline_k_sweep.png", dpi=130)
        plt.close(fig)

    series0, fc0 = runs[best_k]
    if len(affected_idx):
        sp = max(affected_idx, key=lambda n: np.mean([row[n] for row in series0["static"]]) - np.mean([row[n] for row in series0["oracle"]]))
    else:
        sp = 0
    x = list(range(len(series0["static"])))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(x, [util[timeline[i0 + i]][hot] for i in x], color="black", label="actual utilization")
    axes[0].plot(x, [fc0[i][1][hot] for i in x], color="#c00000", label="forecast (%d min ahead)" % bundle["horizon_minutes"])
    axes[0].set_ylabel("%s utilization (%%)" % hot)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for p, c in (("static", "#7f7f7f"), ("reactive", "#8faadc"), ("forecast", "#c00000"), ("oracle", "#70ad47")):
        axes[1].plot(x, [row[sp] for row in series0[p]], color=c, label=p)
    axes[1].set_ylabel("latency %s -> %s (ms)" % pairs[sp])
    axes[1].set_xlabel("test-period step (5 min each)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_offline_timeline.png", dpi=130)
    plt.close(fig)

    print()
    print("Saved: %s" % csv_path)
    print("       %s, %s, %s" % (out_dir / "fig_offline_policies.png", out_dir / "fig_offline_k_sweep.png", out_dir / "fig_offline_timeline.png"))


if __name__ == "__main__":
    main()
