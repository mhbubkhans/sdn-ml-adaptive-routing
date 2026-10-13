#!/usr/bin/env python3
"""
phase_d_tests.py   (Phase D - Step D6; the same script is the experiment runner of Phase E)

WHAT IT DOES
  Runs ONE complete live experiment for the policy the controller was started with:

    1. starts the Abilene network in Mininet (controller mode) and the load emulator
    2. waits for the controller's replay to begin (live_state.json)
    3. at EVERY replay step: pings the watched host pairs, records the round-trip
       time, packet loss and the path the controller reports, and checks that the
       emulated link really has the delay the load model asks for
    4. when the replay ends: restores the links, prints a summary and saves
       results/phase_d_<policy>.csv

  Run it once per policy (static, reactive, forecast) with the SAME settings, then
  compare with compare_policies.py.

REQUIREMENTS
  Terminal 1:  ML_POLICY=forecast ML_START=... ML_STEPS=... ryu-manager ~/sdn_project/controller/ml_controller.py
  Terminal 2:  sudo python3 phase_d_tests.py
  Start Terminal 2 right after Terminal 1: the replay starts about 10 s after all switches connect.

USAGE
  sudo python3 phase_d_tests.py
  sudo python3 phase_d_tests.py --pairs h2-h6,h1-h10 --probe-count 4
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "topology"))
import load_model as LM         # noqa: E402
import load_emulator as LE      # noqa: E402
import abilene_topo as AT       # noqa: E402

results = []


def check(tag, text):
    results.append(tag)
    print("[%s] %s" % (tag, text))


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout


def ping_stats(src, dst, count):
    out = src.cmd("ping -c %d -i 0.2 -W 2 %s" % (count, dst.IP()))
    loss = re.search(r"(\d+)% packet loss", out)
    rtt = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/", out)
    return (int(loss.group(1)) if loss else 100), (float(rtt.group(2)) if rtt else None)


def pctl(values, q):
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser(description="Live experiment runner for one policy")
    ap.add_argument("--pairs", default="h2-h6,h1-h10,h12-h8,h3-h9")
    ap.add_argument("--probe-count", type=int, default=3, help="pings per pair per step")
    ap.add_argument("--hot-link", default="ATLAng_IPLSng", help="link whose emulated delay is verified")
    ap.add_argument("--wait", type=int, default=120, help="seconds to wait for the replay to start")
    ap.add_argument("--state", default=LE.DEFAULT_STATE)
    ap.add_argument("--map", default=LE.DEFAULT_MAP)
    ap.add_argument("--results-dir", default=str(HERE.parent / "results"))
    ap.add_argument("--no-emulator", action="store_true", help="do not apply the load (debugging)")
    args = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit("ERROR: needs root. Run:  sudo python3 phase_d_tests.py")

    pairs = [tuple(x.split("-")) for x in args.pairs.split(",") if x]
    topo_data = AT.load_map(args.map)
    hot_spec = next(l for l in topo_data["links"] if l["link_id"] == args.hot_link)
    hot_iface = "%s-eth%d" % (hot_spec["src_switch"], hot_spec["src_port"])
    if os.path.exists(args.state):
        os.remove(args.state)          # never trust a stale file from an earlier run

    from mininet.log import setLogLevel
    setLogLevel("warning")
    try:
        from mininet.clean import cleanup
        cleanup()
    except Exception:
        pass

    net, _ = AT.build_network(topo_data, "controller")
    emulator = None
    rows = []
    try:
        net.start()
        net.staticArp()

        print("\n=== T1: switches connected ===")
        t0 = time.time()
        while time.time() - t0 < 20 and not all(s.connected() for s in net.switches):
            time.sleep(1)
        up = sum(1 for s in net.switches if s.connected())
        if up != len(net.switches):
            check("FAIL", "%d of %d switches connected. Is the ML controller running?" % (up, len(net.switches)))
            raise SystemExit(1)
        check("PASS", "all %d switches connected" % up)

        if not args.no_emulator:
            emulator = LE.LoadEmulator(topo_data, args.state, log=lambda s: None)
            emulator.start()

        print("\n=== T2: waiting for the controller's replay to start (up to %d s) ===" % args.wait)
        t0, st = time.time(), None
        while time.time() - t0 < args.wait:
            st = LE.read_state(args.state)
            if LE.is_fresh(st):
                break
            time.sleep(1)
        if not LE.is_fresh(st):
            check("FAIL", "no live_state.json from the controller. Was it started with ml_controller.py (not baseline_controller.py)?")
            raise SystemExit(1)
        policy, total = st["policy"], st["steps"]
        check("PASS", "replay running: policy=%s, %d steps of %.0f s, cost=%s k=%g, forecast %d min ahead"
              % (policy, total, st["step_seconds"], st["cost_model"], st["k"], st["horizon_minutes"]))

        print("\n=== live run: probing %s at every step ===" % args.pairs)
        deadline = time.time() + total * (st["step_seconds"] + 5) + 60
        last = -1
        tc_ok = tc_n = 0
        st = None
        finished_in_time = False
        while time.time() < deadline:
            st = LE.read_state(args.state)
            if st is None:
                time.sleep(0.3)
                continue
            if st.get("finished"):
                finished_in_time = True
                break
            if st["step"] == last:
                time.sleep(0.3)
                continue
            last = st["step"]
            time.sleep(1.0)                                  # let the emulator and the flows settle
            for a, b in pairs:
                loss, rtt = ping_stats(net.get(a), net.get(b), args.probe_count)
                path = st["paths"].get("%s-%s" % (a, b), "?")
                rows.append({
                    "policy": policy, "step": st["step"], "timestamp": st["timestamp"], "pair": "%s-%s" % (a, b),
                    "path": path, "rtt_ms": "" if rtt is None else "%.2f" % rtt, "loss_pct": loss,
                    "hot_util": "%.1f" % st["util"].get(args.hot_link, 0.0),
                    "hot_forecast": "%.1f" % st["forecast"].get(args.hot_link, 0.0),
                    "rules_changed_step": st["rules_changed_step"],
                })
            exp_delay = LM.effective_delay_ms(hot_spec["delay_ms"], st["util"].get(args.hot_link, 0.0))
            got = AT.parse_delay_ms(run(["tc", "qdisc", "show", "dev", hot_iface]))
            tc_n += 1
            if got is not None and abs(got - exp_delay) <= max(0.5, 0.06 * exp_delay):
                tc_ok += 1
            print("  step %2d/%d %s | %s util %5.1f%% | %s" % (
                st["step"] + 1, total, st["timestamp"], args.hot_link, st["util"].get(args.hot_link, 0.0),
                "  ".join("%s %s ms" % (r["pair"], r["rtt_ms"] or "n/a") for r in rows[-len(pairs):])))
        if not finished_in_time:
            check("WARN", "the replay did not finish in time; results are partial")

        final = LE.read_state(args.state) or st or {}
        if emulator:
            emulator.stop(restore=True)
            emulator = None

        # ------------------------------------------------------------ evaluation
        print("\n=== T3: load emulation applied to the links ===")
        if args.no_emulator:
            print("[INFO] skipped (--no-emulator)")
        elif tc_n and tc_ok / float(tc_n) >= 0.8:
            check("PASS", "%s: emulated delay matched the load model in %d of %d steps" % (args.hot_link, tc_ok, tc_n))
        else:
            check("WARN", "%s: emulated delay matched in only %d of %d steps. Check with: sudo tc qdisc show dev %s" % (args.hot_link, tc_ok, tc_n, hot_iface))

        print("\n=== T4: traffic kept flowing ===")
        losses = [r["loss_pct"] for r in rows]
        mean_loss = sum(losses) / max(1, len(losses))
        check("PASS" if mean_loss <= 25 else "WARN", "average packet loss during the run: %.1f%% (limit 25%%)" % mean_loss)

        print("\n=== T5: routing behaviour ===")
        changes = final.get("path_changes_total", 0)
        if policy == "static":
            check("PASS" if changes == 0 else "WARN", "static policy: %d watched path changes (expected 0)" % changes)
        elif changes > 0:
            check("PASS", "%s policy rerouted watched pairs %d time(s), %d rule updates in total" % (policy, changes, final.get("rules_changed_total", 0)))
        else:
            check("WARN", "no watched path changed. Use a replay window with a surge (offline_eval.py prints one) or a larger ML_K")
        if final.get("live_mae") is not None:
            print("[INFO] live forecast error %.2f vs persistence %.2f (%%-points of utilization)" % (final["live_mae"], final["live_mae_persistence"]))

        # ------------------------------------------------------------ summary and CSV
        print("\n" + "-" * 78)
        print("SUMMARY for policy '%s'" % policy)
        print("%-8s %10s %10s %10s %8s %14s" % ("pair", "mean ms", "p95 ms", "max ms", "loss %", "path changes"))
        for a, b in pairs:
            pr = "%s-%s" % (a, b)
            rr = [float(r["rtt_ms"]) for r in rows if r["pair"] == pr and r["rtt_ms"] != ""]
            ls = [r["loss_pct"] for r in rows if r["pair"] == pr]
            seq = [r["path"] for r in rows if r["pair"] == pr]
            pc = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
            if rr:
                print("%-8s %10.1f %10.1f %10.1f %8.1f %14d" % (pr, sum(rr) / len(rr), pctl(rr, 95), max(rr), sum(ls) / len(ls), pc))
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.results_dir) / ("phase_d_%s.csv" % policy)
        if rows:
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print("\nSaved %d measurements to %s" % (len(rows), out_path))
        print("RESULT: %d PASS, %d WARN, %d FAIL" % (results.count("PASS"), results.count("WARN"), results.count("FAIL")))
    finally:
        if emulator:
            emulator.stop(restore=True)
        net.stop()
    sys.exit(1 if "FAIL" in results else 0)


if __name__ == "__main__":
    main()
