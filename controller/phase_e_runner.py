#!/usr/bin/env python3
"""
phase_e_runner.py   (Phase E - the final live experiment runner)

WHAT IT ADDS TO phase_d_tests.py
    - TCP THROUGHPUT: at every N-th step, iperf is run between chosen host pairs
    - FAILURE DURING THE REPLAY: at a chosen step a link is brought DOWN, the reroute
      outage is measured (ping every 50 ms), and at a later step the link is restored
    - SCENARIO LABELS: results are saved as results/phase_e_<scenario>_<policy>.csv
    - the path of each pair is computed locally from the weights in live_state.json
      and the links this script knows are down, so it is correct right after a failure

  Run it once per policy (static, reactive, forecast) for each scenario, with the
  SAME controller settings every time. analyze_results.py then compares everything.

REQUIREMENTS
  Terminal 1:  ML_POLICY=<policy> ML_K=<k> ML_START=<ts> ML_STEPS=<n> ML_STEP_SECONDS=<s> \
                   ryu-manager ~/sdn_project/controller/ml_controller.py
  Terminal 2:  sudo python3 phase_e_runner.py --scenario <name> [options]
  (start Terminal 2 right after Terminal 1; restart the controller for every run)

SCENARIOS USED IN THE GUIDE
  throughput   ML_STEP_SECONDS=14 ML_STEPS=30    --iperf
  failure      ML_STEP_SECONDS=10 ML_STEPS=40    --fail-link ATLAng_HSTNng --fail-step 10 --restore-step 25
  window2      ML_STEP_SECONDS=8  ML_STEPS=40    (another replay window, pings only)
"""
import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "topology"))
import routing_core as RC       # noqa: E402
import load_model as LM         # noqa: E402
import load_emulator as LE      # noqa: E402
import abilene_topo as AT       # noqa: E402
import phase_d_tests as PD      # noqa: E402  (ping_stats, run, pctl)
import phase_c_tests as PC      # noqa: E402  (measure_outage)

results = []


def check(tag, text):
    results.append(tag)
    print("[%s] %s" % (tag, text))


def parse_rate_mbps(text):
    m = re.search(r"([\d.]+)\s*([KMG]?)bits/sec", str(text))
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2)
    return v / 1000.0 if u == "K" else (v * 1000.0 if u == "G" else v)


def main():
    ap = argparse.ArgumentParser(description="Final live experiment runner")
    ap.add_argument("--scenario", default="normal", help="label used in the result file names")
    ap.add_argument("--pairs", default="h2-h6,h1-h10,h12-h8,h3-h9")
    ap.add_argument("--control", default="h3-h9", help="control pair (its path never crosses a surging link)")
    ap.add_argument("--probe-count", type=int, default=3)
    ap.add_argument("--iperf", action="store_true", help="measure TCP throughput")
    ap.add_argument("--iperf-pairs", default="h2-h6,h12-h8")
    ap.add_argument("--iperf-every", type=int, default=2, help="run iperf at every N-th step")
    ap.add_argument("--iperf-seconds", type=int, default=3)
    ap.add_argument("--fail-link", default=None, help="link_id to bring down during the replay")
    ap.add_argument("--fail-step", type=int, default=10)
    ap.add_argument("--restore-step", type=int, default=25)
    ap.add_argument("--fail-pair", default="h12-h8", help="pair used to measure the reroute outage")
    ap.add_argument("--hot-link", default="ATLAng_IPLSng")
    ap.add_argument("--wait", type=int, default=120)
    ap.add_argument("--state", default=LE.DEFAULT_STATE)
    ap.add_argument("--map", default=LE.DEFAULT_MAP)
    ap.add_argument("--results-dir", default=str(HERE.parent / "results"))
    args = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit("ERROR: needs root. Run:  sudo python3 phase_e_runner.py ...")

    pairs = [tuple(x.split("-")) for x in args.pairs.split(",") if x]
    iperf_pairs = [tuple(x.split("-")) for x in args.iperf_pairs.split(",") if x] if args.iperf else []
    topo_data = AT.load_map(args.map)
    core = RC.Topology(topo_data, "delay")
    hot_spec = core.links[args.hot_link]
    hot_iface = "%s-eth%d" % (hot_spec["src_switch"], hot_spec["src_port"])
    fail_spec = None
    if args.fail_link:
        if args.fail_link not in core.links:
            sys.exit("ERROR: unknown link %s. Valid: %s" % (args.fail_link, ", ".join(sorted(core.links))))
        if args.fail_link == "ATLAM5_ATLAng":
            sys.exit("ERROR: ATLAM5_ATLAng is the only link of s1; failing it would cut h1 off completely.")
        fail_spec = core.links[args.fail_link]
        fa, fb = args.fail_pair.split("-")
    if os.path.exists(args.state):
        os.remove(args.state)

    from mininet.log import setLogLevel
    setLogLevel("warning")
    try:
        from mininet.clean import cleanup
        cleanup()
    except Exception:
        pass

    net, _ = AT.build_network(topo_data, "controller")
    emulator = None
    rows, events = [], []
    down = set()
    failed = restored = False
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

        emulator = LE.LoadEmulator(topo_data, args.state, log=lambda s: None)
        emulator.start()

        print("\n=== T2: waiting for the controller's replay (up to %d s) ===" % args.wait)
        t0, st = time.time(), None
        while time.time() - t0 < args.wait:
            st = LE.read_state(args.state)
            if LE.is_fresh(st):
                break
            time.sleep(1)
        if not LE.is_fresh(st):
            check("FAIL", "no live_state.json. Start the controller with ml_controller.py (a NEW one for every run)")
            raise SystemExit(1)
        policy, total = st["policy"], st["steps"]
        check("PASS", "scenario '%s': policy=%s, %d steps of %.0f s, k=%g, forecast %d min ahead"
              % (args.scenario, policy, total, st["step_seconds"], st["k"], st["horizon_minutes"]))
        if st["step_seconds"] < 8 + (4 if iperf_pairs else 0):
            print("[INFO] step length %.0f s is short for this scenario: some steps may be skipped (use ML_STEP_SECONDS=%d)"
                  % (st["step_seconds"], 14 if iperf_pairs else 10))

        print("\n=== live run ===")
        deadline = time.time() + total * (st["step_seconds"] + 8) + 90
        last, tc_ok, tc_n, finished = -1, 0, 0, False
        while time.time() < deadline:
            st = LE.read_state(args.state)
            if st is None:
                time.sleep(0.3)
                continue
            if st.get("finished"):
                finished = True
                break
            if st["step"] == last:
                time.sleep(0.3)
                continue
            last = st["step"]
            time.sleep(1.0)
            weights = st.get("weights")
            tput = {}
            if iperf_pairs and last % args.iperf_every == 0:
                for a, b in iperf_pairs:
                    try:
                        res = net.iperf([net.get(a), net.get(b)], seconds=args.iperf_seconds)
                        tput["%s-%s" % (a, b)] = parse_rate_mbps(res[-1])
                    except Exception as exc:
                        print("[WARN] iperf %s-%s failed: %s" % (a, b, exc))
            step_rows = []
            for a, b in pairs:
                loss, rtt = PD.ping_stats(net.get(a), net.get(b), args.probe_count)
                pth = core.path(a, b, down, weights)
                tp = tput.get("%s-%s" % (a, b))
                step_rows.append({
                    "scenario": args.scenario, "policy": policy, "step": st["step"], "timestamp": st["timestamp"],
                    "pair": "%s-%s" % (a, b), "path": core.describe(pth) if pth else "UNREACHABLE",
                    "rtt_ms": "" if rtt is None else "%.2f" % rtt, "loss_pct": loss,
                    "tput_mbps": "" if tp is None else "%.2f" % tp,
                    "hot_util": "%.1f" % st["util"].get(args.hot_link, 0.0),
                    "hot_forecast": "%.1f" % st["forecast"].get(args.hot_link, 0.0),
                    "rules_changed_step": st["rules_changed_step"], "links_down": "+".join(sorted(down)),
                })
            rows.extend(step_rows)
            exp_delay = LM.effective_delay_ms(hot_spec["delay_ms"], st["util"].get(args.hot_link, 0.0))
            got = AT.parse_delay_ms(PD.run(["tc", "qdisc", "show", "dev", hot_iface]))
            tc_n += 1
            if got is not None and abs(got - exp_delay) <= max(0.5, 0.06 * exp_delay):
                tc_ok += 1
            print("  step %2d/%d %s | %s util %5.1f%% | %s%s" % (
                st["step"] + 1, total, st["timestamp"], args.hot_link, st["util"].get(args.hot_link, 0.0),
                "  ".join("%s %s ms" % (r["pair"], r["rtt_ms"] or "n/a") for r in step_rows),
                ("  | iperf " + " ".join("%s %.1f Mbps" % (k, v) for k, v in tput.items() if v is not None)) if tput else ""))

            # ---- failure injection and recovery (after the probes of the step)
            if fail_spec and not failed and last >= args.fail_step:
                src, dst = net.get(fa), net.get(fb)
                sa, sb = fail_spec["src_switch"], fail_spec["dst_switch"]
                lost, outage, rec = PC.measure_outage(src, dst, lambda: net.configLinkStatus(sa, sb, "down"),
                                                      count=80, interval=0.05, event_after=0.8)
                down.add(args.fail_link)
                failed = True
                events.append({"scenario": args.scenario, "policy": policy, "event": "link_down", "link": args.fail_link,
                               "step": last, "timestamp": st["timestamp"], "pair": args.fail_pair,
                               "lost_pings": lost, "outage_ms": "%.0f" % outage, "recovered": rec})
                print("  >>> LINK DOWN %s: reroute outage about %.0f ms (%d pings lost), traffic %s" % (
                    args.fail_link, outage, lost, "recovered" if rec else "NOT RECOVERED"))
            elif fail_spec and failed and not restored and last >= args.restore_step:
                src, dst = net.get(fa), net.get(fb)
                sa, sb = fail_spec["src_switch"], fail_spec["dst_switch"]
                lost, outage, rec = PC.measure_outage(src, dst, lambda: net.configLinkStatus(sa, sb, "up"),
                                                      count=80, interval=0.05, event_after=0.8)
                down.discard(args.fail_link)
                restored = True
                events.append({"scenario": args.scenario, "policy": policy, "event": "link_up", "link": args.fail_link,
                               "step": last, "timestamp": st["timestamp"], "pair": args.fail_pair,
                               "lost_pings": lost, "outage_ms": "%.0f" % outage, "recovered": rec})
                print("  >>> LINK UP %s: outage about %.0f ms (%d pings lost)" % (args.fail_link, outage, lost))
        if not finished:
            check("WARN", "the replay did not finish in time; results are partial")

        final = LE.read_state(args.state) or st or {}
        emulator.stop(restore=True)
        emulator = None
        if fail_spec and failed and not restored:
            net.configLinkStatus(fail_spec["src_switch"], fail_spec["dst_switch"], "up")

        # ------------------------------------------------------------ checks
        print("\n=== T3: load emulation ===")
        if tc_n and tc_ok / float(tc_n) >= 0.8:
            check("PASS", "%s: emulated delay matched the load model in %d of %d steps" % (args.hot_link, tc_ok, tc_n))
        else:
            check("WARN", "%s: emulated delay matched in only %d of %d steps" % (args.hot_link, tc_ok, tc_n))

        print("\n=== T4: traffic kept flowing ===")
        ok_rows = [r for r in rows if r["links_down"] == ""] or rows
        mean_loss = sum(r["loss_pct"] for r in ok_rows) / max(1, len(ok_rows))
        check("PASS" if mean_loss <= 10 else "FAIL", "average packet loss outside failure periods: %.1f%% (limit 10%%)" % mean_loss)

        print("\n=== T5: routing behaviour ===")
        if fail_spec:
            before = [r["path"] for r in rows if r["pair"] == args.fail_pair and r["links_down"] == ""]
            during = [r["path"] for r in rows if r["pair"] == args.fail_pair and r["links_down"] != ""]
            if not failed:
                check("WARN", "the failure step %d was never reached" % args.fail_step)
            elif during and before and during[-1] != before[0]:
                check("PASS", "%s moved to a detour while %s was down: %s" % (args.fail_pair, args.fail_link, during[-1]))
            else:
                check("WARN", "%s kept its path while the link was down (the link was probably not on its path)" % args.fail_pair)
            for ev in events:
                check("PASS" if ev["recovered"] else "FAIL",
                      "%s %s: outage about %s ms, %s" % (ev["event"], ev["link"], ev["outage_ms"], "recovered" if ev["recovered"] else "NOT recovered"))
        elif policy == "static":
            check("PASS" if final.get("path_changes_total", 0) == 0 else "WARN",
                  "static policy: %d watched path changes (expected 0)" % final.get("path_changes_total", 0))
        else:
            check("PASS" if final.get("path_changes_total", 0) > 0 else "WARN",
                  "%s policy: %d watched path changes, %d rule updates" % (policy, final.get("path_changes_total", 0), final.get("rules_changed_total", 0)))
        if final.get("live_mae") is not None:
            print("[INFO] live forecast error %.2f vs persistence %.2f (%%-points of utilization)" % (final["live_mae"], final["live_mae_persistence"]))

        # ------------------------------------------------------------ summary and files
        print("\n" + "-" * 86)
        print("SUMMARY scenario '%s', policy '%s'" % (args.scenario, policy))
        print("%-8s %9s %9s %9s %8s %12s %13s" % ("pair", "mean ms", "p95 ms", "max ms", "loss %", "mean Mbps", "path changes"))
        for a, b in pairs:
            pr = "%s-%s" % (a, b)
            sel = [r for r in rows if r["pair"] == pr]
            rr = [float(r["rtt_ms"]) for r in sel if r["rtt_ms"] != ""]
            tt = [float(r["tput_mbps"]) for r in sel if r["tput_mbps"] != ""]
            seq = [r["path"] for r in sel]
            pc = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
            if rr:
                print("%-8s %9.1f %9.1f %9.1f %8.1f %12s %13d" % (
                    pr, sum(rr) / len(rr), PD.pctl(rr, 95), max(rr), sum(r["loss_pct"] for r in sel) / len(sel),
                    "%.1f" % (sum(tt) / len(tt)) if tt else "-", pc))
        rd = Path(args.results_dir)
        rd.mkdir(parents=True, exist_ok=True)
        base = rd / ("phase_e_%s_%s" % (args.scenario, policy))
        if rows:
            with open(str(base) + ".csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print("\nSaved %d measurements to %s.csv" % (len(rows), base))
        if events:
            with open(str(base) + "_events.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
                w.writeheader()
                w.writerows(events)
            print("Saved %d event(s) to %s_events.csv" % (len(events), base))
        print("RESULT: %d PASS, %d WARN, %d FAIL" % (results.count("PASS"), results.count("WARN"), results.count("FAIL")))
    finally:
        if emulator:
            emulator.stop(restore=True)
        net.stop()
    sys.exit(1 if "FAIL" in results else 0)


if __name__ == "__main__":
    main()
