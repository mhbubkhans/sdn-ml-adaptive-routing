#!/usr/bin/env python3
"""
phase_c_tests.py   (Phase C - Step C6)

WHAT IT DOES
  Starts the Abilene network in Mininet (controller mode) and tests the running
  baseline controller automatically:

    T1  all 12 switches are connected to the controller
    T2  every switch has its 12 forwarding entries
    T3  pingall: every host reaches every host
    T4  six reference host pairs: ping time matches the expected delay AND the
        packets really followed the expected path (checked with flow counters)
    T5  link failures: the link is brought down, traffic must reroute to the
        expected detour; REROUTE TIME is measured; then the link is restored

  It saves every measurement to results/phase_c_baseline.csv. These numbers are
  the BASELINE that your ML controller is compared against in Phase E.

REQUIREMENTS
  Terminal 1:  ryu-manager ~/sdn_project/controller/baseline_controller.py
  Terminal 2:  sudo python3 phase_c_tests.py          (this script)

USAGE
  sudo python3 phase_c_tests.py                  full test (about 3-4 minutes)
  sudo python3 phase_c_tests.py --quick          T1-T3, three pairs, one failure (about 1.5 minutes)
  sudo python3 phase_c_tests.py --metric hops    only if you started the controller with BASELINE_METRIC=hops
  sudo python3 phase_c_tests.py --keep-open      open the mininet> prompt at the end
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
import routing_core as RC   # noqa: E402
import abilene_topo as AT   # noqa: E402

PAIRS = [("h5", "h8"), ("h12", "h8"), ("h9", "h8"), ("h3", "h10"), ("h1", "h10"), ("h11", "h9")]
FAILURES = [  # (link_id, source host, destination host)
    ("HSTNng_LOSAng", "h5", "h8"),
    ("ATLAng_WASHng", "h9", "h8"),
    ("IPLSng_KSCYng", "h3", "h10"),
]

results, rows = [], []


def check(ok, text):
    results.append(ok)
    print("[%s] %s" % ("PASS" if ok else "FAIL", text))
    return ok


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True).stdout


# ---------------------------------------------------------------------------
# measuring helpers
# ---------------------------------------------------------------------------
def dump_flows(sw):
    return run(["ovs-ofctl", "-O", "OpenFlow13", "dump-flows", sw])


def parse_flows(text):
    """-> {dst_mac: n_packets} for our eth_dst forwarding entries."""
    out = {}
    for line in text.splitlines():
        m = re.search(r"n_packets=(\d+).*?(?:dl_dst|eth_dst)=([0-9a-fA-F:]{17})", line)
        if m:
            out[m.group(2).lower()] = int(m.group(1))
    return out


def ping_stats(src, dst, count=5):
    out = src.cmd("ping -c %d -i 0.2 -W 2 %s" % (count, dst.IP()))
    loss = re.search(r"(\d+)% packet loss", out)
    rtt = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/", out)
    return (int(loss.group(1)) if loss else 100), (float(rtt.group(2)) if rtt else None)


def path_by_counters(net, topo, src, dst, count=5):
    """Send pings and see which switches' flow for the destination MAC counted packets."""
    dst_mac = topo.hosts[dst.name]["mac"].lower()
    names = sorted(topo.dpid_of, key=lambda s: int(s[1:]))
    before = {s: parse_flows(dump_flows(s)).get(dst_mac, 0) for s in names}
    src.cmd("ping -c %d -i 0.1 -W 2 %s" % (count, dst.IP()))
    after = {s: parse_flows(dump_flows(s)).get(dst_mac, 0) for s in names}
    return set(s for s in names if after[s] - before[s] > 0)


def measure_outage(src, dst, event, count=120, interval=0.05, event_after=1.0):
    """Ping every 50 ms, trigger `event` after 1 s, count the lost replies.
    Returns (lost_packets, outage_ms, recovered)."""
    p = src.popen(["ping", "-D", "-i", str(interval), "-c", str(count), dst.IP()],
                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(event_after)
    event()
    out = p.communicate()[0].decode("utf-8", "ignore")
    seqs = sorted(set(int(m.group(1)) for m in re.finditer(r"icmp_seq=(\d+)", out)))
    if not seqs:
        return count, count * interval * 1000.0, False
    lost = (seqs[-1] - seqs[0] + 1) - len(seqs)
    recovered = (count - seqs[-1]) <= 5
    return lost, lost * interval * 1000.0, recovered


def rtt_ok(measured, expected):
    return measured is not None and abs(measured - expected) <= max(4.0, 0.15 * expected)


def record(test, pair, link, state, path, rtt, expected_rtt, outage_ms=""):
    rows.append({"test": test, "pair": pair, "link": link, "state": state, "path": path,
                 "measured_rtt_ms": "" if rtt is None else "%.2f" % rtt,
                 "expected_rtt_ms": "%.2f" % expected_rtt, "reroute_outage_ms": outage_ms})


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Automated tests for the baseline controller")
    ap.add_argument("--metric", choices=["delay", "hops"], default="delay")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep-open", action="store_true")
    ap.add_argument("--map", default=str(HERE.parent / "topology" / "topology_map.json"))
    ap.add_argument("--results", default=str(HERE.parent / "results" / "phase_c_baseline.csv"))
    args = ap.parse_args()

    if os.geteuid() != 0:
        sys.exit("ERROR: needs root. Run:  sudo python3 phase_c_tests.py")

    topo_data = AT.load_map(args.map)
    topo = RC.Topology(topo_data, args.metric)

    from mininet.log import setLogLevel
    setLogLevel("warning")
    try:
        from mininet.clean import cleanup
        cleanup()
    except Exception:
        pass

    net, _links = AT.build_network(topo_data, "controller")
    n_sw = len(topo.dpid_of)
    try:
        net.start()
        net.staticArp()

        # ---------------- T1: connected
        print("\n=== T1: switches connected to the controller ===")
        t0 = time.time()
        while time.time() - t0 < 20 and not all(s.connected() for s in net.switches):
            time.sleep(1)
        up = [s.name for s in net.switches if s.connected()]
        if not check(len(up) == n_sw, "%d of %d switches connected" % (len(up), n_sw)):
            print("      Is the controller running?  ryu-manager ~/sdn_project/controller/baseline_controller.py")
            raise SystemExit(1)

        # ---------------- T2: flows installed
        print("\n=== T2: forwarding entries installed ===")
        t0 = time.time()
        counts = {}
        while time.time() - t0 < 20:
            counts = {s: len(parse_flows(dump_flows(s))) for s in topo.dpid_of}
            if all(c >= len(topo.hosts) for c in counts.values()):
                break
            time.sleep(1)
        bad = {s: c for s, c in counts.items() if c != len(topo.hosts)}
        check(not bad, "every switch has %d eth_dst entries%s" % (len(topo.hosts), "" if not bad else "  (wrong: %s)" % bad))

        # ---------------- T3: pingall
        print("\n=== T3: pingall ===")
        loss = net.pingAll()
        check(loss == 0, "pingall: %s%% dropped" % loss)

        # ---------------- T4: reference pairs
        print("\n=== T4: reference pairs (ping time and real path) ===")
        pairs = PAIRS[:3] if args.quick else PAIRS
        for a, b in pairs:
            src, dst = net.get(a), net.get(b)
            exp_path = topo.path(a, b)
            exp_rtt = topo.expected_rtt_ms(exp_path)
            loss, rtt = ping_stats(src, dst)
            seen = path_by_counters(net, topo, src, dst)
            ok_path = seen == set(exp_path)
            check(loss == 0 and rtt_ok(rtt, exp_rtt) and ok_path,
                  "%s -> %s | rtt %s ms (expected about %.1f) | path %s %s"
                  % (a, b, "%.1f" % rtt if rtt else "n/a", exp_rtt, topo.describe(exp_path),
                     "OK" if ok_path else "WRONG, packets crossed: %s" % sorted(seen)))
            record("reference", "%s-%s" % (a, b), "", "normal", topo.describe(exp_path), rtt, exp_rtt)

        # ---------------- T5: failures
        print("\n=== T5: link failure, reroute and recovery ===")
        scenarios = FAILURES[:1] if args.quick else FAILURES
        for link_id, a, b in scenarios:
            spec = topo.links[link_id]
            sa, sb = spec["src_switch"], spec["dst_switch"]
            src, dst = net.get(a), net.get(b)
            normal = topo.path(a, b)
            detour = topo.path(a, b, down_links=[link_id])
            exp_rtt_n, exp_rtt_d = topo.expected_rtt_ms(normal), topo.expected_rtt_ms(detour)
            print("\n  -- %s (%s - %s) fails, traffic %s -> %s" % (link_id, sa, sb, a, b))

            lost, outage, recovered = measure_outage(src, dst, lambda: net.configLinkStatus(sa, sb, "down"))
            time.sleep(1.0)
            loss, rtt = ping_stats(src, dst)
            seen = path_by_counters(net, topo, src, dst)
            ok = loss == 0 and recovered and seen == set(detour) and rtt_ok(rtt, exp_rtt_d)
            check(ok, "after failure: path %s | rtt %s ms (expected about %.1f) | reroute outage about %.0f ms (%d pings lost)"
                  % ("OK " + topo.describe(detour) if seen == set(detour) else "WRONG %s" % sorted(seen),
                     "%.1f" % rtt if rtt else "n/a", exp_rtt_d, outage, lost))
            record("failure", "%s-%s" % (a, b), link_id, "link_down", topo.describe(detour), rtt, exp_rtt_d, "%.0f" % outage)

            lost2, outage2, _ = measure_outage(src, dst, lambda: net.configLinkStatus(sa, sb, "up"))
            time.sleep(1.0)
            loss, rtt = ping_stats(src, dst)
            seen = path_by_counters(net, topo, src, dst)
            ok = loss == 0 and seen == set(normal) and rtt_ok(rtt, exp_rtt_n)
            check(ok, "after recovery: path %s | rtt %s ms (expected about %.1f) | outage about %.0f ms"
                  % ("OK " + topo.describe(normal) if seen == set(normal) else "WRONG %s" % sorted(seen),
                     "%.1f" % rtt if rtt else "n/a", exp_rtt_n, outage2))
            record("failure", "%s-%s" % (a, b), link_id, "link_restored", topo.describe(normal), rtt, exp_rtt_n, "%.0f" % outage2)

        # ---------------- save and summarise
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n" + "-" * 66)
        print("RESULT: %d PASS, %d FAIL" % (sum(results), len(results) - sum(results)))
        print("Measurements saved to: %s" % args.results)
        print("(the reroute outage is the BASELINE value to beat or match in Phase E)")
        if args.keep_open:
            from mininet.cli import CLI
            CLI(net)
    finally:
        net.stop()
    sys.exit(0 if results and all(results) else 1)


if __name__ == "__main__":
    main()
