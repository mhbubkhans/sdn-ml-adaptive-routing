#!/usr/bin/env python3
"""
check_routing_core.py   (Phase C - Step C3)

WHAT IT DOES
  Tests the routing logic of routing_core.py on your real topology, WITHOUT
  Mininet, Ryu or sudo. If this passes, any later problem is in the network or
  controller wiring, not in the maths.

CHECKS
  1. the network has 12 switches, 12 hosts, 15 links
  2. a full forwarding table has 12 x 12 = 144 entries
  3. known reference paths (the table in the Phase A guide) come out right
  4. following the next hops from any host to any host always arrives (no loops)
  5. the detour paths for three link failures are right
  6. after ANY single link failure the network stays connected, except for
     ATLAM5_ATLAng, which isolates h1 (it is s1's only link)

USAGE   (from ~/sdn_project/controller)
  python3 check_routing_core.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import routing_core as RC  # noqa: E402

results = []


def check(ok, text):
    results.append(ok)
    print("[%s] %s" % ("PASS" if ok else "FAIL", text))


def main():
    topo = RC.load_topology(sys.argv[1] if len(sys.argv) > 1 else None, "delay")

    check(len(topo.dpid_of) == 12 and len(topo.hosts) == 12 and len(topo.links) == 15,
          "topology has 12 switches, 12 hosts, 15 links")

    flows = topo.compute_flows()
    check(len(flows) == 144, "full forwarding table has %d entries (expected 144)" % len(flows))

    # reference answers (delay metric), same as the Phase A guide
    refs = [
        ("h5", "h8", ["s5", "s8"], 10.97),
        ("h12", "h8", ["s12", "s2", "s5", "s8"], 20.87),
        ("h9", "h8", ["s9", "s12", "s2", "s5", "s8"], 22.55),
        ("h3", "h10", ["s3", "s6", "s7", "s4", "s10"], 17.10),
        ("h1", "h10", ["s1", "s2", "s6", "s7", "s4", "s10"], 19.42),
        ("h11", "h9", ["s11", "s4", "s7", "s6", "s3", "s9"], 23.12),
    ]
    for a, b, exp_path, exp_delay in refs:
        p = topo.path(a, b)
        d = topo.path_delay_ms(p) if p else -1
        check(p == exp_path and abs(d - exp_delay) < 0.02,
              "%s -> %s : %s  (%.2f ms one-way)" % (a, b, topo.describe(p) if p else None, d))

    # no loops: every host reaches every host
    bad = 0
    for a in topo.hosts:
        for b in topo.hosts:
            if topo.path(a, b) is None:
                bad += 1
    check(bad == 0, "all 144 host pairs reachable with no loops")

    # detours
    det = [
        ("HSTNng_LOSAng", "h5", "h8", ["s5", "s7", "s4", "s10", "s8"], 18.96),
        ("ATLAng_WASHng", "h9", "h8", ["s9", "s3", "s6", "s7", "s4", "s10", "s8"], 25.35),
        ("IPLSng_KSCYng", "h3", "h10", ["s3", "s6", "s2", "s5", "s8", "s10"], 23.14),
    ]
    for link, a, b, exp_path, exp_delay in det:
        p = topo.path(a, b, down_links=[link])
        d = topo.path_delay_ms(p) if p else -1
        check(p == exp_path and abs(d - exp_delay) < 0.02,
              "%s down: %s -> %s via %s (%.2f ms)" % (link, a, b, topo.describe(p) if p else None, d))

    # single failures
    all_ok = True
    for link in topo.links:
        f = topo.compute_flows(down_links=[link])
        expected = 144 if link != "ATLAM5_ATLAng" else 144 - 22
        if len(f) != expected:
            all_ok = False
            print("      link %s down: %d entries (expected %d)" % (link, len(f), expected))
    check(all_ok, "every single link failure keeps the network connected (ATLAM5_ATLAng isolates h1 only)")

    print("-" * 60)
    print("RESULT: %d PASS, %d FAIL" % (sum(results), len(results) - sum(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
