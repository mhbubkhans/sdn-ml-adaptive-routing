#!/usr/bin/env python3
"""
show_paths.py   (Phase C - Steps C3, C5, C6)

WHAT IT DOES
  Shows what the routing logic WILL do, so you can compare it with what the
  running network really does. Works offline: no sudo, no Mininet, no Ryu.

USAGE   (from ~/sdn_project/controller)
  python3 show_paths.py                              # the six reference host pairs
  python3 show_paths.py --pair h5 h8                 # one pair
  python3 show_paths.py --pair h5 h8 --down HSTNng_LOSAng    # with a failed link
  python3 show_paths.py --switch s5                  # the flow table of switch s5
  python3 show_paths.py --all --csv ../results/expected_paths.csv   # all 132 pairs to a file
  python3 show_paths.py --metric hops                # use hop count instead of delay
"""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import routing_core as RC  # noqa: E402

REFERENCE_PAIRS = [("h5", "h8"), ("h12", "h8"), ("h9", "h8"), ("h3", "h10"), ("h1", "h10"), ("h11", "h9")]


def row_for(topo, a, b, down):
    p = topo.path(a, b, down_links=down)
    if p is None:
        return [a, b, "UNREACHABLE", "-", "-", "-"]
    return [a, b, topo.describe(p), str(len(p) - 1), "%.2f" % topo.path_delay_ms(p),
            "%.1f" % topo.expected_rtt_ms(p)]


def main():
    ap = argparse.ArgumentParser(description="Show expected paths and flow tables")
    ap.add_argument("--map", default=None)
    ap.add_argument("--metric", choices=["delay", "hops"], default="delay")
    ap.add_argument("--pair", nargs=2, metavar=("SRC", "DST"))
    ap.add_argument("--down", nargs="*", default=[], help="link_ids that are down")
    ap.add_argument("--switch", default=None, help="show the flow table of this switch, e.g. s5")
    ap.add_argument("--all", action="store_true", help="all 132 ordered pairs")
    ap.add_argument("--csv", default=None, help="write the table to this CSV file")
    args = ap.parse_args()

    topo = RC.load_topology(args.map, args.metric)
    for l in args.down:
        if l not in topo.links:
            sys.exit("ERROR: unknown link_id %s. Valid ones: %s" % (l, ", ".join(sorted(topo.links))))

    print("metric: %s | links down: %s" % (args.metric, ", ".join(args.down) or "none"))

    if args.switch:
        if args.switch not in topo.dpid_of:
            sys.exit("ERROR: unknown switch %s (use s1 ... s12)" % args.switch)
        dpid = topo.dpid_of[args.switch]
        flows = topo.compute_flows(args.down)
        print("\nFlow table of %s (%s), dpid %d: match eth_dst -> output port" % (args.switch, topo.node_of_switch[args.switch], dpid))
        print("%-6s %-18s %-8s %-12s %s" % ("host", "eth_dst", "port", "next hop", "meaning"))
        for hname in sorted(topo.hosts, key=lambda x: int(x[1:])):
            h = topo.hosts[hname]
            key = (dpid, h["mac"])
            if key not in flows:
                print("%-6s %-18s %-8s %-12s %s" % (hname, h["mac"], "-", "-", "unreachable (no entry)"))
                continue
            port = flows[key]
            if h["switch"] == args.switch:
                nxt, meaning = "(local)", "deliver to host %s" % hname
            else:
                nxt = next(topo.name_of[n] for (d, n), p in topo.neighbor_port.items() if d == dpid and p == port)
                meaning = "toward %s via %s" % (h["switch"], nxt)
            print("%-6s %-18s %-8s %-12s %s" % (hname, h["mac"], port, nxt, meaning))
        return

    if args.pair:
        pairs = [tuple(args.pair)]
    elif args.all:
        pairs = [(a, b) for a in sorted(topo.hosts, key=lambda x: int(x[1:]))
                 for b in sorted(topo.hosts, key=lambda x: int(x[1:])) if a != b]
    else:
        pairs = REFERENCE_PAIRS
    for a, b in pairs:
        for n in (a, b):
            if n not in topo.hosts:
                sys.exit("ERROR: unknown host %s (use h1 ... h12)" % n)

    header = ["from", "to", "path", "hops", "one_way_ms", "expected_ping_rtt_ms"]
    rows = [row_for(topo, a, b, args.down) for a, b in pairs]
    if not args.all or args.csv is None:
        print()
        print("%-5s %-5s %-62s %-5s %-11s %s" % tuple(header))
        for r in rows[:40] if args.all else rows:
            print("%-5s %-5s %-62s %-5s %-11s %s" % tuple(r))
        if args.all and len(rows) > 40:
            print("... (%d more rows; use --csv to save them all)" % (len(rows) - 40))
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print("\nSaved %d rows to %s" % (len(rows), args.csv))


if __name__ == "__main__":
    main()
