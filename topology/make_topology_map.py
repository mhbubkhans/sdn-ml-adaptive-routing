#!/usr/bin/env python3
"""
make_topology_map.py   (Phase A - Step A3)

WHAT IT DOES
  Reads the REAL Abilene topology (nodes.csv, links.csv from SNDlib) and writes
  topology_map.json. That JSON file is the single source of truth used by:
    - abilene_topo.py      (Phase A: builds the Mininet network)
    - the Ryu controllers  (Phase C/D: knows hosts, ports, link delays)
    - the ML / test scripts (Phase B/E: link_id names match link_timeseries.csv)

  It does NOT need Mininet, pandas or sudo. Plain Python 3 only.

NAMING RULES (fixed for the whole project)
  Node number i (1..12, in the order of nodes.csv):
      switch  s<i>      dpid = i
      host    h<i>      IP  = 10.0.0.<i>/24     MAC = 00:00:00:00:00:<i in hex>
  Port numbers on every switch:
      port 1      -> its own host
      port 2,3,.. -> inter-switch links, numbered in the order they appear in links.csv

SCALING (Mininet cannot emulate 9.92 Gbps on a laptop)
  emulated_bandwidth = real_bandwidth * bw_scale        (default 0.01  -> 99.2 / 24.8 Mbps)
  emulated_delay     = real_propagation_delay * delay_scale   (default 1.0)

USAGE
  python3 make_topology_map.py
  python3 make_topology_map.py --bw-scale 0.01 --delay-scale 1.0
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Build topology_map.json from Abilene CSV files")
    ap.add_argument("--data-dir", default=str(here.parent / "data"),
                    help="folder containing nodes.csv and links.csv (default: ../data)")
    ap.add_argument("--out", default=str(here / "topology_map.json"),
                    help="output JSON path (default: next to this script)")
    ap.add_argument("--bw-scale", type=float, default=0.01,
                    help="multiply real link bandwidth by this factor (default 0.01)")
    ap.add_argument("--delay-scale", type=float, default=1.0,
                    help="multiply real propagation delay by this factor (default 1.0)")
    ap.add_argument("--controller-ip", default="127.0.0.1")
    ap.add_argument("--controller-port", type=int, default=6653)
    return ap.parse_args()


def read_csv(path):
    if not Path(path).exists():
        sys.exit("ERROR: file not found: %s\n"
                 "Hint: nodes.csv and links.csv must be inside ~/sdn_project/data/" % path)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def is_connected(node_ids, links):
    """Simple BFS check that every switch can reach every other switch."""
    adj = {n: set() for n in node_ids}
    for l in links:
        adj[l["src_node"]].add(l["dst_node"])
        adj[l["dst_node"]].add(l["src_node"])
    start = node_ids[0]
    seen, stack = {start}, [start]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(node_ids)


def main():
    args = parse_args()
    nodes = read_csv(Path(args.data_dir) / "nodes.csv")
    links = read_csv(Path(args.data_dir) / "links.csv")

    if len(nodes) > 254:
        sys.exit("ERROR: too many nodes for a /24 address plan")

    # ---------- switches and hosts ----------
    node_index = {}
    switches, hosts = [], []
    for i, row in enumerate(nodes, start=1):
        node_id = row["node_id"].strip()
        node_index[node_id] = i
        switches.append({
            "name": "s%d" % i,
            "dpid": i,
            "dpid_hex": "%016x" % i,
            "node_id": node_id,
            "longitude": float(row["longitude"]),
            "latitude": float(row["latitude"]),
            "host": "h%d" % i,
        })
        hosts.append({
            "name": "h%d" % i,
            "ip": "10.0.0.%d" % i,
            "ip_cidr": "10.0.0.%d/24" % i,
            "mac": "00:00:00:00:00:%02x" % i,
            "switch": "s%d" % i,
            "switch_port": 1,
        })

    # ---------- inter-switch links ----------
    next_port = {i: 2 for i in range(1, len(nodes) + 1)}   # port 1 is the host
    out_links = []
    seen_pairs = set()
    for row in links:
        a = row["source"].strip()
        b = row["target"].strip()
        if a not in node_index or b not in node_index:
            sys.exit("ERROR: link %s refers to an unknown node (%s, %s)" % (row["link_id"], a, b))
        pair = frozenset((a, b))
        if pair in seen_pairs:
            sys.exit("ERROR: duplicate link between %s and %s" % (a, b))
        seen_pairs.add(pair)

        ia, ib = node_index[a], node_index[b]
        pa, pb = next_port[ia], next_port[ib]
        next_port[ia] += 1
        next_port[ib] += 1

        real_bw = float(row["bandwidth_mbps"])
        real_delay = float(row["propagation_delay_ms"])
        if real_bw <= 0:
            sys.exit("ERROR: link %s has non-positive bandwidth" % row["link_id"])

        out_links.append({
            "link_id": row["link_id"].strip(),
            "src_node": a,
            "dst_node": b,
            "src_switch": "s%d" % ia,
            "dst_switch": "s%d" % ib,
            "src_port": pa,
            "dst_port": pb,
            "bandwidth_real_mbps": real_bw,
            "bandwidth_emulated_mbps": round(real_bw * args.bw_scale, 3),
            "delay_real_ms": real_delay,
            "delay_ms": round(real_delay * args.delay_scale, 3),
            "distance_km": float(row["distance_km"]),
        })

    if not is_connected([s["node_id"] for s in switches], out_links):
        sys.exit("ERROR: the topology is not connected - check links.csv")

    topo = {
        "meta": {
            "description": "Abilene backbone (SNDlib) mapped to a Mininet network",
            "bw_scale": args.bw_scale,
            "delay_scale": args.delay_scale,
            "subnet": "10.0.0.0/24",
            "openflow_version": "OpenFlow13",
            "controller_ip": args.controller_ip,
            "controller_port": args.controller_port,
            "host_port_on_switch": 1,
        },
        "switches": switches,
        "hosts": hosts,
        "links": out_links,
    }

    with open(args.out, "w") as f:
        json.dump(topo, f, indent=2)

    # ---------- human-readable summary ----------
    print("topology_map.json written to: %s" % args.out)
    print("switches: %d | hosts: %d | inter-switch links: %d | bw_scale: %s | delay_scale: %s"
          % (len(switches), len(hosts), len(out_links), args.bw_scale, args.delay_scale))
    print()
    print("%-4s %-4s %-8s %-15s %-17s" % ("sw", "host", "node", "host IP", "host MAC"))
    for s, h in zip(switches, hosts):
        print("%-4s %-4s %-8s %-15s %-17s" % (s["name"], h["name"], s["node_id"], h["ip"], h["mac"]))
    print()
    print("%-16s %-9s %-7s %-9s %-10s %-9s" % ("link_id", "switches", "ports", "real_bw", "emul_bw", "delay_ms"))
    for l in out_links:
        print("%-16s %-9s %-7s %-9.0f %-10.2f %-9.3f" % (
            l["link_id"], l["src_switch"] + "-" + l["dst_switch"],
            "%d-%d" % (l["src_port"], l["dst_port"]),
            l["bandwidth_real_mbps"], l["bandwidth_emulated_mbps"], l["delay_ms"]))
    print()
    print("OK: next step is  sudo python3 abilene_topo.py --mode standalone --selftest")


if __name__ == "__main__":
    main()
