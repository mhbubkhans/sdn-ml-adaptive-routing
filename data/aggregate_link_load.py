"""
Turn SNDlib Abilene demand-matrix snapshots (O-D traffic, no link info) into
a time-varying LINK-LEVEL dataset, by routing each demand over the real
15-link topology (shortest path, weighted by propagation delay) and summing
the traffic that crosses each physical link.

Requires: links.csv, nodes.csv  (from parse_sndlib_abilene.py)
Input:    a folder of demandMatrix-abilene-zhang-5min-YYYYMMDD-HHMM.txt files
Output:   link_timeseries.csv with columns:
          timestamp, link_id, source, target, bandwidth_mbps,
          propagation_delay_ms, aggregated_load_mbps, utilization_pct

Usage:
    python aggregate_link_load.py <demand_folder> <links_csv> <output_csv>
"""

import re
import csv
import glob
import os
import sys
import networkx as nx


def extract_block(text, section_name):
    start = text.find(f"{section_name} (")
    if start == -1:
        start = text.find(f"{section_name}(")
    start = text.find("(", start)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    raise ValueError(f"Unbalanced parentheses in section {section_name}")


def parse_demands(text):
    block = extract_block(text, "DEMANDS")
    pattern = re.compile(
        r"(\S+)\s*\(\s*(\S+)\s+(\S+)\s*\)\s*"
        r"(\d+)\s+(-?\d+\.?\d*)\s+(\S+)"
    )
    demands = []
    for demand_id, src, tgt, _unit, value, _maxlen in pattern.findall(block):
        demands.append((src, tgt, float(value)))
    return demands


def parse_timestamp_from_filename(fname):
    # demandMatrix-abilene-zhang-5min-20040406-1220.txt -> "20040406-1220"
    m = re.search(r"(\d{8}-\d{4})", fname)
    return m.group(1) if m else fname


def load_topology(links_csv):
    G = nx.Graph()
    link_lookup = {}  # frozenset({a,b}) -> dict(link_id, bandwidth_mbps, delay_ms)
    with open(links_csv, newline="") as f:
        for row in csv.DictReader(f):
            src, tgt = row["source"], row["target"]
            bw = float(row["bandwidth_mbps"])
            delay = float(row["propagation_delay_ms"])
            G.add_edge(src, tgt, weight=delay)
            link_lookup[frozenset({src, tgt})] = {
                "link_id": row["link_id"],
                "source": src,
                "target": tgt,
                "bandwidth_mbps": bw,
                "propagation_delay_ms": delay,
            }
    return G, link_lookup


def main(demand_folder, links_csv, output_csv):
    G, link_lookup = load_topology(links_csv)

    files = sorted(glob.glob(os.path.join(demand_folder, "demandMatrix-*.txt")))
    if not files:
        print(f"No demandMatrix-*.txt files found in {demand_folder}")
        return

    rows = []
    for fpath in files:
        fname = os.path.basename(fpath)
        timestamp = parse_timestamp_from_filename(fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        demands = parse_demands(text)

        # reset per-timestamp load accumulator
        link_load = {key: 0.0 for key in link_lookup}

        skipped = 0
        for src, tgt, value in demands:
            if src == tgt or src not in G or tgt not in G:
                continue
            try:
                path = nx.shortest_path(G, src, tgt, weight="weight")
            except nx.NetworkXNoPath:
                skipped += 1
                continue
            for a, b in zip(path[:-1], path[1:]):
                key = frozenset({a, b})
                if key in link_load:
                    link_load[key] += value

        for key, load in link_load.items():
            info = link_lookup[key]
            util_pct = round(100 * load / info["bandwidth_mbps"], 4)
            rows.append({
                "timestamp": timestamp,
                "link_id": info["link_id"],
                "source": info["source"],
                "target": info["target"],
                "bandwidth_mbps": info["bandwidth_mbps"],
                "propagation_delay_ms": info["propagation_delay_ms"],
                "aggregated_load_mbps": round(load, 4),
                "utilization_pct": util_pct,
            })

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Processed {len(files)} timestamp file(s) -> {len(rows)} rows")
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    demand_folder = sys.argv[1] if len(sys.argv) > 1 else "demand_samples"
    links_csv = sys.argv[2] if len(sys.argv) > 2 else "links.csv"
    output_csv = sys.argv[3] if len(sys.argv) > 3 else "link_timeseries.csv"
    main(demand_folder, links_csv, output_csv)
