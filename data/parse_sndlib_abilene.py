"""
Parse an SNDlib native-format network file (tested on Abilene) into three
clean CSVs ready to feed into a Mininet topology script and an ML model:

  1. nodes.csv   -> node_id, longitude, latitude
  2. links.csv   -> link_id, source, target, bandwidth_mbps,
                     distance_km, propagation_delay_ms
  3. demands.csv -> demand_id, source, target, demand_mbps

SNDlib does not include a real per-link delay field (routing_cost is left
as 0.00 in the Abilene file), so this script derives an honest, defensible
propagation delay from each node pair's real geographic coordinates:

    delay_ms = great_circle_distance_km * (1000 / fiber_speed_km_per_s)

Fiber propagation speed is taken as ~2/3 the speed of light in vacuum,
the standard approximation used in networking texts and papers.

Usage:
    python parse_sndlib_abilene.py abilene_network.txt
"""

import re
import csv
import math
import sys

SPEED_OF_LIGHT_KM_S = 299_792.458
FIBER_SPEED_KM_S = SPEED_OF_LIGHT_KM_S * (2 / 3)  # ~199,861.6 km/s


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance between two (lon, lat) points, in km."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def extract_block(text, section_name):
    """Pull out the raw text inside SECTION_NAME ( ... ) balancing parens."""
    start = text.find(f"{section_name} (")
    if start == -1:
        start = text.find(f"{section_name}(")
    if start == -1:
        raise ValueError(f"Section {section_name} not found")
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


def parse_nodes(block):
    # NodeId ( lon lat )
    pattern = re.compile(r"(\S+)\s*\(\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*\)")
    nodes = {}
    for node_id, lon, lat in pattern.findall(block):
        nodes[node_id] = (float(lon), float(lat))
    return nodes


def parse_links(block):
    # LinkId ( src tgt ) pre_cap pre_cost routing_cost setup_cost ( modules... )
    pattern = re.compile(
        r"(\S+)\s*\(\s*(\S+)\s+(\S+)\s*\)\s*"
        r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
    )
    links = []
    for link_id, src, tgt, pre_cap, pre_cost, routing_cost, setup_cost in pattern.findall(block):
        links.append({
            "link_id": link_id,
            "source": src,
            "target": tgt,
            "bandwidth_mbps": float(pre_cap),
        })
    return links


def parse_demands(block):
    # DemandId ( src tgt ) routing_unit demand_value max_path_length
    pattern = re.compile(
        r"(\S+)\s*\(\s*(\S+)\s+(\S+)\s*\)\s*"
        r"(\d+)\s+(-?\d+\.?\d*)\s+(\S+)"
    )
    demands = []
    for demand_id, src, tgt, _unit, value, _maxlen in pattern.findall(block):
        demands.append({
            "demand_id": demand_id,
            "source": src,
            "target": tgt,
            "demand_mbps": float(value),
        })
    return demands


def main(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    nodes = parse_nodes(extract_block(text, "NODES"))
    links = parse_links(extract_block(text, "LINKS"))
    demands = parse_demands(extract_block(text, "DEMANDS"))

    # --- nodes.csv ---
    with open("nodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "longitude", "latitude"])
        for node_id, (lon, lat) in nodes.items():
            w.writerow([node_id, lon, lat])

    # --- links.csv (bandwidth from SNDlib + derived propagation delay) ---
    with open("links.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "link_id", "source", "target", "bandwidth_mbps",
            "distance_km", "propagation_delay_ms"
        ])
        for link in links:
            lon1, lat1 = nodes[link["source"]]
            lon2, lat2 = nodes[link["target"]]
            dist_km = haversine_km(lon1, lat1, lon2, lat2)
            delay_ms = (dist_km / FIBER_SPEED_KM_S) * 1000
            w.writerow([
                link["link_id"], link["source"], link["target"],
                link["bandwidth_mbps"], round(dist_km, 2), round(delay_ms, 4)
            ])

    # --- demands.csv ---
    with open("demands.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["demand_id", "source", "target", "demand_mbps"])
        for d in demands:
            w.writerow([d["demand_id"], d["source"], d["target"], d["demand_mbps"]])

    print(f"Parsed {len(nodes)} nodes, {len(links)} links, {len(demands)} demands.")
    print("Wrote: nodes.csv, links.csv, demands.csv")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "abilene_network.txt"
    main(path)
