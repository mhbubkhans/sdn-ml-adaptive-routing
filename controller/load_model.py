#!/usr/bin/env python3
"""
load_model.py   (Phase D - shared module, pure Python)

WHAT IT DOES
  Holds the three small pieces of maths that the offline evaluator, the Ryu ML
  controller, the load emulator and the tests must ALL agree on:

  1. LOAD -> LINK QUALITY MODEL
     A link that carries more traffic is slower and has less spare capacity.
     We use the classic queueing shape:
         queue delay (ms)      = Q_MS * u / (1 - u)              u = utilization (0..0.95)
         effective delay (ms)  = propagation delay + queue delay
         effective bandwidth   = bandwidth * (1 - u)   (never below 5 % of the link)
     Example (Q_MS = 3): u = 50 % adds 3 ms, u = 80 % adds 12 ms, u >= 95 % adds 57 ms.
     This is what the load emulator applies to the Mininet links, and what the
     offline evaluator uses to score paths.

  2. ROUTING COST FUNCTIONS (how a utilization number becomes a link cost)
     "quadratic" : cost = delay * (1 + k * u^2)    (u capped at 150 %), k is tuned
     "queue"     : cost = delay + queue delay      (uses the model above directly)

  3. TIMELINE READER
     Reads link_timeseries.csv into {timestamp: {link_id: utilization_pct}}.

  Utilization values above 100 % in the dataset mean "offered load above capacity".
  The model caps u at 0.95, because a real link cannot be more than full.
"""
import csv
from datetime import datetime, timedelta

Q_MS = 3.0          # queue delay scale in milliseconds
U_CAP = 0.95        # utilization cap used by the queueing model
STEP_MINUTES = 5


def _u(util_pct):
    return min(max(float(util_pct), 0.0) / 100.0, U_CAP)


def queue_delay_ms(util_pct, q_ms=Q_MS):
    u = _u(util_pct)
    return q_ms * u / (1.0 - u)


def effective_delay_ms(base_delay_ms, util_pct):
    return float(base_delay_ms) + queue_delay_ms(util_pct)


def effective_bw_mbps(base_bw_mbps, util_pct):
    u = _u(util_pct)
    return max(float(base_bw_mbps) * (1.0 - u), float(base_bw_mbps) * (1.0 - U_CAP))


def link_cost(delay_ms, util_pct, model="quadratic", k=6.0, cap_pct=150.0):
    """Routing cost of one link for a given (forecast or current) utilization."""
    if model == "queue":
        return float(delay_ms) + queue_delay_ms(util_pct)
    u = min(max(float(util_pct), 0.0), cap_pct) / 100.0
    return float(delay_ms) * (1.0 + k * u * u)


def path_latency_ms(topo, path, util_by_link):
    """One-way latency of a path under the load model."""
    return sum(effective_delay_ms(topo.links[l]["delay_ms"], util_by_link.get(l, 0.0))
               for l in topo.path_links(path))


def path_max_util(topo, path, util_by_link):
    ids = topo.path_links(path)
    return max(util_by_link.get(l, 0.0) for l in ids) if ids else 0.0


def load_timeline(csv_path):
    """-> (sorted list of timestamps, {timestamp: {link_id: utilization_pct}})"""
    util = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            util.setdefault(row["timestamp"], {})[row["link_id"]] = float(row["utilization_pct"])
    return sorted(util), util


def parse_ts(ts):
    return datetime.strptime(ts, "%Y%m%d-%H%M")


def fmt_ts(dt):
    return dt.strftime("%Y%m%d-%H%M")


def add_steps(ts, n):
    return fmt_ts(parse_ts(ts) + timedelta(minutes=STEP_MINUTES * n))


def pretty_ts_to_compact(text):
    """'2004-04-08 02:20:00' -> '20040408-0220' (the format stored in the model file)."""
    return datetime.strptime(text.strip()[:16], "%Y-%m-%d %H:%M").strftime("%Y%m%d-%H%M")
