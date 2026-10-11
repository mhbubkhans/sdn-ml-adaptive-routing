#!/usr/bin/env python3
"""
routing_core.py   (Phase C - shared module)

WHAT IT DOES
  All the ROUTING MATHS of the project lives here, in plain Python with no extra
  libraries (no networkx, no Ryu, no Mininet). So it can be used by
    - baseline_controller.py   (Phase C: shortest path by delay)
    - the ML controller        (Phase D: same code, but with costs predicted by ML)
    - phase_c_tests.py         (to know what the correct answer is)
    - check_routing_core.py    (offline self-check, no sudo needed)

  It reads topology_map.json (made in Phase A) and answers:
    "For every switch and every destination host, which OUTPUT PORT must a
     packet leave through so that it follows the cheapest path?"

THE IDEA IN ONE PARAGRAPH
  For each destination host we run Dijkstra's algorithm starting at the switch
  that host is attached to. This gives a "shortest-path tree" pointing toward
  that destination. Every other switch simply forwards packets for that
  destination to its next hop on the tree. Trees built from one consistent cost
  can never contain loops, which matters because Abilene itself contains loops.
  A flow entry is then:   match eth_dst = <host MAC>   ->   output <port>.

HOW COSTS ARE CHOSEN
  metric "delay" (default): cost of a link = its delay in ms   (like a latency-aware OSPF)
  metric "hops"           : every link costs 1
  weights={link_id: cost} : overrides both. Phase D passes ML-predicted costs here.

FLOW KEY FORMAT
  compute_flows() returns {(dpid, destination_mac): output_port}
"""
import heapq
import json
from pathlib import Path


class Topology(object):
    def __init__(self, data, metric="delay"):
        if metric not in ("delay", "hops"):
            raise ValueError("metric must be 'delay' or 'hops'")
        self.metric = metric
        self.meta = data["meta"]
        self.dpid_of = {s["name"]: int(s["dpid"]) for s in data["switches"]}
        self.name_of = {v: k for k, v in self.dpid_of.items()}
        self.node_of_switch = {s["name"]: s["node_id"] for s in data["switches"]}
        self.hosts = {h["name"]: h for h in data["hosts"]}
        self.links = {l["link_id"]: l for l in data["links"]}

        self.port_to_link = {}        # (dpid, port) -> link_id
        self.neighbor_port = {}       # (dpid, neighbour_dpid) -> port on dpid that leads there
        self.adj = {d: [] for d in self.name_of}   # dpid -> [(neighbour_dpid, link_id)]
        for l in data["links"]:
            a, b = self.dpid_of[l["src_switch"]], self.dpid_of[l["dst_switch"]]
            self.port_to_link[(a, l["src_port"])] = l["link_id"]
            self.port_to_link[(b, l["dst_port"])] = l["link_id"]
            self.neighbor_port[(a, b)] = l["src_port"]
            self.neighbor_port[(b, a)] = l["dst_port"]
            self.adj[a].append((b, l["link_id"]))
            self.adj[b].append((a, l["link_id"]))
        self.link_between = {}
        for l in data["links"]:
            a, b = self.dpid_of[l["src_switch"]], self.dpid_of[l["dst_switch"]]
            self.link_between[frozenset((a, b))] = l["link_id"]

    # ------------------------------------------------------------------ costs
    def cost(self, link_id, weights=None):
        if weights is not None and link_id in weights:
            return float(weights[link_id])
        if self.metric == "hops":
            return 1.0
        return float(self.links[link_id]["delay_ms"])

    # ------------------------------------------------------------------ Dijkstra
    def _dijkstra(self, root, down, weights):
        """Shortest paths FROM `root` (a dpid) to every switch.
        Returns dist and nxt, where nxt[v] is the neighbour v must forward to
        in order to reach root."""
        dist = {root: 0.0}
        nxt = {root: None}
        done = set()
        heap = [(0.0, root)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for v, lid in self.adj[u]:
                if lid in down:
                    continue
                nd = d + self.cost(lid, weights)
                if v not in dist or nd < dist[v] - 1e-12:
                    dist[v] = nd
                    nxt[v] = u
                    heapq.heappush(heap, (nd, v))
        return dist, nxt

    # ------------------------------------------------------------------ flows
    def compute_flows(self, down_links=(), weights=None):
        """The complete forwarding table of the whole network.
        down_links: iterable of link_ids that are currently down."""
        down = set(down_links)
        flows = {}
        for host in self.hosts.values():
            root = self.dpid_of[host["switch"]]
            _, nxt = self._dijkstra(root, down, weights)
            for dpid in self.name_of:
                if dpid == root:
                    flows[(dpid, host["mac"])] = host["switch_port"]
                elif dpid in nxt:
                    flows[(dpid, host["mac"])] = self.neighbor_port[(dpid, nxt[dpid])]
                # else: destination unreachable from this switch -> no entry
        return flows

    # ------------------------------------------------------------------ paths
    def path(self, src_host, dst_host, down_links=(), weights=None):
        """List of switch names from the source host's switch to the destination's,
        or None if unreachable."""
        down = set(down_links)
        s = self.dpid_of[self.hosts[src_host]["switch"]]
        d = self.dpid_of[self.hosts[dst_host]["switch"]]
        _, nxt = self._dijkstra(d, down, weights)
        if s not in nxt:
            return None
        out, cur, guard = [self.name_of[s]], s, 0
        while cur != d:
            cur = nxt[cur]
            out.append(self.name_of[cur])
            guard += 1
            if guard > len(self.name_of):
                return None          # would be a loop (cannot happen with consistent costs)
        return out

    def path_links(self, path):
        """link_ids along a list of switch names."""
        ids = []
        for a, b in zip(path[:-1], path[1:]):
            ids.append(self.link_between[frozenset((self.dpid_of[a], self.dpid_of[b]))])
        return ids

    def path_delay_ms(self, path):
        """Real one-way propagation delay of a path (sum of link delays)."""
        return sum(self.links[l]["delay_ms"] for l in self.path_links(path))

    def expected_rtt_ms(self, path):
        """Ping round-trip time in the emulation is about twice the one-way delay
        (delay is applied in each direction), plus a little overhead."""
        return 2.0 * self.path_delay_ms(path)

    def describe(self, path):
        return " > ".join(self.node_of_switch[s] for s in path)


def load_topology(path=None, metric="delay"):
    """Load topology_map.json (default: ../topology/topology_map.json)."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "topology" / "topology_map.json"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("%s not found. Run Phase A step A3 (make_topology_map.py)." % path)
    with open(path) as f:
        return Topology(json.load(f), metric)
