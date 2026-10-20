#!/usr/bin/env python3
"""
server.py - Live dashboard backend (Phase F, UI part). No sudo needed.
Serves: /                 index.html
        /api/topology     topology_map.json
        /api/live         {state: live_state.json or null,
                           links: {link_id: {present, up, mbps}}}
Link status and traffic come from the Linux kernel (/sys/class/net, /proc/net/dev),
the same counters bwm-ng reads.
USAGE:  python3 server.py [--port 8090] [--host 127.0.0.1]
"""
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAP = ROOT / "topology" / "topology_map.json"
STATE = ROOT / "results" / "live_state.json"
INDEX = HERE / "index.html"
BAD = ("down", "lowerlayerdown", "missing")

topo = json.load(open(MAP))
prev = {}
lock = threading.Lock()


def read_counters():
    out = {}
    with open("/proc/net/dev") as f:
        for line in f.readlines()[2:]:
            name, data = line.split(":", 1)
            c = data.split()
            # rx_bytes, tx_bytes, rx_packets, tx_packets
            out[name.strip()] = (int(c[0]), int(c[8]), int(c[1]), int(c[9]))
    return out



def operstate(iface):
    try:
        with open("/sys/class/net/%s/operstate" % iface) as f:
            return f.read().strip()
    except OSError:
        return "missing"


def link_status():
    now = time.time()
    cnt = read_counters()
    res = {}
    with lock:
        for l in topo["links"]:
            ia = "%s-eth%d" % (l["src_switch"], l["src_port"])
            ib = "%s-eth%d" % (l["dst_switch"], l["dst_port"])
            present = ia in cnt and ib in cnt
            up = present and operstate(ia) not in BAD and operstate(ib) not in BAD
            mbps = fwd = rev = 0.0
            if ia in cnt:
                rx, tx, rxp, txp = cnt[ia]
                p = prev.get(ia)
                if p and now > p[0]:
                    dt = now - p[0]
                    mbps = ((rx - p[1]) + (tx - p[2])) * 8.0 / dt / 1e6
                    fwd = (txp - p[4]) / dt   # packets/s leaving the src switch (src -> dst)
                    rev = (rxp - p[3]) / dt   # packets/s arriving at the src switch (dst -> src)
                prev[ia] = (now, rx, tx, rxp, txp)
            res[l["link_id"]] = {"present": present, "up": up,
                                 "mbps": round(max(mbps, 0.0), 2),
                                 "fwd_pps": round(max(fwd, 0.0), 1),
                                 "rev_pps": round(max(rev, 0.0), 1)}
    return res


def read_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(INDEX.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/topology":
            self._send(json.dumps(topo).encode(), "application/json")
        elif path == "/api/live":
            body = json.dumps({"state": read_state(), "links": link_status()}).encode()
            self._send(body, "application/json")
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print("Dashboard: http://%s:%d   (Ctrl+C to stop)" % (a.host, a.port))
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()

