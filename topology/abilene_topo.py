#!/usr/bin/env python3
"""
abilene_topo.py   (Phase A - Steps A4, A5, A6)

WHAT IT DOES
  Builds the REAL Abilene backbone (12 routers, 15 links) as a Mininet network,
  using topology_map.json created by make_topology_map.py.

  Each Abilene router  -> one Open vSwitch   (s1 .. s12)
  Each switch          -> one host attached  (h1 .. h12)  = "a user in that city"
  Each Abilene link    -> one Mininet link with REAL (scaled) bandwidth and delay

TWO MODES
  --mode standalone   No controller needed. Switches run as normal learning
                      switches with Spanning Tree Protocol (STP) so the loops in
                      Abilene do not melt the network. Use this to TEST the
                      topology (Phase A). Paths are NOT shortest paths here.

  --mode controller   Switches talk OpenFlow 1.3 to a Ryu controller running on
                      this machine (default 127.0.0.1:6653). This is the mode the
                      final ML project uses (Phase C onwards).

USAGE (always with sudo, always from ~/sdn_project/topology)
  sudo python3 abilene_topo.py --mode standalone --selftest
  sudo python3 abilene_topo.py --mode standalone
  sudo python3 abilene_topo.py --mode controller

OTHER OPTIONS
  --selftest     run automatic checks (counts, delay/bandwidth, ping, iperf)
  --no-cli       exit after the selftest instead of opening the mininet> prompt
  --quiet        less Mininet output
  --no-iperf     skip the iperf bandwidth test inside the selftest
  --no-clean     do not run Mininet's automatic cleanup at start

IMPORTING (Phase E)
  from abilene_topo import load_map, build_network, set_link
"""
import argparse
import json
import os
import re
import sys
import time
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# Loading and building
# ----------------------------------------------------------------------------
def load_map(path):
    p = Path(path)
    if not p.exists():
        sys.exit("ERROR: %s not found.\n"
                 "Run this first:  python3 make_topology_map.py" % p)
    with open(p) as f:
        return json.load(f)


def build_network(topo, mode="standalone", ctrl_ip=None, ctrl_port=None):
    """Create (but do not start) the Mininet network.

    Returns: net, link_objs   (link_objs maps link_id -> Mininet link object)
    """
    from mininet.net import Mininet
    from mininet.node import RemoteController, OVSKernelSwitch, OVSBridge
    from mininet.link import TCLink, Link

    meta = topo["meta"]
    if mode == "controller":
        switch_cls = partial(OVSKernelSwitch, protocols=meta["openflow_version"])
    else:
        switch_cls = partial(OVSBridge, stp=True)

    net = Mininet(controller=None, switch=switch_cls, link=TCLink,
                  autoSetMacs=False, autoStaticArp=False)

    if mode == "controller":
        net.addController("c0", controller=RemoteController,
                          ip=ctrl_ip or meta["controller_ip"],
                          port=int(ctrl_port or meta["controller_port"]))

    switches = {}
    for s in topo["switches"]:
        switches[s["name"]] = net.addSwitch(s["name"], dpid=s["dpid_hex"])

    # one host per switch (plain link: no bandwidth/delay limit)
    for h in topo["hosts"]:
        host = net.addHost(h["name"], ip=h["ip_cidr"], mac=h["mac"])
        net.addLink(host, switches[h["switch"]], cls=Link,
                    port1=0, port2=h["switch_port"])

    # the 15 real Abilene links
    link_objs = {}
    for l in topo["links"]:
        link_objs[l["link_id"]] = net.addLink(
            switches[l["src_switch"]], switches[l["dst_switch"]],
            cls=TCLink, port1=l["src_port"], port2=l["dst_port"],
            bw=l["bandwidth_emulated_mbps"], delay="%gms" % l["delay_ms"])
    return net, link_objs


def set_link(topo, link_objs, link_id, bw_mbps=None, delay_ms=None):
    """Change bandwidth and/or delay of a RUNNING link (used in Phase E).

    Mininet re-applies BOTH values every time, so we always pass both.
    """
    spec = next(l for l in topo["links"] if l["link_id"] == link_id)
    bw = spec["bandwidth_emulated_mbps"] if bw_mbps is None else bw_mbps
    dl = spec["delay_ms"] if delay_ms is None else delay_ms
    lo = link_objs[link_id]
    for intf in (lo.intf1, lo.intf2):
        intf.config(bw=bw, delay="%gms" % dl)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def reachable(src, dst):
    out = src.cmd("ping -c 1 -W 1 %s" % dst.IP())
    return re.search(r"\b1 received", out) is not None


def wait_for_stp(src, dst, timeout):
    """STP needs ~30-50 s before traffic flows. Poll until src can ping dst."""
    sys.stdout.write("Waiting for Spanning Tree to converge (up to %d s) " % timeout)
    sys.stdout.flush()
    start = time.time()
    while time.time() - start < timeout:
        if reachable(src, dst):
            print(" done after %d s" % (time.time() - start))
            return True
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(3)
    print(" TIMEOUT")
    return False


def parse_delay_ms(text):
    m = re.search(r"delay\s+([\d.]+)\s*(us|ms|s)\b", text)
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2)
    return v / 1000.0 if unit == "us" else (v if unit == "ms" else v * 1000.0)


def parse_rate_mbit(text):
    m = re.search(r"rate\s+([\d.]+)\s*([KMG]?)bit", text)
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2)
    if u == "K":
        return v / 1000.0
    if u == "G":
        return v * 1000.0
    if u == "M":
        return v
    return v / 1e6


class Report:
    def __init__(self):
        self.counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    def say(self, tag, msg):
        if tag in self.counts:
            self.counts[tag] += 1
        print("[%s] %s" % (tag, msg))


# ----------------------------------------------------------------------------
# Self test
# ----------------------------------------------------------------------------
def run_selftest(net, topo, link_objs, mode, do_iperf):
    rep = Report()
    print("\n================ PHASE A SELF-TEST ================")
    n_sw, n_h, n_l = len(topo["switches"]), len(topo["hosts"]), len(topo["links"])

    # 1. object counts
    rep.say("PASS" if len(net.switches) == n_sw else "FAIL",
            "switches: %d (expected %d)" % (len(net.switches), n_sw))
    rep.say("PASS" if len(net.hosts) == n_h else "FAIL",
            "hosts: %d (expected %d)" % (len(net.hosts), n_h))
    rep.say("PASS" if len(net.links) == n_h + n_l else "FAIL",
            "links: %d (expected %d host links + %d Abilene links = %d)"
            % (len(net.links), n_h, n_l, n_h + n_l))

    # 2. delay and bandwidth really applied by Linux traffic control (tc)
    bad, total = [], 0
    for l in topo["links"]:
        lo = link_objs[l["link_id"]]
        for intf in (lo.intf1, lo.intf2):
            total += 1
            node = intf.node
            d = parse_delay_ms(node.cmd("tc qdisc show dev %s" % intf.name))
            r = parse_rate_mbit(node.cmd("tc class show dev %s" % intf.name))
            exp_d, exp_r = l["delay_ms"], l["bandwidth_emulated_mbps"]
            ok_d = d is not None and abs(d - exp_d) <= max(0.02, 0.03 * exp_d)
            ok_r = r is not None and abs(r - exp_r) <= 0.02 * exp_r
            if not (ok_d and ok_r):
                bad.append("%s on %s: delay %s ms (want %s), rate %s Mbit (want %s)"
                           % (l["link_id"], intf.name, d, exp_d, r, exp_r))
    if not bad:
        rep.say("PASS", "all %d link interfaces have the expected delay and bandwidth" % total)
    else:
        rep.say("WARN", "%d of %d interfaces did not match (may only be tc print format):" % (len(bad), total))
        for line in bad[:6]:
            print("        " + line)

    # 3. forwarding test
    if mode == "standalone":
        loss = net.pingAll()
        if loss == 0:
            rep.say("PASS", "pingall: 0% dropped, every host reaches every host")
        else:
            rep.say("FAIL", "pingall: %s%% dropped (wait 30 s and run 'pingall' again at mininet>)" % loss)
        if do_iperf:
            try:
                h_first, h_last = net.get("h1"), net.get("h%d" % n_h)
                res = net.iperf([h_first, h_last], seconds=3)
                rep.say("PASS", "iperf h1 <-> h%d (STP path, limited by slowest link): %s" % (n_h, res))
            except Exception as exc:  # noqa
                rep.say("WARN", "iperf could not run: %s" % exc)
    else:
        up = [s.name for s in net.switches if s.connected()]
        if len(up) == n_sw:
            rep.say("PASS", "all %d switches are connected to the Ryu controller" % n_sw)
        else:
            rep.say("FAIL", "only %d of %d switches connected (is ryu-manager running?)" % (len(up), n_sw))
        print("[INFO] pingall is NOT expected to work until your controller installs flows (Phase C)")

    c = rep.counts
    print("---------------------------------------------------")
    print("RESULT: %d PASS, %d WARN, %d FAIL" % (c["PASS"], c["WARN"], c["FAIL"]))
    print("===================================================\n")
    return c["FAIL"] == 0


def print_banner(mode):
    print("\n" + "=" * 62)
    print(" Abilene network is UP  (mode: %s)" % mode)
    print(" Try:  pingall | net | links | h1 ping -c 3 h12 | iperf h1 h12")
    print("       sh tc qdisc show dev s2-eth2 | sh ovs-vsctl show")
    print("       link s2 s6 down   |   link s2 s6 up")
    print(" Leave with:  exit   (then it cleans up automatically)")
    print("=" * 62 + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Build the Abilene network in Mininet")
    ap.add_argument("--map", default=str(HERE / "topology_map.json"))
    ap.add_argument("--mode", choices=["standalone", "controller"], default="standalone")
    ap.add_argument("--controller-ip", default=None)
    ap.add_argument("--controller-port", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-cli", action="store_true")
    ap.add_argument("--no-iperf", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--stp-timeout", type=int, default=150)
    return ap.parse_args()


def main():
    args = parse_args()
    if os.geteuid() != 0:
        sys.exit("ERROR: Mininet needs root. Run:  sudo python3 abilene_topo.py ...")
    topo = load_map(args.map)

    from mininet.log import setLogLevel
    from mininet.cli import CLI
    setLogLevel("warning" if args.quiet else "info")

    if not args.no_clean:
        try:
            from mininet.clean import cleanup
            cleanup()
        except Exception as exc:  # noqa
            print("(cleanup skipped: %s)" % exc)

    net, link_objs = build_network(topo, args.mode, args.controller_ip, args.controller_port)
    exit_code = 0
    try:
        net.start()
        net.staticArp()          # hosts know each other's MAC: no ARP floods in a looped network
        n_h = len(topo["hosts"])

        if args.mode == "standalone":
            ok = wait_for_stp(net.get("h1"), net.get("h%d" % n_h), args.stp_timeout)
            if not ok:
                print("WARNING: STP did not converge in time. Try 'pingall' again in a minute.")
        else:
            print("Waiting up to 15 s for switches to connect to the controller ...")
            t0 = time.time()
            while time.time() - t0 < 15:
                if all(s.connected() for s in net.switches):
                    break
                time.sleep(1)

        if args.selftest:
            if not run_selftest(net, topo, link_objs, args.mode, not args.no_iperf):
                exit_code = 1
        if not args.no_cli:
            print_banner(args.mode)
            CLI(net)
    finally:
        net.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
