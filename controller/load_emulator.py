#!/usr/bin/env python3
"""
load_emulator.py   (Phase D - Steps D5, D6)

WHAT IT DOES
  Makes the Mininet network FEEL the replayed Abilene load. Every time the ML
  controller writes a new step to live_state.json, this program sets, for each
  of the 15 links,
        delay      = propagation delay + queue delay        (see load_model.py)
        bandwidth  = emulated bandwidth x (1 - utilization) (never below 5 %)
  with the Linux tool tc. So a ping or iperf that crosses a busy link really gets
  slower, and a controller that routes around the busy link really gets better
  results.

  It talks to the interfaces with plain tc commands (no Mininet objects), so it
  can run in its own terminal, or inside phase_d_tests.py.

IMPORTANT HONEST NOTE
  The replayed load is "open loop": it comes from the dataset and does not change
  when your controller moves the (small) test traffic. The experiment therefore
  measures how well the controller steers the monitored flows around the load that
  is already there.

USAGE (sudo, while the network is running)
  sudo python3 load_emulator.py              # follow live_state.json, restore links at the end
  sudo python3 load_emulator.py --dry-run    # only print the tc commands
  sudo python3 load_emulator.py --restore    # put every link back to its normal values and exit
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import load_model as LM  # noqa: E402

DEFAULT_STATE = str(HERE.parent / "results" / "live_state.json")
DEFAULT_MAP = str(HERE.parent / "topology" / "topology_map.json")


def read_state(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (IOError, OSError, ValueError):
        return None


def is_fresh(state, max_age=120.0):
    return state is not None and (time.time() - state.get("written_at", 0)) < max_age


class LoadEmulator(object):
    def __init__(self, topo_data, state_path=DEFAULT_STATE, dry_run=False, log=print, min_change=0.02):
        self.links = {l["link_id"]: l for l in topo_data["links"]}
        self.ifaces = {lid: ["%s-eth%d" % (l["src_switch"], l["src_port"]),
                             "%s-eth%d" % (l["dst_switch"], l["dst_port"])] for lid, l in self.links.items()}
        self.state_path = state_path
        self.dry_run = dry_run
        self.log = log
        self.min_change = min_change
        self.applied = {}            # link_id -> (bw, delay) last applied
        self.built = set()           # interfaces that already have our tc structure
        self.last_step = None
        self.last_state = None
        self._stop = threading.Event()
        self._thread = None
        self.tc_errors = 0

    # ------------------------------------------------------------ tc primitives
    def _tc(self, args):
        cmd = ["tc"] + args
        if self.dry_run:
            self.log("   [dry-run] " + " ".join(cmd))
            return 0
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        return r.returncode

    def _rebuild(self, iface, bw, delay):
        """Create our own htb + netem structure (same layout Mininet's TCLink uses)."""
        self._tc(["qdisc", "del", "dev", iface, "root"])
        ok = self._tc(["qdisc", "add", "dev", iface, "root", "handle", "5:", "htb", "default", "1"]) == 0
        ok = ok and self._tc(["class", "add", "dev", iface, "parent", "5:", "classid", "5:1", "htb",
                              "rate", "%.3fMbit" % bw, "burst", "15k"]) == 0
        ok = ok and self._tc(["qdisc", "add", "dev", iface, "parent", "5:1", "handle", "10:", "netem",
                              "delay", "%.3fms" % delay, "limit", "1000"]) == 0
        if ok:
            self.built.add(iface)
        else:
            self.tc_errors += 1
        return ok

    def _set_iface(self, iface, bw, delay):
        if iface in self.built:
            ok = self._tc(["class", "change", "dev", iface, "parent", "5:", "classid", "5:1", "htb",
                           "rate", "%.3fMbit" % bw, "burst", "15k"]) == 0
            ok = ok and self._tc(["qdisc", "change", "dev", iface, "parent", "5:1", "handle", "10:", "netem",
                                  "delay", "%.3fms" % delay, "limit", "1000"]) == 0
            if ok:
                return True
            self.built.discard(iface)           # something changed under us: rebuild
        return self._rebuild(iface, bw, delay)

    # ------------------------------------------------------------ applying
    def target_for(self, lid, util_pct):
        l = self.links[lid]
        return (LM.effective_bw_mbps(l["bandwidth_emulated_mbps"], util_pct),
                LM.effective_delay_ms(l["delay_ms"], util_pct))

    def _apply_link(self, lid, bw, delay):
        prev = self.applied.get(lid)
        if prev is not None:
            if abs(bw - prev[0]) <= self.min_change * prev[0] and abs(delay - prev[1]) <= max(0.05, self.min_change * prev[1]):
                return False
        for iface in self.ifaces[lid]:
            self._set_iface(iface, bw, delay)
        self.applied[lid] = (bw, delay)
        return True

    def apply_state(self, state):
        n = 0
        for lid, u in state.get("util", {}).items():
            if lid in self.links:
                bw, delay = self.target_for(lid, u)
                n += 1 if self._apply_link(lid, bw, delay) else 0
        return n

    def restore(self):
        """Put every link back to its normal (no extra load) values."""
        for lid, l in self.links.items():
            for iface in self.ifaces[lid]:
                self._set_iface(iface, float(l["bandwidth_emulated_mbps"]), float(l["delay_ms"]))
            self.applied[lid] = (float(l["bandwidth_emulated_mbps"]), float(l["delay_ms"]))
        self.log("[load emulator] all links restored to normal values")

    # ------------------------------------------------------------ following the controller
    def poll_once(self):
        st = read_state(self.state_path)
        if not is_fresh(st):
            return None
        if st.get("finished"):
            if self.last_step != "finished":
                self.last_step = "finished"
                self.restore()
            return st
        if st.get("step") != self.last_step:
            self.last_step = st["step"]
            changed = self.apply_state(st)
            self.last_state = st
            self.log("[load emulator] step %d/%d %s: %d link(s) updated" % (st["step"] + 1, st["steps"], st["timestamp"], changed))
        return st

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:           # never kill the thread
                self.log("[load emulator] error: %s" % exc)
            self._stop.wait(0.4)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, restore=True):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if restore:
            self.restore()


def main():
    ap = argparse.ArgumentParser(description="Apply the replayed load to the Mininet links")
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true", help="restore all links and exit")
    args = ap.parse_args()
    if os.geteuid() != 0 and not args.dry_run:
        sys.exit("ERROR: tc needs root. Run:  sudo python3 load_emulator.py")
    with open(args.map) as f:
        topo_data = json.load(f)
    em = LoadEmulator(topo_data, args.state, args.dry_run)
    if args.restore:
        em.restore()
        return
    print("Load emulator running. Waiting for the controller's replay (live_state.json) ... Ctrl+C to stop.")
    try:
        while True:
            st = em.poll_once()
            if st and st.get("finished"):
                time.sleep(2)
            time.sleep(0.4)
    except KeyboardInterrupt:
        pass
    finally:
        em.restore()


if __name__ == "__main__":
    main()
