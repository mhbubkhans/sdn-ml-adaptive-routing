"""
baseline_controller.py   (Phase C - the baseline Ryu controller)

WHAT IT DOES
  Routes traffic over the Abilene network along the shortest path by delay.
  It is the BASELINE: no machine learning, similar in spirit to a latency-aware
  OSPF. In Phase E the ML controller is compared against it.

HOW IT WORKS
  1. It knows the whole network in advance from topology_map.json (made in Phase A):
     switches, hosts (IP, MAC, which switch, which port) and every link with its
     ports and delay. So it needs no discovery and no learning.
  2. When a switch connects it
        - deletes any old flows on that switch,
        - installs ONE flow per destination host:
              match eth_dst = <MAC of that host>   ->   output <port toward it>
          (12 flows per switch, 144 in the whole network).
  3. When a link goes down (or up) the switch sends a PortStatus message. The
     controller marks the link, recomputes all routes and updates ONLY the flow
     entries that changed. This is how traffic is rerouted live.
  4. Packets that match no flow are dropped. (Hosts use static ARP, set by
     abilene_topo.py, so no broadcasts are needed.)

RUN   (no sudo)
  ryu-manager ~/sdn_project/controller/baseline_controller.py

OPTIONS (environment variables, put them before the command)
  BASELINE_METRIC=hops        use hop count instead of delay
  BASELINE_LOG_MISSES=1       log the first 20 packets that match no flow (debugging)
  ABILENE_MAP=/path/map.json  use another topology map

PHASE D EXTENSION POINT
  get_weights() returns None here, meaning "use the delay of each link". The ML
  controller will subclass this class and return predicted costs instead.
"""
import os
import sys
import time

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib.packet import ethernet, packet
from ryu.ofproto import ofproto_v1_3

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                                   # very unusual loader
    HERE = os.path.expanduser("~/sdn_project/controller")
sys.path.insert(0, HERE)
import routing_core as RC  # noqa: E402

PRIORITY = 10          # priority of the forwarding entries
COOKIE = 0xAB11        # tag so our flows are recognisable in dump-flows


class BaselineController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(BaselineController, self).__init__(*args, **kwargs)
        map_path = os.environ.get("ABILENE_MAP", os.path.join(HERE, "..", "topology", "topology_map.json"))
        self.metric = os.environ.get("BASELINE_METRIC", "delay")
        self.log_misses = os.environ.get("BASELINE_LOG_MISSES", "0") == "1"
        self.core = RC.load_topology(map_path, self.metric)
        self.datapaths = {}          # dpid -> datapath object of connected switches
        self.port_down = set()       # {(dpid, port_no)} ports reported down
        self.down_links = set()      # {link_id} links considered down
        self.installed = {}          # {(dpid, dst_mac): out_port} what we believe is installed
        self.miss_count = 0
        self.logger.info("Baseline controller ready: %d switches, %d hosts, %d links, metric=%s",
                         len(self.core.dpid_of), len(self.core.hosts), len(self.core.links), self.metric)

    # ------------------------------------------------------------------ extension point
    def get_weights(self):
        """None = use the metric (delay or hops). Phase D returns {link_id: predicted_cost}."""
        return None

    # ------------------------------------------------------------------ low-level helpers
    def _add_flow(self, dp, dst_mac, out_port):
        parser, ofp = dp.ofproto_parser, dp.ofproto
        match = parser.OFPMatch(eth_dst=dst_mac)
        actions = [parser.OFPActionOutput(out_port)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(parser.OFPFlowMod(datapath=dp, cookie=COOKIE, priority=PRIORITY,
                                      match=match, instructions=inst))

    def _delete_flow(self, dp, dst_mac):
        parser, ofp = dp.ofproto_parser, dp.ofproto
        dp.send_msg(parser.OFPFlowMod(datapath=dp, command=ofp.OFPFC_DELETE_STRICT,
                                      priority=PRIORITY, match=parser.OFPMatch(eth_dst=dst_mac),
                                      out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY))

    def _install_switch(self, dp):
        """Install this switch's complete table from the current network state."""
        for key in [k for k in self.installed if k[0] == dp.id]:
            del self.installed[key]
        flows = self.core.compute_flows(self.down_links, self.get_weights())
        n = 0
        for (dpid, mac), port in flows.items():
            if dpid == dp.id:
                self._add_flow(dp, mac, port)
                self.installed[(dpid, mac)] = port
                n += 1
        return n

    def recompute(self, reason):
        """Recompute all routes and update ONLY the entries that changed."""
        t0 = time.time()
        flows = self.core.compute_flows(self.down_links, self.get_weights())
        changed = 0
        for key, port in flows.items():
            dp = self.datapaths.get(key[0])
            if dp is None:
                continue
            if self.installed.get(key) != port:
                self._add_flow(dp, key[1], port)
                self.installed[key] = port
                changed += 1
        for key in list(self.installed):
            if key not in flows:
                dp = self.datapaths.get(key[0])
                if dp is not None:
                    self._delete_flow(dp, key[1])
                del self.installed[key]
                changed += 1
        self.logger.info("ROUTES RECOMPUTED (%s): %d flow entries changed in %.1f ms",
                         reason, changed, (time.time() - t0) * 1000.0)
        return changed

    def _port_state(self, dp, port_no, state, config):
        ofp = dp.ofproto
        link_id = self.core.port_to_link.get((dp.id, port_no))
        if link_id is None:
            return                                   # host port or switch-internal port
        is_down = bool(state & ofp.OFPPS_LINK_DOWN) or bool(config & ofp.OFPPC_PORT_DOWN)
        key = (dp.id, port_no)
        if is_down == (key in self.port_down):
            return
        if is_down:
            self.port_down.add(key)
        else:
            self.port_down.discard(key)
        new_down = set(self.core.port_to_link[k] for k in self.port_down)
        if new_down == self.down_links:
            return
        for lid in sorted(new_down - self.down_links):
            self.logger.info("LINK DOWN: %s  (%s - %s)", lid, self.core.links[lid]["src_switch"], self.core.links[lid]["dst_switch"])
        for lid in sorted(self.down_links - new_down):
            self.logger.info("LINK UP:   %s  (%s - %s)", lid, self.core.links[lid]["src_switch"], self.core.links[lid]["dst_switch"])
        self.down_links = new_down
        self.recompute("link state change")

    # ------------------------------------------------------------------ OpenFlow events
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def on_switch_features(self, ev):
        dp = ev.msg.datapath
        ofp, parser = dp.ofproto, dp.ofproto_parser
        self.datapaths[dp.id] = dp
        # start from a clean table
        dp.send_msg(parser.OFPFlowMod(datapath=dp, table_id=ofp.OFPTT_ALL, command=ofp.OFPFC_DELETE,
                                      out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY,
                                      match=parser.OFPMatch()))
        if self.log_misses:                           # debugging: send unmatched packets to us
            actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
            inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
            dp.send_msg(parser.OFPFlowMod(datapath=dp, priority=0, match=parser.OFPMatch(), instructions=inst))
        n = self._install_switch(dp)
        name = self.core.name_of.get(dp.id, "?")
        self.logger.info("Switch %s (dpid %s) connected: %d forwarding entries installed", name, dp.id, n)
        if len(self.datapaths) == len(self.core.dpid_of):
            self.logger.info("ALL %d SWITCHES CONNECTED - baseline routing is ready", len(self.datapaths))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def on_state_change(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            # ask for the current port states, in case a link was already down
            dp.send_msg(dp.ofproto_parser.OFPPortDescStatsRequest(dp, 0))
        elif ev.state == DEAD_DISPATCHER and dp.id in self.datapaths:
            self.logger.info("Switch %s disconnected", self.core.name_of.get(dp.id, dp.id))
            del self.datapaths[dp.id]
            for key in [k for k in self.installed if k[0] == dp.id]:
                del self.installed[key]

    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def on_port_desc(self, ev):
        dp = ev.msg.datapath
        for p in ev.msg.body:
            self._port_state(dp, p.port_no, p.state, p.config)

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def on_port_status(self, ev):
        msg = ev.msg
        self._port_state(msg.datapath, msg.desc.port_no, msg.desc.state, msg.desc.config)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def on_packet_in(self, ev):
        if not self.log_misses or self.miss_count >= 20:
            return
        self.miss_count += 1
        msg = ev.msg
        eth = packet.Packet(msg.data).get_protocols(ethernet.ethernet)
        if eth:
            self.logger.info("UNMATCHED packet on %s port %s: %s -> %s type 0x%04x",
                             self.core.name_of.get(msg.datapath.id, msg.datapath.id),
                             msg.match["in_port"], eth[0].src, eth[0].dst, eth[0].ethertype)
