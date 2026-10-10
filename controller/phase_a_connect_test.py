"""
phase_a_connect_test.py   (Phase A - Step A6, optional)

A tiny Ryu application. It does NOT forward any traffic. It only prints a line
every time one of your 12 Mininet switches connects to the controller, so you
can prove that Mininet <-> Ryu <-> OpenFlow 1.3 works for the Abilene network.

RUN (terminal 1, no sudo):
    ryu-manager ~/sdn_project/controller/phase_a_connect_test.py

THEN (terminal 2):
    cd ~/sdn_project/topology
    sudo python3 abilene_topo.py --mode controller

EXPECTED in terminal 1: twelve lines like
    Switch connected: dpid=1  (1/12 seen)
    ...
    Switch connected: dpid=12 (12/12 seen)   <- all switches connected
"""
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3


class ConnectTest(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ConnectTest, self).__init__(*args, **kwargs)
        self.seen = set()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def on_switch_features(self, ev):
        dp = ev.msg.datapath
        self.seen.add(dp.id)
        self.logger.info("Switch connected: dpid=%s  (%d/12 seen)", dp.id, len(self.seen))
        if len(self.seen) == 12:
            self.logger.info("ALL 12 SWITCHES CONNECTED - Phase A controller link works")
