#!/usr/bin/env python3
"""
predictor.py   (Phase B - Step B8; imported by the Ryu controller in Phase D)

WHAT IT DOES
  Wraps the trained model so a controller can use it in three lines:

      from predictor import UtilizationPredictor
      pred = UtilizationPredictor()                 # loads ml/model.pkl and the topology map
      pred.update_all({"HSTNng_LOSAng": 3.2, ...}, "20040408-1200")   # newest utilization per link
      future = pred.predict_all()                   # {"HSTNng_LOSAng": 4.1, ...}  (%, HORIZON ahead)
      cost = pred.link_cost(delay_ms, future["HSTNng_LOSAng"])

  It remembers the last 6 values (30 minutes) of every link, builds the features
  with features.py (same code as training) and asks the model for the forecast.

IMPORTANT
  The model was trained with the REAL Abilene bandwidth and delay (9920 Mbps etc.).
  So this class takes bandwidth_real_mbps and delay_real_ms from topology_map.json,
  NOT the scaled Mininet values. Mixing them up would silently ruin predictions.

CLI DEMO (from ~/sdn_project/ml)
  python3 predictor.py --at 20040408-1200
      -> shows, for every link, the utilization now, the forecast, what really
         happened later (if the CSV has it) and the routing cost derived from it.
"""
import argparse
import sys
from collections import deque
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as F  # noqa: E402


class UtilizationPredictor(object):
    def __init__(self, model_path=None, map_path=None):
        model_path = Path(model_path) if model_path else HERE / "model.pkl"
        map_path = Path(map_path) if map_path else HERE.parent / "topology" / "topology_map.json"
        if not model_path.exists():
            raise FileNotFoundError("%s not found. Train first: python3 train_model.py --horizon 3 --make-default" % model_path)
        if not map_path.exists():
            raise FileNotFoundError("%s not found. Run Phase A step A3 (make_topology_map.py)." % map_path)
        import json
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.names = bundle["feature_names"]
        self.use_time = bundle["use_time_features"]
        self.horizon = bundle["horizon"]
        self.horizon_minutes = bundle["horizon_minutes"]
        self.model_name = bundle["model_name"]
        self.target_mode = bundle.get("target_mode", "level")
        self.idx_now = self.names.index("util_now")
        with open(map_path) as f:
            topo = json.load(f)
        # REAL values, as used during training
        self.links = {l["link_id"]: {"bw": float(l["bandwidth_real_mbps"]),
                                     "delay_real": float(l["delay_real_ms"]),
                                     "delay_emulated": float(l["delay_ms"])}
                      for l in topo["links"]}
        self.history = {lid: deque(maxlen=F.WINDOW) for lid in self.links}
        self.minute_of_day = None

    # ---------------------------------------------------------------- feeding
    def update(self, link_id, util_pct, timestamp=None):
        """Add the newest utilization (%) of one link. timestamp like '20040408-1200'."""
        self.history[link_id].append(float(util_pct))
        if timestamp is not None:
            self.minute_of_day = F.minute_of_day_from_ts(timestamp)

    def update_all(self, util_by_link, timestamp=None):
        for lid, u in util_by_link.items():
            if lid in self.history:
                self.history[lid].append(float(u))
        if timestamp is not None:
            self.minute_of_day = F.minute_of_day_from_ts(timestamp)

    def ready(self):
        """True when every link has at least WINDOW values."""
        return all(len(h) >= F.WINDOW for h in self.history.values())

    # ---------------------------------------------------------------- predicting
    def predict_all(self):
        """Forecast (%) for every link, `horizon` steps ahead. Links with too little
        history simply get their latest value (persistence)."""
        out, rows, ids = {}, [], []
        for lid, h in self.history.items():
            if len(h) >= F.WINDOW:
                info = self.links[lid]
                feat = F.row_features(list(h), info["bw"], info["delay_real"],
                                      minute_of_day=self.minute_of_day, use_time=self.use_time)
                rows.append(F.to_vector(feat, self.names))
                ids.append(lid)
            else:
                out[lid] = float(h[-1]) if len(h) else 0.0
        if rows:
            X = np.asarray(rows, dtype=float)
            preds = self.model.predict(X)
            if self.target_mode == "delta":            # model predicts the change; add "now" back
                preds = preds + X[:, self.idx_now]
            for lid, p in zip(ids, preds):
                out[lid] = max(0.0, float(p))          # utilization cannot be negative
        return out

    def predict_link(self, link_id):
        return self.predict_all()[link_id]

    # ---------------------------------------------------------------- cost
    @staticmethod
    def link_cost(delay_ms, predicted_util_pct, k=3.0, cap_pct=150.0):
        """Routing cost of a link = delay x (1 + k x u^2), u = predicted utilization / 100.
        A lightly loaded link costs about its delay. A link predicted to be overloaded
        becomes several times more expensive, so Dijkstra steers traffic away from it.
        k and the shape are STARTING VALUES; they are tuned in Phase D."""
        u = min(max(predicted_util_pct, 0.0), cap_pct) / 100.0
        return float(delay_ms) * (1.0 + k * u * u)


# ---------------------------------------------------------------------------- demo
def demo():
    import pandas as pd
    ap = argparse.ArgumentParser(description="Show forecasts for every link at one moment")
    ap.add_argument("--at", required=True, help="moment 'now', like 20040408-1200")
    ap.add_argument("--csv", default=str(HERE.parent / "data" / "link_timeseries.csv"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--map", default=None)
    ap.add_argument("--k", type=float, default=3.0)
    args = ap.parse_args()

    pred = UtilizationPredictor(args.model, args.map)
    df = pd.read_csv(args.csv)
    stamps = sorted(df["timestamp"].unique())
    upto = [s for s in stamps if s <= args.at]
    if len(upto) < F.WINDOW:
        sys.exit("ERROR: need at least %d timestamps up to %s in the CSV" % (F.WINDOW, args.at))
    piv = df.pivot(index="timestamp", columns="link_id", values="utilization_pct")
    for s in upto[-F.WINDOW:]:
        pred.update_all(piv.loc[s].to_dict(), s)
    now_stamp = upto[-1]
    future = pred.predict_all()

    target_stamp = (F.parse_ts(now_stamp) + timedelta(minutes=pred.horizon_minutes)).strftime("%Y%m%d-%H%M")
    have_actual = target_stamp in piv.index
    print("model: %s | horizon: %d min | now: %s | forecast for: %s"
          % (pred.model_name, pred.horizon_minutes, now_stamp, target_stamp))
    print("%-16s %10s %10s %10s %9s %9s %9s" % ("link_id", "util_now", "forecast", "actual", "abs_err", "delay_ms", "cost"))
    errs = []
    for lid in sorted(pred.links, key=lambda x: -future[x]):
        now_u = piv.loc[now_stamp, lid]
        act = piv.loc[target_stamp, lid] if have_actual else float("nan")
        err = abs(future[lid] - act) if have_actual else float("nan")
        if have_actual:
            errs.append(err)
        cost = pred.link_cost(pred.links[lid]["delay_emulated"], future[lid], k=args.k)
        print("%-16s %10.2f %10.2f %10.2f %9.2f %9.3f %9.3f"
              % (lid, now_u, future[lid], act, err, pred.links[lid]["delay_emulated"], cost))
    if errs:
        print("\nmean absolute error over the 15 links: %.3f %%-points" % (sum(errs) / len(errs)))
    else:
        print("\n(no actual value in the CSV for %s, so no error is shown)" % target_stamp)
    print("[PASS] predictor works: this exact class is what the Ryu controller will import in Phase D")


if __name__ == "__main__":
    demo()
