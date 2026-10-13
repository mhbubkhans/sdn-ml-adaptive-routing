"""
ml_controller.py   (Phase D - the ML-driven Ryu controller)

WHAT IT DOES
  Same controller as Phase C (it IS the Phase C controller, extended), but the
  link costs are no longer fixed. A background "replay" loop walks through the
  test period of link_timeseries.csv, one 5-minute data step every ML_STEP_SECONDS
  real seconds. At every step it
    1. feeds the newest utilization of every link to the Phase B predictor,
    2. obtains the forecast of every link (HORIZON minutes ahead),
    3. converts utilization into a link cost,
    4. recomputes all routes with these costs and updates only the changed rules,
    5. writes live_state.json, which the load emulator uses to make the Mininet
       links behave like the replayed load (slower and with less capacity).

THREE POLICIES (choose with ML_POLICY, so ONE program gives the whole comparison)
    static     costs ignore the load (identical to the Phase C baseline)
    reactive   costs use the CURRENT utilization (load aware, but no ML)
    forecast   costs use the ML FORECAST of utilization   (default, the project idea)

RUN   (no sudo)
  ML_POLICY=forecast ML_K=6 ML_START=20040408-0735 ML_STEPS=40 \
      ryu-manager ~/sdn_project/controller/ml_controller.py

SETTINGS (environment variables)
  ML_POLICY        static | reactive | forecast            (default forecast)
  ML_COST_MODEL    quadratic | queue                       (default quadratic)
  ML_K             k of the quadratic cost                 (default 6)
  ML_START         first replayed moment, e.g. 20040408-0735  (default: just after the test start)
  ML_STEPS         number of data steps to replay          (default 40)
  ML_STEP_SECONDS  real seconds per data step              (default 8)
  ML_START_DELAY   seconds to wait after all switches connect (default 10)
  ML_WATCH         pairs whose path is reported, e.g. h2-h6,h1-h10   (default h2-h6,h1-h10,h12-h8,h3-h9)
  ML_MODEL, ML_CSV, ML_STATE, ABILENE_MAP   file locations (defaults are the project folders)
  (BASELINE_METRIC and BASELINE_LOG_MISSES of the Phase C controller also work.)
"""
import json
import os
import sys
import time
from collections import deque

from ryu.lib import hub

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.path.expanduser("~/sdn_project/controller")
ML_DIR = os.path.abspath(os.path.join(HERE, "..", "ml"))
sys.path.insert(0, HERE)
sys.path.insert(0, ML_DIR)

import joblib  # noqa: E402
import features as F  # noqa: E402
import load_model as LM  # noqa: E402
from baseline_controller import BaselineController  # noqa: E402
from predictor import UtilizationPredictor  # noqa: E402

DEFAULT_WATCH = "h2-h6,h1-h10,h12-h8,h3-h9"


class MLController(BaselineController):

    def __init__(self, *args, **kwargs):
        super(MLController, self).__init__(*args, **kwargs)
        env = os.environ.get
        self.policy = env("ML_POLICY", "forecast")
        if self.policy not in ("static", "reactive", "forecast"):
            raise ValueError("ML_POLICY must be static, reactive or forecast")
        self.cost_model = env("ML_COST_MODEL", "quadratic")
        self.k = float(env("ML_K", "6"))
        self.steps = int(env("ML_STEPS", "40"))
        self.step_seconds = float(env("ML_STEP_SECONDS", "8"))
        self.start_delay = float(env("ML_START_DELAY", "10"))
        self.state_path = env("ML_STATE", os.path.join(HERE, "..", "results", "live_state.json"))
        csv_path = env("ML_CSV", os.path.join(HERE, "..", "data", "link_timeseries.csv"))
        model_path = env("ML_MODEL", os.path.join(ML_DIR, "model.pkl"))
        map_path = env("ABILENE_MAP", os.path.join(HERE, "..", "topology", "topology_map.json"))
        self.watch = [tuple(x.split("-")) for x in env("ML_WATCH", DEFAULT_WATCH).split(",") if x]

        self.timeline, self.util = LM.load_timeline(csv_path)
        self.pred = UtilizationPredictor(model_path, map_path)
        bundle = joblib.load(model_path)
        self.horizon = bundle["horizon"]
        self.horizon_minutes = bundle["horizon_minutes"]
        test_start = LM.pretty_ts_to_compact(bundle["trained_until"])
        start = env("ML_START") or LM.add_steps(test_start, F.WINDOW)
        if start not in self.util:
            raise ValueError("ML_START=%s is not in the CSV timeline" % start)
        self.start_idx = self.timeline.index(start)
        if self.start_idx < F.WINDOW:
            raise ValueError("ML_START too early: need %d earlier steps of history" % F.WINDOW)
        if self.start_idx + self.steps > len(self.timeline):
            self.steps = len(self.timeline) - self.start_idx
        if start < test_start:
            self.logger.warning("ML_START is BEFORE the test period (%s): the model trained on these data, "
                                "forecasts will look better than they really are", test_start)

        self.current_weights = None
        self.rules_total = 0
        self.path_changes = 0
        self.mae_sum = self.mae_pers_sum = 0.0
        self.mae_n = 0
        self.last_paths = {}
        self.pending = deque()
        self.step_now = -1
        self.finished = False
        self.logger.info("ML controller: policy=%s cost=%s k=%g | forecast horizon %d min | replay %s, %d steps x %.0f s | model %s",
                         self.policy, self.cost_model, self.k, self.horizon_minutes, start, self.steps,
                         self.step_seconds, os.path.basename(model_path))
        hub.spawn(self._run)

    # ---------------------------------------------------------------- extension point of Phase C
    def get_weights(self):
        return self.current_weights

    # ---------------------------------------------------------------- replay
    def _run(self):
        try:
            n = len(self.core.dpid_of)
            while len(self.datapaths) < n:
                hub.sleep(1)
            self.logger.info("All %d switches connected. Replay starts in %.0f s (start the experiment now) ...", n, self.start_delay)
            hub.sleep(self.start_delay)
            self._replay()
        except Exception:
            self.logger.exception("replay loop crashed")

    def _weights_for(self, util_now, forecast):
        if self.policy == "static":
            return None
        src = util_now if self.policy == "reactive" else forecast
        return {lid: LM.link_cost(l["delay_ms"], src[lid], self.cost_model, self.k) for lid, l in self.core.links.items()}

    def _watched_paths(self):
        out = {}
        for a, b in self.watch:
            p = self.core.path(a, b, self.down_links, self.current_weights)
            out["%s-%s" % (a, b)] = self.core.describe(p) if p else "UNREACHABLE"
        return out

    def _write_state(self, util_now, forecast, rules_step, paths):
        state = {
            "policy": self.policy, "cost_model": self.cost_model, "k": self.k,
            "horizon_minutes": self.horizon_minutes, "step": self.step_now, "steps": self.steps,
            "step_seconds": self.step_seconds,
            "timestamp": self.timeline[self.start_idx + max(self.step_now, 0)],
            "finished": self.finished, "written_at": time.time(),
            "util": util_now, "forecast": forecast, "weights": self.current_weights,
            "paths": paths, "rules_changed_step": rules_step, "rules_changed_total": self.rules_total,
            "path_changes_total": self.path_changes,
            "live_mae": (self.mae_sum / self.mae_n) if self.mae_n else None,
            "live_mae_persistence": (self.mae_pers_sum / self.mae_n) if self.mae_n else None,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, self.state_path)          # atomic: readers never see half a file

    def _replay(self):
        i0 = self.start_idx
        for j in range(i0 - (F.WINDOW - 1), i0):                       # history before the first step
            self.pred.update_all(self.util[self.timeline[j]], self.timeline[j])
        hot = max(self.core.links, key=lambda l: max(self.util[t][l] for t in self.timeline[i0:i0 + self.steps]))
        for step in range(self.steps):
            self.step_now = step
            idx = i0 + step
            ts = self.timeline[idx]
            now = self.util[ts]
            self.pred.update_all(now, ts)
            forecast = self.pred.predict_all()

            # live accuracy: compare forecasts made `horizon` steps ago with what really happened
            while self.pending and self.pending[0][0] <= idx:
                due, fc_old, util_old = self.pending.popleft()
                if due == idx:
                    errs = [abs(fc_old[l] - now[l]) for l in now]
                    pers = [abs(util_old[l] - now[l]) for l in now]
                    self.mae_sum += sum(errs) / len(errs)
                    self.mae_pers_sum += sum(pers) / len(pers)
                    self.mae_n += 1
            self.pending.append((idx + self.horizon, forecast, now))

            self.current_weights = self._weights_for(now, forecast)
            rules = self.recompute("step %d/%d %s policy=%s" % (step + 1, self.steps, ts, self.policy)) if self.policy != "static" else 0
            self.rules_total += rules
            paths = self._watched_paths()
            changed = [p for p in paths if self.last_paths.get(p) not in (None, paths[p])]
            self.path_changes += len(changed)
            self.last_paths = paths
            self._write_state(now, forecast, rules, paths)
            self.logger.info("STEP %2d/%d %s | %s util %5.1f%% forecast %5.1f%% | rules changed %2d%s",
                             step + 1, self.steps, ts, hot, now[hot], forecast[hot], rules,
                             "".join(" | PATH CHANGE %s: %s" % (p, paths[p]) for p in changed))
            hub.sleep(self.step_seconds)

        self.finished = True
        self._write_state(self.util[self.timeline[i0 + self.steps - 1]], {}, 0, self.last_paths)
        self.logger.info("REPLAY FINISHED | policy=%s | total rules changed %d | watched path changes %d | live forecast MAE %.2f vs persistence MAE %.2f (%%-points)",
                         self.policy, self.rules_total, self.path_changes,
                         self.mae_sum / max(1, self.mae_n), self.mae_pers_sum / max(1, self.mae_n))
