#!/usr/bin/env python3
"""
features.py   (Phase B - shared module, used by train_model.py AND predictor.py)

WHY THIS FILE EXISTS
  The ML model must receive EXACTLY the same kind of numbers while training
  (offline, from the CSV) and while predicting (live, inside the Ryu controller).
  If the two computations differ even slightly, the model gives wrong answers
  without any error message ("train/serve skew"). So the feature recipe lives
  in ONE place, and both programs call it.

WHAT A "WINDOW" IS
  The model looks at the last WINDOW = 6 utilization values of one link
  (6 x 5 minutes = 30 minutes of history), oldest first, newest last.

THE FEATURES (all computed for one link at one moment "now")
  util_now        utilization now (%)
  util_lag1       utilization 5 minutes ago
  util_lag2       utilization 10 minutes ago
  util_lag5       utilization 25 minutes ago
  diff1           util_now - util_lag1             (is it rising or falling?)
  slope3          (util_now - util_lag2) / 2       (trend over 10 minutes)
  mean3, mean6    average of the last 3 / 6 values
  std6            standard deviation of the last 6 values (how jumpy)
  max6, min6      highest / lowest of the last 6 values
  bandwidth_mbps  REAL link capacity (from the dataset, e.g. 9920 or 2480)
  delay_ms        REAL propagation delay of the link
  load_now_mbps   util_now x bandwidth / 100       (traffic in Mbps)
  hour_sin, hour_cos   time of day as a point on a circle (optional)
"""
import math
from datetime import datetime

WINDOW = 6            # how many most-recent values the features use
STEP_MINUTES = 5      # the dataset has one snapshot every 5 minutes

BASE_FEATURES = [
    "util_now", "util_lag1", "util_lag2", "util_lag5",
    "diff1", "slope3", "mean3", "mean6", "std6", "max6", "min6",
    "bandwidth_mbps", "delay_ms", "load_now_mbps",
]
TIME_FEATURES = ["hour_sin", "hour_cos"]


def feature_names(use_time=True):
    """Ordered list of feature names. The model is trained in exactly this order."""
    return BASE_FEATURES + (TIME_FEATURES if use_time else [])


def parse_ts(ts):
    """'20040408-1220' -> datetime(2004, 4, 8, 12, 20)"""
    return datetime.strptime(ts, "%Y%m%d-%H%M")


def minute_of_day_from_ts(ts):
    d = parse_ts(ts)
    return d.hour * 60 + d.minute


def row_features(window, bandwidth_mbps, delay_ms, minute_of_day=None, use_time=True):
    """Compute the feature dictionary for ONE link at ONE moment.

    window         list/array of utilization (%) values, OLDEST first, NEWEST last.
                   Only the last WINDOW values are used.
    bandwidth_mbps REAL capacity of the link (not the scaled Mininet value!)
    delay_ms       REAL propagation delay of the link (not scaled)
    minute_of_day  0..1439, needed only if use_time is True
    """
    if len(window) < WINDOW:
        raise ValueError("need at least %d history values, got %d" % (WINDOW, len(window)))
    w = [float(x) for x in list(window)[-WINDOW:]]
    now = w[-1]
    mean6 = sum(w) / float(WINDOW)
    f = {
        "util_now": now,
        "util_lag1": w[-2],
        "util_lag2": w[-3],
        "util_lag5": w[0],
        "diff1": now - w[-2],
        "slope3": (now - w[-3]) / 2.0,
        "mean3": sum(w[-3:]) / 3.0,
        "mean6": mean6,
        "std6": math.sqrt(sum((x - mean6) ** 2 for x in w) / float(WINDOW)),
        "max6": max(w),
        "min6": min(w),
        "bandwidth_mbps": float(bandwidth_mbps),
        "delay_ms": float(delay_ms),
        "load_now_mbps": now * float(bandwidth_mbps) / 100.0,
    }
    if use_time:
        if minute_of_day is None:
            raise ValueError("minute_of_day is required when use_time=True")
        angle = 2.0 * math.pi * float(minute_of_day) / 1440.0
        f["hour_sin"] = math.sin(angle)
        f["hour_cos"] = math.cos(angle)
    return f


def to_vector(f, names):
    """Turn the feature dictionary into a plain list in the model's column order."""
    return [f[n] for n in names]
