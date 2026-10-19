### Scenario: throughput

| Policy | Mean RTT (ms) | p95 RTT (ms) | Max RTT (ms) | Control pair (ms) | Loss (%) | Path changes | Throughput (Mbps) | Reroute outage (ms) |
|---|---|---|---|---|---|---|---|---|
| static | 34.7 | 49.2 | 50.0 | 13.4 | 0.0 | 0 | 53.1 | - |
| reactive | 32.9 | 46.6 | 53.7 | 12.9 | 0.0 | 0 | 54.0 | - |
| forecast | 33.1 | 47.0 | 48.1 | 13.3 | 0.0 | 0 | 56.4 | - |

| Comparison | Difference in mean RTT (ms) | 95% interval | Verdict |
|---|---|---|---|
| forecast - static | -1.57 | [-1.82, -1.07] | forecast is clearly lower |
| forecast - reactive | +0.25 | [-0.15, +0.66] | forecast is not clearly different |

