Phase D settings: ML_K=12 ML_START=20040408-1650 ML_STEPS=40 ML_COST_MODEL=quadratic
Live policy rankings align consistently with offline evaluation results.

### Phase D: Live Evaluation Summary & Findings
- **Phase D Settings**: 
  - `ML_START = 20040408-1650`
  - `ML_STEPS = 40`
  - `ML_K = 12`
  - `ML_COST_MODEL = quadratic`
- **Policies Performance (Overall Mean & p95 RTT)**:
  - **Static Policy**: Mean RTT = `47.94 ms`, p95 RTT = `158.11 ms`
  - **Reactive Policy**: Mean RTT = `34.22 ms`, p95 RTT = `71.36 ms`
  - **Forecast Policy**: Mean RTT = `34.72 ms`, p95 RTT = `72.82 ms`
- **Key Finding**: Dynamic routing policies (forecast and reactive) successfully mitigate heavy traffic surges and drastically lower latency and packet delay variations compared to the static baseline.
Phase E settings: ML_K=6 ML_COST_MODEL=quadratic window1 ML_START=20040408-0220 window2 ML_START=20040408-1305
Phase E failure scenario: link ATLAng_HSTNng down at step 10, up at step 25, outage pair h12-h8
Phase E live runs planned: throughput (30 x 14 s), failure (40 x 10 s), window2 (40 x 8 s), policies static/reactive/forecast
