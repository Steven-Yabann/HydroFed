# SKAB comparison experiments

`scripts/run_comparison.py` reruns the comparisons reported in the
dissertation. Every experiment uses the same SKAB plant partitions, per-plant
scaling, and recording-baseline alert policy.

## Development and final recordings

Within each plant, recordings alternate between two roles in partition order:
the first is a development recording, the second a final recording, and so
on. Each role has 17 fault recordings. The anomaly-free recording trains but
is never tested, as on the SKAB leaderboard.

Settings are chosen on development recordings. Final recordings are only for
reporting, so do not tune anything on them. The hybrid policy (latent size 4,
EWMA 0.2, median + 24 MAD on the last 25% of the baseline, alert on the first
sequence over threshold) was frozen on development recordings before final
recordings were scored.

About 54% of final test sequences are anomalous, so an always-on alarm scores
F1 ≈ 0.70. Report recall, false alarm rate and within-recording AUC rather
than F1 alone.

## Federated, centralised and isolated training

```bash
.venv/bin/python scripts/run_comparison.py regimes --jobs 3
```

This trains the same LSTM autoencoder three ways, with the same inputs, the
same 20 rounds × 5 local epochs, and the same alert policy:

- **federated**: FedAvg across the five plants
- **centralised**: every plant's baseline windows pooled on one server
- **isolated**: each plant trains alone

Federated training runs in-process with the same example-weighted FedAvg as
Flower's strategy, because a Ray simulation per seed is slow. Results are saved
to `artifacts/comparison/regimes.json`. A run takes about 15 minutes on an
Apple M-series CPU with `--jobs 3`.

### Detection, final recordings (mean ± sd over seeds 42–44)

| Regime | Score | Recall | False alarm rate | Within-recording AUC | Faults detected | Median delay |
|---|---|---|---|---|---|---|
| Federated | hybrid | 0.682 ± 0.008 | 0.235 ± 0.007 | 0.845 ± 0.002 | 16.0/17 | 28 s |
| Centralised | hybrid | 0.721 ± 0.002 | 0.282 ± 0.007 | 0.845 ± 0.000 | 16.3/17 | 22 s |
| Isolated | hybrid | 0.713 ± 0.009 | 0.301 ± 0.010 | 0.846 ± 0.001 | 16.7/17 | 22 s |
| Federated | residual | 0.429 ± 0.011 | 0.018 ± 0.008 | 0.803 ± 0.020 | 11.3/17 | 42 s |
| Centralised | residual | 0.454 ± 0.017 | 0.063 ± 0.018 | 0.841 ± 0.003 | 12.3/17 | 37 s |
| Isolated | residual | 0.594 ± 0.024 | 0.082 ± 0.022 | 0.864 ± 0.007 | 14.7/17 | 23 s |

The hybrid detector performs about the same whichever way the model was
trained: AUC is the same, so recall and false alarm differences are different
points on one trade-off. Its sensor-level distance does not depend on the
model at all. The residual-only score isolates the model. Federation costs
about 0.04 AUC against centralised training, and isolated plants do best,
which suggests SKAB's plants are not data-starved enough for pooling to help.

On development recordings the same ordering holds for residual AUC
(isolated 0.923, centralised 0.897, federated 0.873).

### Communication

| Federated rounds | Total | Per plant | × SKAB CSVs (4.39 MB) | × binary rows (1.50 MB) | Break-even at 1 Hz |
|---|---|---|---|---|---|
| 5 | 20.84 MB | 4.17 MB | 5× | 14× | 1.5 days |
| 10 | 41.68 MB | 8.34 MB | 9× | 28× | 3.0 days |
| 20 | 83.36 MB | 16.67 MB | 19× | 56× | 6.0 days |

The LSTM autoencoder has 104,204 parameters (416,816 bytes). Each round, five
plants upload their weights and download the global model. The 20-round total
matches the uplink and downlink bytes recorded by `TrackingFedAvg` in a Flower
run. Byte counts cover the weight arrays only, not gRPC framing.

SKAB holds only about 13 hours of data, so federated training moves more
bytes than sending the raw recordings to a server. A plant streaming 8 sensors
at 1 Hz as float32 produces 2.76 MB a day. One 20-round training run costs
each plant as much as six days of streaming, so federated learning moves less
data whenever plants retrain less often than that. Higher sampling rates, such
as vibration bursts, push the break-even point earlier. The break-even figure
assumes these rates rather than measuring a real plant.

Fewer rounds cost little detection on development recordings (hybrid recall
0.77, 0.82 and 0.82, and AUC 0.903, 0.910 and 0.920, for 5, 10 and 20 rounds).

### Limits

- Three seeds and 17 final fault recordings: differences under about 0.02
  are within run-to-run noise.
- Plants share fault types and sensors. The proposal's non-IID partition
  (different noise and sampling rates) is not implemented, so this does not
  test heterogeneous plants.
- Every plant has thousands of baseline windows. A data-scarcity experiment
  (shorter baselines) is needed before claiming federation helps small plants.
