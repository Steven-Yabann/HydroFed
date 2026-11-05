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

## Heterogeneous and data-scarce plants

```bash
.venv/bin/python scripts/run_comparison.py scenarios --jobs 3
```

The five simulated plants are slices of one test rig, so the regime
comparison above cannot show the two things federated learning is meant to
help with: plants that differ, and plants with little data. This command
reruns it under three conditions, with the frozen alert policy and no new
tuning. Results go to `artifacts/comparison/scenarios.json` and the run takes
about 45 minutes with `--jobs 3`.

- **baseline**: the data unchanged, for reference.
- **heterogeneous**: each plant records the same physical signals differently
  (`hydrofed.experiments.scenarios.HETEROGENEOUS_PROFILES`):

  | Plant | Profile |
  |---|---|
  | plant_1 | the test rig as recorded |
  | plant_2 | extra sensor noise, 0.5 baseline sd |
  | plant_3 | logs every 5 seconds |
  | plant_4 | logs every 2 seconds, gain 1.2 and offset +1 sd, noise 0.25 sd |
  | plant_5 | sensor resolution of 0.5 sd, gain 0.8 and offset −1 sd |

  Each plant's own scaler cancels gains and offsets, which a test confirms,
  so the profiles that matter are noise, resolution and logging interval.
  Delays are converted to seconds for slower loggers.
- **scarce**: plant_4 and plant_5 train their models on only the first 100,
  50 or 25 seconds of each recording, instead of about 390. Every recording
  keeps its full baseline as the alert reference and the test windows do not
  change, so only the training data shrinks. At 25 seconds plant_4 has 144
  training windows instead of 3,519.

### Results, final recordings (mean over seeds 42–44)

Within-recording AUC. "Model only" is the residual score; "No model" is the
sensor-level distance from each recording's baseline, which needs no training.

| Scenario | Plants | Federated, full | Centralised, full | Isolated, full | Federated, model only | Centralised, model only | Isolated, model only | No model |
|---|---|---|---|---|---|---|---|---|
| baseline | all | 0.845 | 0.845 | 0.846 | 0.803 | 0.841 | 0.864 | 0.844 |
| baseline | plant_4, plant_5 | 0.885 | 0.885 | 0.895 | 0.904 | 0.916 | 0.931 | 0.882 |
| scarce, 100 s | plant_4, plant_5 | 0.879 | 0.881 | 0.881 | 0.877 | 0.877 | 0.902 | 0.882 |
| scarce, 50 s | plant_4, plant_5 | 0.881 | 0.879 | 0.881 | 0.855 | 0.870 | 0.878 | 0.882 |
| scarce, 25 s | plant_4, plant_5 | 0.877 | 0.881 | 0.879 | 0.845 | 0.857 | 0.888 | 0.882 |
| heterogeneous | all | 0.753 | 0.760 | 0.760 | 0.753 | 0.768 | 0.802 | 0.753 |

Model-only AUC per plant under heterogeneity:

| Regime | plant_1 | plant_2 (noisy) | plant_3 (5 s) | plant_4 (2 s) | plant_5 (coarse) |
|---|---|---|---|---|---|
| Federated | 0.964 | 0.550 | 0.637 | 0.870 | 0.903 |
| Centralised | 0.978 | 0.592 | 0.594 | 0.915 | 0.918 |
| Isolated | 0.999 | 0.611 | 0.720 | 0.904 | 0.923 |
| No model | 0.994 | 0.535 | 0.786 | 0.831 | 0.799 |

### What the scenarios show

- **Federated learning does not beat plants training alone, even with 25
  seconds of data.** On the scarce plants, isolated models keep the best
  model-only AUC at every level.
- **The trained model barely affects fault ranking.** The full detector's AUC
  equals the no-model AUC in every scenario. An isolated autoencoder trained
  on very little data reconstructs little, so its residual behaves like the
  baseline distance (0.888 against 0.882 at 25 seconds), and that distance is
  a strong detector on SKAB. A better-trained shared model reconstructs
  faults too, which lowers its residual AUC. More training data, the thing
  federated learning supplies, is therefore not what limits detection here.
- **Different plants hurt shared models.** Under heterogeneity, both shared
  models fit the 5-second plant worse than its own model does (federated
  0.637, centralised 0.594, isolated 0.720), the expected penalty of one
  model for unlike plants. Heavy noise (plant_2) defeats every detector.
- **Federated stays close to centralised** in every scenario: within 0.01 on
  the full detector, and within about 0.04 on the model-only score.

For the dissertation, these results support "federated learning matches
centralised training without moving raw data" but not "federated learning
improves detection". On SKAB the per-recording baseline does most of the
work, so a dataset or task where the model must learn shared structure, such
as long-term drift or faults that do not shift sensor levels, would be needed
to show a federated benefit.

### Limits

- Profiles are simulated transformations of one rig, not real plant
  differences, and their strengths were chosen by hand, not tuned.
- plant_1 has a single final fault recording, so its per-plant values rest on
  one fault.
- Scarce plants contribute two plants' final faults (8 recordings). Differences
  of a few hundredths are within run-to-run noise.

## LSTM against transformer and classic machine learning

```bash
pip install -e ".[experiments]"   # XGBoost
.venv/bin/python scripts/run_comparison.py models --jobs 3
```

The candidates fall into three categories:

| Category | Candidates | Learns from |
|---|---|---|
| Recurrent | LSTM autoencoder (latent 4) | normal baseline windows |
| Transformer | transformer autoencoder with a latent-4 bottleneck, d_model 32 or 64 | normal baseline windows |
| Classic | Random Forest, XGBoost (supervised); Isolation Forest (unsupervised) | labelled faults / normal baseline windows |
| Federated classic | XGBoost trained with Flower's bagging aggregation | labelled faults, kept at each plant |

The autoencoders train federated on the same schedule as above. Classic
models use summary statistics per sensor and window (mean, sd, min, max, last
value, change), standardised against each recording's own baseline, so every
model gets the same baseline information. Settings are fixed in the code, not
tuned.

Tree ensembles cannot be averaged with FedAvg. Federated XGBoost follows
Flower's `FedXgbBagging` strategy and reuses its aggregation function. Each
round, every plant loads the shared model and boosts one tree on its own
labelled windows. It uploads only that tree, and the server appends all the
new trees. It uses the same tree settings as centralised XGBoost, over 60
rounds. Only plants with labelled faults take part. plant_1's development
recording is the anomaly-free one, so four plants contribute 240 trees.

Selection uses development recordings only:

- **Model**: highest within-recording AUC on development recordings. For the
  autoencoders this uses the residual-only score, which is the model's own
  signal. Supervised models are scored out of fold, holding out whole
  recordings.
- **Threshold**: each model gets the predeclared rule, the highest recall with
  at most 15% false alarms on development recordings.

The classic models run in a separate process. PyTorch and XGBoost load
different OpenMP runtimes, and on macOS one crashes the other.

### Selection on development recordings

| Category | Model | Needs fault labels | Dev AUC | Selected |
|---|---|---|---|---|
| Recurrent | LSTM autoencoder | no | 0.873 | yes |
| Transformer | transformer, d_model 32 | no | 0.869 | yes |
| Transformer | transformer, d_model 64 | no | 0.863 | |
| Classic | Random Forest | yes | 0.923 | |
| Classic | XGBoost | yes | 0.924 | yes |
| Classic | Isolation Forest | no | 0.863 | |
| Federated classic | XGBoost, federated bagging | yes | 0.913 | yes |

Random Forest and XGBoost are effectively tied, and so are the two
transformers.

### Selected models on final recordings (mean ± sd over seeds 42–44)

| Model | Score | Recall | False alarm rate | Within-recording AUC | Faults detected | Median delay |
|---|---|---|---|---|---|---|
| LSTM autoencoder | hybrid | 0.682 ± 0.008 | 0.235 ± 0.007 | 0.845 ± 0.002 | 16.0/17 | 28 s |
| LSTM autoencoder | residual | 0.487 ± 0.015 | 0.084 ± 0.013 | 0.803 ± 0.020 | 13.7/17 | 36 s |
| Transformer (d_model 32) | hybrid | 0.638 ± 0.013 | 0.210 ± 0.022 | 0.845 ± 0.001 | 15.7/17 | 40 s |
| Transformer (d_model 32) | residual | 0.438 ± 0.028 | 0.096 ± 0.027 | 0.776 ± 0.020 | 11.3/17 | 45 s |
| XGBoost | trained on all fault types | 0.747 ± 0.004 | 0.137 ± 0.009 | 0.862 ± 0.002 | 16.7/17 | 7 s |
| XGBoost | fault type not seen in training | 0.624 ± 0.004 | 0.098 ± 0.004 | 0.813 ± 0.002 | 16.7/17 | 24 s |
| XGBoost, federated | trained on all fault types | 0.688 ± 0.011 | 0.118 ± 0.004 | 0.837 ± 0.005 | 17.0/17 | 9 s |
| XGBoost, federated | fault type not seen in training | 0.605 ± 0.012 | 0.084 ± 0.004 | 0.791 ± 0.006 | 17.0/17 | 37 s |

| Model | Training | Size | Communication for one training run |
|---|---|---|---|
| LSTM autoencoder | FedAvg, 20 rounds, 5 plants | 104,204 parameters | 83.4 MB (4.17 MB a round) |
| Transformer (d_model 32) | FedAvg, 20 rounds, 5 plants | 35,020 parameters | 28.0 MB (1.40 MB a round) |
| XGBoost, federated | bagging, 60 rounds, 4 plants | 240 trees (0.48 MB as JSON) | 56.1 MB: 0.55 MB up, 55.5 MB down |
| XGBoost | centralised | 300 trees | raw data sent to a server |

Federated XGBoost traffic is almost all downlink. Each round the server sends
the whole growing model to every plant, as Flower's bagging strategy does,
serialised as JSON.

The transformer does not beat the LSTM. Its residual AUC is lower (0.776
against 0.803), and inside the hybrid detector the two are tied. It is about
a third of the size, so it needs a third of the communication.

XGBoost is the strongest detector here: more recall at fewer false alarms,
and alerts within seconds. That advantage comes from labels. It learns
SKAB's faults from labelled development recordings, and the final recordings
contain the same fault types on the same test rig. When it is tested on a
fault type it never saw (training on `valve1` and `valve2` faults, testing on
`other`, and so on), recall drops to 0.62. It still keeps a lower false
alarm rate than the autoencoders.

What this means for HydroFed:

- The autoencoders need no fault labels. A new plant rarely has labelled
  faults, so they remain the default detector. Supervised models become an
  option once a fleet has collected labelled incidents.
- XGBoost can be federated with bagging. It loses about 0.025 AUC against
  centralised training, much as federation costs the LSTM's own residual
  score about 0.04. At about the LSTM detector's recall (0.69 against 0.68),
  it raises half the false alarms (11.8% against 23.5%). But it needs labelled
  faults at each plant, and plants without them add nothing.
- Isolation Forest, the unsupervised classic model, ranks faults about as well
  as the LSTM's residuals on development recordings (AUC 0.863 against
  0.873).

### Limits

- The leave-fault-type-out test uses SKAB's three folders. `other` mixes
  several fault mechanisms, so it is a coarse test of novelty.
- Training times in `artifacts/comparison/models.json` were measured with three
  jobs in parallel on a shared laptop, so treat them as relative only.
- Each model has one fixed configuration, plus two transformer sizes. A wider
  search on development recordings could move any of these numbers.
- Federated XGBoost shares trees, which encode split thresholds on each
  plant's features. That reveals more about local data than averaged neural
  network weights, and this comparison does not measure it.

## Showing the comparisons on the dashboard

The dashboard's Comparisons page (`#comparisons`) reads the latest completed
`regime_comparison` and `model_comparison` runs from the API. Save results to
the database the API uses, either while running:

```bash
.venv/bin/python scripts/run_comparison.py regimes --jobs 3 --db artifacts/experiments.db
.venv/bin/python scripts/run_comparison.py models --jobs 3 --db artifacts/experiments.db
```

or from saved JSON results:

```bash
.venv/bin/python scripts/run_comparison.py publish \
  --regimes artifacts/comparison/regimes.json \
  --models artifacts/comparison/models.json \
  --db artifacts/experiments.db
```

For the hosted dashboard, upload that database as described in
[deployment.md](deployment.md#importing-experiment-history).
