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

## Alert threshold

```bash
.venv/bin/python scripts/run_comparison.py thresholds --jobs 3
```

The alert threshold is a multiple of the baseline's typical variation: the
score is in baseline MAD units, so 24 means an alert when the smoothed score
is 24 MADs above the median of recent normal behaviour. This command trains
the federated LSTM autoencoder (20 rounds × 5 local epochs, seeds 42–44) and
scores the hybrid detector on **development recordings only** at each
candidate threshold. Final recordings are not scored. Results are saved to
`artifacts/comparison/thresholds.json`; a run takes about 3 minutes with
`--jobs 3`.

| Threshold (MAD) | Recall | False alarm rate | Precision | Faults detected | Median delay |
|---|---|---|---|---|---|
| 4 | 0.964 ± 0.001 | 0.626 ± 0.013 | 0.653 ± 0.004 | 17.0/17 | 1 s |
| 8 | 0.929 ± 0.004 | 0.475 ± 0.014 | 0.705 ± 0.005 | 17.0/17 | 8 s |
| 12 | 0.899 ± 0.004 | 0.335 ± 0.017 | 0.767 ± 0.009 | 17.0/17 | 21 s |
| 16 | 0.877 ± 0.004 | 0.221 ± 0.008 | 0.829 ± 0.005 | 17.0/17 | 34 s |
| 18 | 0.865 ± 0.002 | 0.173 ± 0.006 | 0.859 ± 0.004 | 17.0/17 | 40 s |
| 20 | 0.850 ± 0.007 | 0.146 ± 0.009 | 0.877 ± 0.008 | 17.0/17 | 40 s |
| 22 | 0.834 ± 0.013 | 0.133 ± 0.014 | 0.884 ± 0.012 | 17.0/17 | 44 s |
| **24** | **0.817 ± 0.013** | **0.123 ± 0.015** | **0.890 ± 0.014** | **16.7/17** | **50 s** |
| 28 | 0.774 ± 0.006 | 0.102 ± 0.017 | 0.903 ± 0.015 | 16.3/17 | 49 s |
| 32 | 0.700 ± 0.004 | 0.089 ± 0.014 | 0.906 ± 0.013 | 16.0/17 | 53 s |
| 40 | 0.576 ± 0.001 | 0.072 ± 0.001 | 0.907 ± 0.001 | 14.0/17 | 58 s |
| 48 | 0.531 ± 0.012 | 0.062 ± 0.008 | 0.913 ± 0.011 | 12.0/17 | 57 s |
| 64 | 0.484 ± 0.007 | 0.020 ± 0.004 | 0.968 ± 0.006 | 10.7/17 | 58 s |

The JSON also has 6, 10, 14 and 26.

The curve has three regions:

- **Below 16** false alarms climb steeply: at 8 almost half of normal windows
  alarm, which an operator would learn to ignore.
- **From 18 to 28** the trade-off is gentle. Each step of 2 MADs gives up
  about 1.5–2.5 points of recall for 1–2.5 points fewer false alarms, and
  nearly every fault is still caught.
- **Above 32** whole faults start to go unnoticed (14/17 at 40, 11/17 at 64)
  while false alarms barely fall.

24 sits in the gentle region and meets a 15% false alarm budget. It is not
the only defensible choice: the predeclared rule used for the model
comparison (highest recall with at most 15% false alarms) picks 20 on these
runs, which flags about 3 points more of the faulty windows for about 2 points
more false alarms, and alerts 10 s sooner. 24 was frozen before this sweep was
recorded and every final result uses it, so it stays; switching now would
mean choosing a threshold after seeing final results.

The false alarm rate does not carry over from development to final
recordings: at 24 it is 0.123 on development recordings and 0.235 on final
ones (next section). Thresholds calibrated on a short baseline understate how
far a plant drifts later, which is why a real deployment would need a
baseline that keeps adapting.

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

## Personalised and robust federated training

Plain FedAvg trails centralised and isolated training on the model's own
(residual-only) score. Five alternatives were screened first in a separate
scratch experiment, then the two that helped were added as regimes:

```bash
.venv/bin/python scripts/run_comparison.py scenarios --regimes fedprox federated_finetune --jobs 5
```

- **fedprox**: FedAvg with FedProx's proximal term, μ = 0.1, which keeps each
  plant's update near the round's global model.
- **federated_finetune**: FedAvg, then each plant fine-tunes the global model
  on its own data for 5 epochs. Fine-tuning moves no extra bytes.

Settings were fixed before running, not tuned. The repo regimes reproduce the
screen's numbers exactly.

### Model-only AUC, final recordings (mean over seeds 42–44)

| Scenario | Plants | FedAvg | FedProx | FedAvg + fine-tune | Centralised | Isolated |
|---|---|---|---|---|---|---|
| baseline | all | 0.803 | 0.828 | **0.843** | 0.841 | 0.864 |
| heterogeneous | all | 0.753 | 0.773 | **0.776** | 0.768 | 0.802 |
| scarce, 100 s | plant_4, plant_5 | 0.877 | 0.870 | **0.885** | 0.877 | 0.902 |
| scarce, 50 s | plant_4, plant_5 | 0.855 | 0.861 | **0.886** | 0.870 | 0.878 |
| scarce, 25 s | plant_4, plant_5 | 0.845 | **0.858** | 0.836 | 0.857 | 0.888 |

The full hybrid detector's AUC stays within 0.01 of FedAvg for both regimes in
every scenario, so neither changes the headline detection rate.

- **Fine-tuning closes the gap to centralised training.** It matches
  centralised training on the unchanged data and beats it when plants differ,
  while raw data stays at each plant. With only 25 seconds of data it
  does slightly worse than FedAvg, likely because 5 epochs overfit 144
  windows.
- **FedProx is mixed.** It gives the 5-second plant under heterogeneity the
  best score of any method (0.784, against 0.720 for the plant alone), but it
  costs the scarce plants accuracy when they have full data (0.862 against
  FedAvg's 0.904).
- **Isolated plants still lead** in every scenario except 50 s scarcity,
  where fine-tuning's 0.886 against 0.878 is within run-to-run noise.

### Screened and not added

| Strategy | baseline | heterogeneous | scarce, 25 s | Why not added |
|---|---|---|---|---|
| Coordinate-wise median | 0.817 | 0.754 | 0.812 | No consistent gain; discards honest plants' updates |
| Trimmed mean (1 of 5 each side) | 0.794 | 0.758 | 0.844 | Worse than FedAvg on unchanged data |
| Clustered FL (2 clusters after 3 rounds) | 0.826 | 0.772 | 0.855 | No better than FedProx; clusters changed between seeds when data was scarce |

Server-side validation sets and reputation systems were not screened. They
defend against unreliable or malicious clients, and a validation set would
put plant data on the server, against HydroFed's privacy aim. The screening
script and its results are kept in `artifacts/comparison/strategy-screen/`.

## Learning from alert reviews

```bash
.venv/bin/python scripts/run_comparison.py feedback --jobs 3
```

Operators mark dashboard alerts as resolved (a real fault) or as a false
alarm. This development-only analysis measures three ways the detector could
use those reviews. A simulated reviewer follows the SKAB labels: an alert that
overlaps a labelled fault is resolved 30 seconds after the fault ends, and any
other alert is a false alarm. Each plant's earlier development recordings
supply the reviews, and its later ones (8 recordings, 8 faults) are evaluated.
Final recordings are not used. Results go to `artifacts/comparison/feedback.json`.

Where false alarms come from sets what reviews can fix. In the operator feed
(seed 42), 96% of development false-alarm seconds come after a labelled fault
ends, inside alerts that do overlap the fault. Only 4% come from alerts with no
fault. Alerts that ring on after a fault would be reviewed as resolved, not as
false alarms.

- **Re-baseline on resolved**: after a resolved fault, the plant's normal
  reference is updated from the next 60 seconds. *Re-learn* replaces the
  reference; *re-centre* keeps the original spread and calibration and only
  shifts each sensor's normal level to where the plant settled. Variants
  shelve alarms during that minute, or have the reviewer resolve too early,
  halfway through the fault.
- **Per-plant threshold**: rises just above the plant's past false alarms,
  capped at 48 and kept below every past confirmed fault's peak.
- **Reviews as labels**: federated XGBoost trained on the earlier recordings
  with labels taken from reviewed alerts (whole resolved alerts count as
  faults; faults that never alerted count as normal), alerting at a fault
  probability of 0.5.

### Later development recordings (mean over seeds 42–44)

| Method | Recall | False alarm rate | Within-recording AUC | Faults | Median delay |
|---|---|---|---|---|---|
| Current detector | 0.750 | 5.9% | 0.915 | 7.7/8 | 53 s |
| Re-centre after a resolved fault | 0.750 | 5.4% | 0.937 | 7.7/8 | 53 s |
| Re-centre, alarms shelved for the minute | 0.750 | 3.8% | 0.957 | 7.7/8 | 53 s |
| Re-learn after a resolved fault | 0.750 | 19.7% | 0.856 | 7.7/8 | 53 s |
| Re-centre, resolved too early | 0.563 | 34.7% | 0.684 | 7.7/8 | 53 s |
| Re-learn, resolved too early | 0.709 | 46.5% | 0.560 | 7.7/8 | 53 s |
| Per-plant threshold from false alarms | 0.692 | 4.8% | 0.915 | 7.7/8 | 53 s |
| Federated XGBoost, labels from reviews | 0.738 | 5.4% | 0.947 | 8/8 | 28 s |
| Federated XGBoost, true labels | 0.558 | 13.3% | 0.903 | 8/8 | 20 s |

### What the reviews can and cannot do

- **Re-centring after a resolved fault works where it acts.** On seed 42 it
  removed every false alarm after the new reference took over (39 seconds to
  0) without losing a fault. The 194 false-alarm seconds that remain fall in
  the 90 seconds before it takes over. SKAB recordings end about 170 seconds
  after their fault, so little time is left to benefit. In longer operation
  the effect would build up, but this data cannot show that, or whether a
  later fault would be hidden.
- **Re-learning from one minute is fragile.** A minute of a plant settling
  after a fault has too little spread, so normal operation afterwards looks
  abnormal and false alarms more than triple.
- **A wrong review is costly.** Resolving a fault while it is still happening
  pulls the fault into the reference: false alarms rise about sixfold and
  recall falls (re-centring: 5.9% to 34.7% false alarms, recall 0.750 to 0.563),
  so a resolution should only update the reference after an administrator
  confirms it.
- **Learned thresholds trade recall for false alarms.** Only plant_5's
  threshold rose, in two of three seeds (to 31 and 37), and that cost recall
  (0.750 to 0.692) more than it saved in false alarms (5.9% to 4.8%).
- **Reviews make usable labels.** Federated XGBoost trained on review labels
  matched the detector's recall and false alarm rate, with higher AUC and
  half the delay. It also beat the same model trained on the true labels at
  the fixed 0.5 cut-off, which is surprising: review labels mark whole alerts,
  so they contain more fault examples. Treat that comparison as unsettled
  rather than as evidence that review labels are better.

### Limits

- Development-only analysis of 8 later recordings and 3 seeds; it is not a
  held-out result.
- The simulated reviewer is perfect apart from the deliberate too-early case.
  Real reviews would contain mistakes, and the dashboard shows SKAB's label
  in the review drawer.
- Review delay, reference length and the threshold cap were set before the
  full run to fit SKAB's short recordings, not tuned.

## Plants that miss federated rounds

```bash
.venv/bin/python scripts/run_comparison.py connectivity --jobs 6
```

Remote plants may not reach the server every round. Only the plants that
connect in a round train and are averaged, weighted by examples as in Flower.
A plant that missed rounds receives the final shared model when it
reconnects. The scenarios were fixed before running:

- **full**: all five plants every round, the reference.
- **sampled k of 5**: k plants drawn at random each round, as Flower's
  fraction_fit does (0.8, 0.6 and 0.4).
- **late join**: plant_4 and plant_5 are offline for rounds 1–15 and join for
  the last 5.
- **never join**: plant_4 and plant_5 miss every round and only receive the
  final model, like a new plant joining a trained fleet.

Results use the frozen alert policy on final recordings, with no new tuning,
and go to `artifacts/comparison/connectivity.json`.

### Results, final recordings (mean ± sd over seeds 42–44)

Within-recording AUC-ROC. "Affected" means plant_4 and plant_5.

| Scenario | MB moved | Full detector | Model only | Affected: full detector | Affected: model only | Affected: faults | Others: model only |
|---|---|---|---|---|---|---|---|
| full | 83.4 | 0.845 ± 0.002 | 0.803 ± 0.020 | 0.885 | 0.904 ± 0.023 | 8/8 | 0.713 ± 0.018 |
| sampled 4 of 5 | 66.7 | 0.843 ± 0.003 | 0.791 ± 0.027 | 0.882 | 0.879 ± 0.028 | 8/8 | 0.714 ± 0.031 |
| sampled 3 of 5 | 50.0 | 0.843 ± 0.001 | 0.776 ± 0.038 | 0.881 | 0.868 ± 0.030 | 7.7/8 | 0.694 ± 0.048 |
| sampled 2 of 5 | 33.3 | 0.844 ± 0.003 | 0.781 ± 0.004 | 0.882 | 0.870 ± 0.045 | 7.7/8 | 0.702 ± 0.035 |
| late join | 58.4 | 0.844 ± 0.000 | 0.793 ± 0.012 | 0.880 | 0.886 ± 0.031 | 7.7/8 | 0.710 ± 0.005 |
| never join | 50.0 | 0.841 ± 0.001 | 0.799 ± 0.004 | 0.878 | 0.837 ± 0.024 | 8/8 | 0.765 ± 0.016 |

- **Missing rounds barely changed detection.** The full detector's AUC-ROC
  stayed between 0.841 and 0.845 in every scenario, because its
  baseline-distance part needs no training.
- **The model degraded gradually.** With 2 to 4 plants per round, the
  model-only AUC-ROC fell by 0.01 to 0.03, within the spread between seeds.
- **Plants that never connected got a weaker model.** For plant_4 and
  plant_5, the model-only AUC-ROC fell from 0.904 to 0.837, about three
  standard deviations. The full detector still caught all 8 of their faults.
  Joining for the last 5 rounds recovered most of the loss (0.886).
- **The other plants gained when those two stayed away** (0.713 to 0.765): a
  shared model fits the plants that train it, the same penalty for unlike
  plants seen under heterogeneity.
- **Communication fell in proportion to participation**, from 83.4 MB with
  every plant to 33.3 MB with two plants per round.

### Limits

- Dropout was simulated per round. Real outages may last longer or fall
  mid-round, and late uploads from a stale model were not modelled.
- Clients were sampled uniformly. Plants that drop out because of their
  location or size, and so differ from the rest, were tested only through
  the late-join and never-join cases.
- Three seeds and 8 affected faults; differences of a few hundredths in the
  model-only score are within run-to-run noise.

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
