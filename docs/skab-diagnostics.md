# Checking a saved SKAB model by recording

Run this from the repository root using the same partitions and training
boundary as the saved model:

```bash
.venv/bin/python scripts/evaluate_skab_checkpoint.py \
  artifacts/skab_runs/temporal_100epoch_seed42.pt \
  --partitions data/partitions_skab \
  --device mps \
  --threshold-percentile 90 \
  --threshold-scale 1.0 \
  --persistence 1 \
  --threshold-scope recording \
  --recording-report artifacts/skab_runs/diagnostics/seed42_100epoch.json
```

Use `--device cpu` when MPS is unavailable. Add `--plant plant_1` to inspect
just that plant. By default, the evaluator reuses the checkpoint's training
boundary and training policy. Older checkpoints without this metadata use 400
observations per recording and the chronological policy. `--train-size` and
`--training-policy` can override those values for a deliberate comparison.
`--output` can additionally save the existing plant-level evaluation JSON.

The evaluator uses the scaler stored in the checkpoint and computes each
training/test score once. This report does not retrain the model, filter any
windows, or use test labels to calibrate thresholds.

## Reading the report

Each plant contains:

- `training_windows_labeled_anomalous`: windows flagged by the partition's
  window labels. With `label_mode=any`, a window is flagged if any point in it
  is anomalous. A majority-label partition can undercount contaminated windows.
- `training_anomalous_endpoints`: training sequences whose final point is
  labeled anomalous, regardless of the window-label mode.
- `pooled_test_metrics`: the scores and predictions evaluated across the whole
  plant. Its AUC can be high merely because recordings have different normal
  score levels.
- `macro_recording_auc_roc` and `macro_recording_average_precision`: unweighted
  means calculated within recordings containing both test classes. The report
  explicitly counts the recordings included in these averages.
- `recordings`: each recording's actual threshold, training contamination,
  training score distribution, and test metrics.

The test metrics include precision, recall, F1, ROC AUC, average precision,
confusion counts, and the false-positive rate. Average precision summarizes
the precision-recall curve; compare it with the recording's `anomaly_rate`.
`always_alert_f1` gives a simple reference: the F1 obtained by labeling every
test endpoint anomalous.

For anomaly-free recordings, inspect false positives and the false-positive
rate. ROC AUC and average precision are `null` for single-class recordings.
A recording with training windows but no test windows has `test_metrics: null`.

Score distributions include the mean, minimum, median, maximum, and selected
quantiles. Compare normal training and normal test scores for drift, and normal
versus anomalous test scores for fault separation.

Counts refer to overlapping windows or sequence endpoints, not distinct fault
events or independent samples. Repeatedly selecting a policy using these test
labels makes this a development analysis; use separate validation data for
future choices and reserve an untouched final evaluation.

## Clean-training experiment

The LSTM autoencoder learns normal behaviour by reconstructing its training
sequences. If a sequence already contains a labelled fault, training the model
to reconstruct it works against that objective. Run the controlled
`normal-only` experiment with:

```bash
.venv/bin/python scripts/run_experiment.py \
  --partitions data/partitions_skab \
  --rounds 20 \
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --hidden-size 64 \
  --latent-size 16 \
  --num-layers 2 \
  --dropout 0.1 \
  --device mps \
  --seed 42 \
  --evaluate \
  --evaluation-protocol skab \
  --skab-training-policy normal-only \
  --skab-threshold-percentile 90 \
  --skab-threshold-scale 1.0 \
  --skab-persistence 1 \
  --skab-threshold-scope recording \
  --db artifacts/skab_runs/clean_100epoch_seed42.db \
  --output artifacts/skab_runs/clean_100epoch_seed42.pt
```

Only labelled anomalous windows inside each recording's training region are
excluded. The chronological boundary and held-out test data do not change.
This is a label-assisted experiment: in a real plant, the equivalent input
would be a trusted fault-free commissioning period rather than test labels.

## Hybrid recording-baseline experiment

The reconstruction score sums absolute error across sensors. SKAB faults
usually move one sensor, so that sum dilutes them. The hybrid score keeps the
sensors separate and measures each sequence against its own recording's
fault-free start. It takes the larger of the sensor-level distance and the
model-residual distance, then smooths the result:

```bash
.venv/bin/python scripts/run_experiment.py \
  --partitions data/partitions_skab \
  --rounds 20 \
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --hidden-size 64 \
  --latent-size 4 \
  --num-layers 2 \
  --dropout 0.1 \
  --device mps \
  --seed 42 \
  --evaluate \
  --evaluation-protocol skab \
  --skab-training-policy normal-only \
  --skab-scaling plant \
  --skab-score hybrid \
  --skab-exclude-anomaly-free \
  --skab-persistence 1 \
  --db artifacts/skab_runs/hybrid_latent4_seed42.db \
  --output artifacts/skab_runs/hybrid_latent4_seed42.pt
```

Thresholds use no test labels: each is the median plus 24 MADs of the last 25%
of that recording's baseline. `--latent-size 4`, the smoothing, and the
threshold were chosen on half of the recordings, so metrics over all recordings
include tuned data. Read `macro_recording_auc_roc` rather than `auc_roc`, which
can look excellent when recordings merely differ in score level.

`scripts/evaluate_skab_checkpoint.py` only supports shared-scaler checkpoints,
so it rejects checkpoints from this experiment.

The 400-observation baseline does not survive long runs. On the 2.6-hour
anomaly-free recording, 94-97% of normal sequences raise an alert because the
sensors drift. `--skab-exclude-anomaly-free` matches the SKAB leaderboard, but
a real deployment would need a baseline that keeps adapting.
