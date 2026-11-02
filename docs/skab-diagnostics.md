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
just that plant. `--train-size` defaults to 400 observations per recording;
it must match the original training split. `--output` can additionally save
the existing plant-level evaluation JSON.

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
