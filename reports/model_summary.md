# Model Summary

Synthetic manufacturing validation failure model.

## Holdout Metrics

- Accuracy: 0.703
- Precision: 0.763
- Recall: 0.634
- F1: 0.693
- ROC AUC: 0.760

## Top Process Drivers

- `temp_abs_deviation`: mean ROC AUC drop 0.059
- `calibration_days_since`: mean ROC AUC drop 0.048
- `rework_count`: mean ROC AUC drop 0.047
- `machine_downtime_min`: mean ROC AUC drop 0.031
- `incoming_inspection_score`: mean ROC AUC drop 0.016
- `pressure_abs_deviation`: mean ROC AUC drop 0.014
- `batch_size`: mean ROC AUC drop 0.013
- `shift_Night`: mean ROC AUC drop 0.012

## Interpretation

The highest-ranked features point to process conditions that would be natural candidates for root-cause review: rework history, calibration age, equipment downtime, material age, incoming inspection score, and deviation from target process settings.

Because the dataset is synthetic, results should be interpreted as a demonstration of the workflow rather than evidence about a real manufacturing line.
