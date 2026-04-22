import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_manufacturing_validation.csv"
REPORT_DIR = ROOT / "reports"
METRICS_PATH = REPORT_DIR / "model_metrics.json"
IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"
SUMMARY_PATH = REPORT_DIR / "model_summary.md"

TARGET = "validation_fail"
DROP_COLUMNS = ["lot_id", TARGET]


def validate_input(df):
    required = {
        "lot_id",
        "line_id",
        "shift",
        "supplier_tier",
        "operator_experience_months",
        "batch_size",
        "ambient_humidity_pct",
        "machine_downtime_min",
        "calibration_days_since",
        "material_age_days",
        "incoming_inspection_score",
        "process_temperature_c",
        "process_pressure_psi",
        "rework_count",
        TARGET,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    null_counts = df[list(required)].isna().sum()
    if int(null_counts.sum()) > 0:
        raise ValueError(f"Unexpected missing values found: {null_counts[null_counts > 0].to_dict()}")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must be a binary 0/1 field")


def add_engineered_features(df):
    out = df.copy()
    out["temp_abs_deviation"] = (out["process_temperature_c"] - 72).abs()
    out["pressure_abs_deviation"] = (out["process_pressure_psi"] - 35).abs()
    out["calibration_overdue_flag"] = (out["calibration_days_since"] > 28).astype(int)
    out["high_rework_flag"] = (out["rework_count"] >= 2).astype(int)
    out["large_batch_flag"] = (out["batch_size"] >= 175).astype(int)
    return out


def train_test_split(df, test_fraction=0.25, seed=7):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    test_size = int(len(df) * test_fraction)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def prepare_features(train_df, test_df):
    train_x = pd.get_dummies(train_df.drop(columns=DROP_COLUMNS), drop_first=True)
    test_x = pd.get_dummies(test_df.drop(columns=DROP_COLUMNS), drop_first=True)
    train_x, test_x = train_x.align(test_x, join="left", axis=1, fill_value=0)

    means = train_x.mean()
    stds = train_x.std(ddof=0).replace(0, 1)

    train_scaled = (train_x - means) / stds
    test_scaled = (test_x - means) / stds

    return train_scaled, test_scaled, train_df[TARGET].to_numpy(), test_df[TARGET].to_numpy()


def fit_logistic_regression(x, y, learning_rate=0.08, epochs=2600, l2=0.015):
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.astype(float)
    weights = np.zeros(x_arr.shape[1])
    bias = 0.0

    for _ in range(epochs):
        logits = x_arr @ weights + bias
        preds = 1 / (1 + np.exp(-np.clip(logits, -35, 35)))
        error = preds - y_arr
        grad_w = (x_arr.T @ error) / len(y_arr) + l2 * weights
        grad_b = error.mean()
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias


def predict_proba(x, weights, bias):
    logits = x.to_numpy(dtype=float) @ weights + bias
    return 1 / (1 + np.exp(-np.clip(logits, -35, 35)))


def roc_auc_score(y_true, scores):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive_ranks = ranks[y_true == 1].sum()
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (positive_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def evaluate(y_true, probabilities, threshold=0.5):
    preds = (probabilities >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def permutation_importance(x, y, weights, bias, repeats=10, seed=11):
    rng = np.random.default_rng(seed)
    baseline = evaluate(y, predict_proba(x, weights, bias))["roc_auc"]
    rows = []

    for feature in x.columns:
        drops = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            score = evaluate(y, predict_proba(shuffled, weights, bias))["roc_auc"]
            drops.append(baseline - score)
        rows.append(
            {
                "feature": feature,
                "mean_auc_drop": round(float(np.mean(drops)), 5),
                "std_auc_drop": round(float(np.std(drops)), 5),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_auc_drop", ascending=False)


def write_summary(metrics, importance):
    top_features = importance.head(8)
    lines = [
        "# Model Summary",
        "",
        "Synthetic manufacturing validation failure model.",
        "",
        "## Holdout Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- Precision: {metrics['precision']:.3f}",
        f"- Recall: {metrics['recall']:.3f}",
        f"- F1: {metrics['f1']:.3f}",
        f"- ROC AUC: {metrics['roc_auc']:.3f}",
        "",
        "## Top Process Drivers",
        "",
    ]

    for _, row in top_features.iterrows():
        lines.append(f"- `{row['feature']}`: mean ROC AUC drop {row['mean_auc_drop']:.3f}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The highest-ranked features point to process conditions that would be natural candidates for root-cause review: rework history, calibration age, equipment downtime, material age, incoming inspection score, and deviation from target process settings.",
            "",
            "Because the dataset is synthetic, results should be interpreted as a demonstration of the workflow rather than evidence about a real manufacturing line.",
        ]
    )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run src/generate_synthetic_data.py before training the model.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    validate_input(df)
    df = add_engineered_features(df)

    train_df, test_df = train_test_split(df)
    train_x, test_x, train_y, test_y = prepare_features(train_df, test_df)

    weights, bias = fit_logistic_regression(train_x, train_y)
    probabilities = predict_proba(test_x, weights, bias)
    metrics = evaluate(test_y, probabilities)
    importance = permutation_importance(test_x, test_y, weights, bias)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    importance.to_csv(IMPORTANCE_PATH, index=False)
    write_summary(metrics, importance)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
