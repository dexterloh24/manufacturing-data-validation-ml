from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_manufacturing_validation.csv"


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def build_dataset(rows=850, seed=42):
    rng = np.random.default_rng(seed)

    line_id = rng.choice(["Line_A", "Line_B", "Line_C"], size=rows, p=[0.42, 0.36, 0.22])
    shift = rng.choice(["Day", "Night"], size=rows, p=[0.58, 0.42])
    supplier_tier = rng.choice(["Tier_1", "Tier_2", "Tier_3"], size=rows, p=[0.62, 0.27, 0.11])

    operator_experience_months = np.clip(rng.normal(34, 18, rows), 1, 96).round(1)
    batch_size = np.clip(rng.normal(145, 38, rows), 45, 260).round().astype(int)
    ambient_humidity_pct = np.clip(rng.normal(46, 11, rows), 20, 82).round(1)
    machine_downtime_min = np.clip(rng.gamma(shape=2.2, scale=9.0, size=rows), 0, 85).round(1)
    calibration_days_since = np.clip(rng.normal(18, 9, rows), 0, 52).round(1)
    material_age_days = np.clip(rng.gamma(shape=2.4, scale=7.0, size=rows), 1, 65).round(1)
    incoming_inspection_score = np.clip(rng.normal(91, 5.5, rows), 68, 100).round(1)
    process_temperature_c = np.clip(rng.normal(72, 3.8, rows), 60, 85).round(1)
    process_pressure_psi = np.clip(rng.normal(35, 4.2, rows), 23, 48).round(1)

    rework_base = (
        0.25
        + 0.018 * machine_downtime_min
        + 0.03 * np.maximum(calibration_days_since - 24, 0)
        + 0.12 * (supplier_tier == "Tier_3")
    )
    rework_count = rng.poisson(np.clip(rework_base, 0.05, 2.4))

    temp_deviation = np.abs(process_temperature_c - 72)
    pressure_deviation = np.abs(process_pressure_psi - 35)

    risk_score = (
        -2.05
        + 0.72 * rework_count
        + 0.042 * (machine_downtime_min - 18)
        + 0.064 * (calibration_days_since - 18)
        + 0.034 * (material_age_days - 16)
        - 0.090 * (incoming_inspection_score - 90)
        + 0.26 * temp_deviation
        + 0.19 * pressure_deviation
        + 0.0060 * (batch_size - 140)
        - 0.014 * (operator_experience_months - 30)
        + 0.50 * (line_id == "Line_C")
        + 0.35 * (shift == "Night")
        + 0.48 * (supplier_tier == "Tier_3")
    )

    failure_probability = sigmoid(risk_score)
    validation_fail = rng.binomial(1, failure_probability)

    return pd.DataFrame(
        {
            "lot_id": [f"LOT-{idx:04d}" for idx in range(1, rows + 1)],
            "line_id": line_id,
            "shift": shift,
            "supplier_tier": supplier_tier,
            "operator_experience_months": operator_experience_months,
            "batch_size": batch_size,
            "ambient_humidity_pct": ambient_humidity_pct,
            "machine_downtime_min": machine_downtime_min,
            "calibration_days_since": calibration_days_since,
            "material_age_days": material_age_days,
            "incoming_inspection_score": incoming_inspection_score,
            "process_temperature_c": process_temperature_c,
            "process_pressure_psi": process_pressure_psi,
            "rework_count": rework_count,
            "validation_fail": validation_fail,
        }
    )


def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {DATA_PATH}")
    print(f"Validation failure rate: {df['validation_fail'].mean():.1%}")


if __name__ == "__main__":
    main()
