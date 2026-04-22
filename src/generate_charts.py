import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_manufacturing_validation.csv"
IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"
FIGURE_DIR = ROOT / "reports" / "figures"


BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
GRAY = "#64748b"
DARK = "#0f172a"
LIGHT = "#f8fafc"
BORDER = "#cbd5e1"


def svg_page(width, height, body):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="white"/>
  {body}
</svg>
"""


def text(x, y, value, size=14, fill=DARK, weight="400", anchor="start"):
    value = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{value}</text>'


def rect(x, y, width, height, fill, stroke="none", radius=0):
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def line(x1, y1, x2, y2, stroke=BORDER, width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"/>'


def feature_importance_chart():
    df = pd.read_csv(IMPORTANCE_PATH).head(10).iloc[::-1]
    width, height = 900, 520
    left, top, bar_width, row_h = 270, 95, 500, 34
    max_value = df["mean_auc_drop"].max()

    parts = [
        text(40, 42, "Top Process Drivers", 24, DARK, "700"),
        text(40, 68, "Permutation importance measured as ROC AUC drop on holdout data", 13, GRAY),
        line(left, top - 20, left, top + row_h * len(df), BORDER),
    ]

    for idx, row in enumerate(df.itertuples(index=False)):
        y = top + idx * row_h
        label = row.feature.replace("_", " ")
        value = row.mean_auc_drop
        scaled = 0 if max_value == 0 else bar_width * value / max_value
        parts.append(text(left - 12, y + 21, label, 13, DARK, anchor="end"))
        parts.append(rect(left, y + 6, scaled, 18, BLUE, radius=3))
        parts.append(text(left + scaled + 8, y + 21, f"{value:.3f}", 12, GRAY))

    return svg_page(width, height, "\n  ".join(parts))


def confusion_matrix_chart():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cm = metrics["confusion_matrix"]
    matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    colors = [[GREEN, RED], [RED, GREEN]]
    width, height = 720, 500
    cell = 150
    x0, y0 = 210, 140
    max_count = max(max(row) for row in matrix)

    parts = [
        text(40, 42, "Confusion Matrix", 24, DARK, "700"),
        text(40, 68, "Holdout classification results at 0.50 probability threshold", 13, GRAY),
        text(x0 + cell, 108, "Predicted", 14, DARK, "700", "middle"),
        text(72, y0 + cell, "Actual", 14, DARK, "700", "middle"),
        text(x0 + cell / 2, y0 - 16, "Pass", 13, GRAY, "700", "middle"),
        text(x0 + cell * 1.5, y0 - 16, "Fail", 13, GRAY, "700", "middle"),
        text(x0 - 16, y0 + cell / 2 + 5, "Pass", 13, GRAY, "700", "end"),
        text(x0 - 16, y0 + cell * 1.5 + 5, "Fail", 13, GRAY, "700", "end"),
    ]

    for r in range(2):
        for c in range(2):
            count = matrix[r][c]
            opacity = 0.18 + 0.62 * (count / max_count)
            parts.append(rect(x0 + c * cell, y0 + r * cell, cell - 6, cell - 6, colors[r][c], radius=8))
            parts.append(f'<rect x="{x0 + c * cell}" y="{y0 + r * cell}" width="{cell - 6}" height="{cell - 6}" rx="8" fill="white" opacity="{1 - opacity:.2f}"/>')
            parts.append(text(x0 + c * cell + cell / 2 - 3, y0 + r * cell + 64, labels[r][c], 13, DARK, "700", "middle"))
            parts.append(text(x0 + c * cell + cell / 2 - 3, y0 + r * cell + 102, count, 34, DARK, "700", "middle"))

    return svg_page(width, height, "\n  ".join(parts))


def failure_rate_by_line_chart(df):
    grouped = df.groupby("line_id")["validation_fail"].mean().sort_values(ascending=False)
    width, height = 760, 440
    left, top, bar_width, row_h = 170, 115, 450, 58
    max_value = max(grouped.max(), 0.01)

    parts = [
        text(40, 42, "Validation Failure Rate by Manufacturing Line", 23, DARK, "700"),
        text(40, 68, "Synthetic lot-level failure rate by line", 13, GRAY),
    ]

    for idx, (line_id, rate) in enumerate(grouped.items()):
        y = top + idx * row_h
        scaled = bar_width * rate / max_value
        parts.append(text(left - 14, y + 28, line_id, 14, DARK, "700", "end"))
        parts.append(rect(left, y + 8, scaled, 28, BLUE, radius=4))
        parts.append(text(left + scaled + 10, y + 29, f"{rate:.1%}", 13, GRAY))

    return svg_page(width, height, "\n  ".join(parts))


def calibration_trend_chart(df):
    bins = [0, 7, 14, 21, 28, 35, 60]
    labels = ["0-7", "8-14", "15-21", "22-28", "29-35", "36+"]
    out = df.copy()
    out["calibration_bucket"] = pd.cut(out["calibration_days_since"], bins=bins, labels=labels, include_lowest=True)
    grouped = out.groupby("calibration_bucket", observed=True)["validation_fail"].mean()

    width, height = 820, 460
    left, top, chart_w, chart_h = 90, 92, 650, 270
    max_rate = max(grouped.max(), 0.01)

    parts = [
        text(40, 42, "Failure Rate vs. Calibration Age", 23, DARK, "700"),
        text(40, 68, "Older calibration age is modeled as a validation risk driver", 13, GRAY),
        line(left, top, left, top + chart_h, BORDER),
        line(left, top + chart_h, left + chart_w, top + chart_h, BORDER),
    ]

    points = []
    step = chart_w / (len(grouped) - 1)
    for idx, (bucket, rate) in enumerate(grouped.items()):
        x = left + idx * step
        y = top + chart_h - (chart_h * rate / max_rate)
        points.append((x, y, bucket, rate))

    for i in range(len(points) - 1):
        parts.append(line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], BLUE, 3))

    for x, y, bucket, rate in points:
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{BLUE}"/>')
        parts.append(text(x, top + chart_h + 28, bucket, 12, GRAY, anchor="middle"))
        parts.append(text(x, y - 12, f"{rate:.0%}", 12, DARK, "700", "middle"))

    parts.append(text(left - 18, top + 6, f"{max_rate:.0%}", 12, GRAY, anchor="end"))
    parts.append(text(left - 18, top + chart_h + 4, "0%", 12, GRAY, anchor="end"))

    return svg_page(width, height, "\n  ".join(parts))


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    outputs = {
        "feature_importance.svg": feature_importance_chart(),
        "confusion_matrix.svg": confusion_matrix_chart(),
        "failure_rate_by_line.svg": failure_rate_by_line_chart(df),
        "calibration_failure_trend.svg": calibration_trend_chart(df),
    }

    for filename, svg in outputs.items():
        path = FIGURE_DIR / filename
        path.write_text(svg, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
