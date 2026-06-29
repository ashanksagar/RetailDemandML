import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import METRICS_PATH, TARGET, ensure_directories


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    errors = actual - pred
    denominator = np.maximum(np.abs(actual), 1e-8)
    smape_denominator = np.maximum((np.abs(actual) + np.abs(pred)) / 2, 1e-8)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(np.abs(errors) / denominator)),
        "smape": float(np.mean(np.abs(errors) / smape_denominator)),
    }


def write_metrics(metrics: dict, path: Path = METRICS_PATH) -> None:
    ensure_directories()
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.predictions)
    metrics = regression_metrics(df[TARGET], df["prediction"])
    write_metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
