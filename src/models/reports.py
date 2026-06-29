import json
from pathlib import Path

import pandas as pd

from src.config import (
    BACKTEST_PATH,
    DATE_COLUMN,
    MODEL_CARD_PATH,
    MODEL_COMPARISON_PATH,
    PROJECT_ROOT,
    PROMOTIONS_PATH,
    REGISTRY_PATH,
    TARGET,
    ensure_directories,
)


def _repo_relative_string(value: str) -> str:
    project_root = str(PROJECT_ROOT)
    if project_root in value:
        path = Path(value)
        return path.relative_to(PROJECT_ROOT).as_posix()
    return value


def _sanitize_paths(value: object) -> object:
    if isinstance(value, dict):
        return {key: _sanitize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_paths(item) for item in value]
    if isinstance(value, str):
        return _repo_relative_string(value)
    return value


def write_model_card(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    metrics: dict,
    comparison_path: Path = MODEL_COMPARISON_PATH,
    backtest_path: Path = BACKTEST_PATH,
    output_path: Path = MODEL_CARD_PATH,
) -> None:
    ensure_directories()
    best_model = "xgboost"
    comparison_summary = "Model comparison was not generated."
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        if not comparison.empty:
            best_model = str(comparison.sort_values("rmse").iloc[0]["model"])
            comparison_summary = "```text\n" + comparison.to_string(index=False) + "\n```"

    backtest_summary = "Backtest report was not generated."
    if backtest_path.exists():
        backtest = pd.read_csv(backtest_path)
        if not backtest.empty:
            cols = ["model", "fold", "train_end", "test_start", "test_end", "rmse", "smape"]
            backtest_summary = "```text\n" + backtest[cols].to_string(index=False) + "\n```"

    registry_summary = "No model has been registered yet."
    if REGISTRY_PATH.exists():
        registry_payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        registry_summary = "```json\n" + json.dumps(_sanitize_paths(registry_payload), indent=2) + "\n```"

    promotion_summary = "No promotion decisions have been recorded yet."
    if PROMOTIONS_PATH.exists():
        decisions = PROMOTIONS_PATH.read_text(encoding="utf-8").strip().splitlines()
        if decisions:
            promotion_summary = "```json\n" + decisions[-1] + "\n```"
    metrics_json = json.dumps(metrics, indent=2, sort_keys=True)

    text = f"""# RetailDemandML Model Card

## Intended Use

Forecast daily retail item demand by store and item for demand planning, inventory analysis, and
forecasting workflow evaluation. It is not intended for automated replenishment without human review,
promotion data, price data, and production monitoring.

## Data Windows

| Split | Start | End | Rows |
| --- | --- | --- | ---: |
| Train | {train[DATE_COLUMN].min().date()} | {train[DATE_COLUMN].max().date()} | {len(train):,} |
| Validation | {validation[DATE_COLUMN].min().date()} | {validation[DATE_COLUMN].max().date()} | {len(validation):,} |
| Test | {test[DATE_COLUMN].min().date()} | {test[DATE_COLUMN].max().date()} | {len(test):,} |

## Selected Model

Best current production candidate: `{best_model}`.

## Test Metrics

```json
{metrics_json}
```

## Model Comparison

{comparison_summary}

## Rolling Backtest

{backtest_summary}

## Registry

{registry_summary}

## Latest Promotion Decision

{promotion_summary}

## Leakage Controls

- Chronological train/validation/test splits.
- Lag and rolling features are shifted so today's target is never included in today's features.
- Encoders are fitted inside model pipelines on training partitions only.
- API feature generation uses the saved historical feature-state artifact created from training data.

## Known Limitations

- Public Store Item Demand data does not include price, promotion, stockout, or competitor signals.
- Cold-start stores/items fall back to global and aggregate history.
- Prediction intervals are residual-based and should be recalibrated on production traffic.
"""
    output_path.write_text(text, encoding="utf-8")


def residual_prediction_interval(
    validation_actual: pd.Series,
    validation_prediction: pd.Series,
    predictions: pd.Series,
    alpha: float = 0.1,
) -> pd.DataFrame:
    residuals = validation_actual.reset_index(drop=True) - validation_prediction.reset_index(drop=True)
    lower_error = residuals.quantile(alpha / 2)
    upper_error = residuals.quantile(1 - alpha / 2)
    return pd.DataFrame(
        {
            "prediction": predictions,
            "prediction_p10": predictions + lower_error,
            "prediction_p90": predictions + upper_error,
        }
    )


def sliced_metrics(df: pd.DataFrame, prediction_column: str = "prediction") -> pd.DataFrame:
    rows = []
    volume = pd.qcut(df[TARGET], q=3, labels=["low", "medium", "high"], duplicates="drop")
    for label, group in df.assign(volume_bucket=volume).groupby("volume_bucket", observed=True):
        error = group[TARGET] - group[prediction_column]
        rows.append(
            {
                "slice": f"volume={label}",
                "rows": len(group),
                "mae": float(error.abs().mean()),
                "rmse": float((error.pow(2).mean()) ** 0.5),
            }
        )
    for store, group in df.groupby("store"):
        error = group[TARGET] - group[prediction_column]
        rows.append(
            {
                "slice": f"store={store}",
                "rows": len(group),
                "mae": float(error.abs().mean()),
                "rmse": float((error.pow(2).mean()) ** 0.5),
            }
        )
    return pd.DataFrame(rows)
