import pandas as pd

from src.config import DATE_COLUMN, GROUP_COLUMNS, TARGET
from src.models.evaluate import regression_metrics


def seasonal_naive_predictions(train: pd.DataFrame, test: pd.DataFrame, lag_days: int = 7) -> pd.DataFrame:
    history = train[[*GROUP_COLUMNS, DATE_COLUMN, TARGET]].copy()
    future = test[[*GROUP_COLUMNS, DATE_COLUMN, TARGET]].copy()
    lookup = history.rename(columns={TARGET: "prediction"})
    future["lookup_date"] = future[DATE_COLUMN] - pd.Timedelta(days=lag_days)
    merged = future.merge(
        lookup,
        left_on=[*GROUP_COLUMNS, "lookup_date"],
        right_on=[*GROUP_COLUMNS, DATE_COLUMN],
        how="left",
        suffixes=("", "_history"),
    )
    fallback = history.groupby(GROUP_COLUMNS)[TARGET].mean().rename("fallback_prediction").reset_index()
    merged = merged.merge(fallback, on=GROUP_COLUMNS, how="left")
    merged["prediction"] = merged["prediction"].fillna(merged["fallback_prediction"])
    return merged[[*GROUP_COLUMNS, DATE_COLUMN, TARGET, "prediction"]]


def evaluate_baseline(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    predictions = seasonal_naive_predictions(train, test)
    return predictions, regression_metrics(predictions[TARGET], predictions["prediction"])
