from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import (
    DATE_COLUMN,
    FEATURE_STORE_PATH,
    GROUP_COLUMNS,
    LAG_PERIODS,
    OPTIONAL_BUSINESS_COLUMNS,
    ROLLING_WINDOWS,
    TARGET,
    ensure_directories,
)
from src.features.build_features import calendar_features


@dataclass
class SeriesState:
    store: str
    item: str
    history: pd.DataFrame
    store_historical_mean: float
    item_historical_mean: float


def build_feature_store(df: pd.DataFrame, path: Path = FEATURE_STORE_PATH) -> dict:
    ensure_directories()
    clean = df.sort_values([*GROUP_COLUMNS, DATE_COLUMN]).copy()
    states: dict[tuple[str, str], SeriesState] = {}
    store_means = clean.groupby("store")[TARGET].mean()
    item_means = clean.groupby("item")[TARGET].mean()
    store_item_means = clean.groupby(GROUP_COLUMNS)[TARGET].mean()

    for (store, item), group in clean.groupby(GROUP_COLUMNS):
        states[(str(store), str(item))] = SeriesState(
            store=str(store),
            item=str(item),
            history=group[[DATE_COLUMN, TARGET]].tail(max(LAG_PERIODS + ROLLING_WINDOWS) + 31).copy(),
            store_historical_mean=float(store_means.loc[store]),
            item_historical_mean=float(item_means.loc[item]),
        )

    artifact = {
        "states": states,
        "store_item_means": {tuple(map(str, key)): float(value) for key, value in store_item_means.items()},
        "global_mean": float(clean[TARGET].mean()),
        "last_date": clean[DATE_COLUMN].max(),
    }
    joblib.dump(artifact, path)
    return artifact


def load_feature_store(path: Path = FEATURE_STORE_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Feature store not found at {path}. Run `make train` first.")
    return joblib.load(path)


def _lookup_lag(history: pd.DataFrame, forecast_date: pd.Timestamp, lag: int, fallback: float) -> float:
    lookup_date = forecast_date - pd.Timedelta(days=lag)
    matched = history.loc[history[DATE_COLUMN] == lookup_date, TARGET]
    if matched.empty:
        return fallback
    return float(matched.iloc[-1])


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def build_online_feature_row(
    store: str,
    item: str,
    forecast_date: str | pd.Timestamp,
    feature_store: dict,
) -> dict:
    key = (str(store), str(item))
    state: SeriesState | None = feature_store["states"].get(key)
    global_mean = float(feature_store["global_mean"])
    target_date = pd.Timestamp(forecast_date)

    if state is None:
        history = pd.DataFrame(columns=[DATE_COLUMN, TARGET])
        store_mean = global_mean
        item_mean = global_mean
        store_item_mean = global_mean
    else:
        history = state.history.copy()
        store_mean = state.store_historical_mean
        item_mean = state.item_historical_mean
        store_item_mean = float(feature_store.get("store_item_means", {}).get(key, global_mean))

    history[DATE_COLUMN] = pd.to_datetime(history[DATE_COLUMN])
    fallback = float(history[TARGET].mean()) if not history.empty else global_mean
    row: dict[str, object] = {"store": str(store), "item": str(item)}
    row.update(calendar_features(pd.Series([target_date])).iloc[0].to_dict())
    for lag in LAG_PERIODS:
        row[f"sales_lag_{lag}"] = _lookup_lag(history, target_date, lag, fallback)
    if "sales_lag_7" in row and "sales_lag_14" in row:
        row["sales_lag_delta_7_14"] = _as_float(row["sales_lag_7"]) - _as_float(row["sales_lag_14"])
    if "sales_lag_7" in row and "sales_lag_28" in row:
        row["sales_lag_ratio_7_28"] = _as_float(row["sales_lag_7"]) / max(_as_float(row["sales_lag_28"]), 1e-6)
    same_weekday_values = [
        _as_float(row[f"sales_lag_{lag}"])
        for lag in [7, 14, 21, 28]
        if f"sales_lag_{lag}" in row
    ]
    row["sales_same_weekday_4wk_mean"] = (
        float(sum(same_weekday_values) / len(same_weekday_values)) if same_weekday_values else fallback
    )
    prior = history[history[DATE_COLUMN] < target_date].sort_values(DATE_COLUMN)
    for window in ROLLING_WINDOWS:
        recent = prior.tail(window)[TARGET]
        row[f"sales_rolling_mean_{window}"] = float(recent.mean()) if not recent.empty else fallback
        if window in {7, 28}:
            row[f"sales_rolling_median_{window}"] = float(recent.median()) if not recent.empty else fallback
            row[f"sales_rolling_std_{window}"] = float(recent.std()) if len(recent) > 1 else 0.0
            row[f"sales_rolling_min_{window}"] = float(recent.min()) if not recent.empty else fallback
            row[f"sales_rolling_max_{window}"] = float(recent.max()) if not recent.empty else fallback
    row["store_historical_mean"] = store_mean
    row["item_historical_mean"] = item_mean
    row["store_item_historical_mean"] = store_item_mean
    row["store_day_of_week_historical_mean"] = store_mean
    row["item_day_of_week_historical_mean"] = item_mean
    row["store_month_historical_mean"] = store_mean
    row["item_month_historical_mean"] = item_mean
    recent_28 = prior.tail(28)[TARGET]
    zero_count = int((recent_28 == 0).sum()) if not recent_28.empty else 0
    row["recent_zero_sales_28"] = zero_count
    row["sales_cv_28"] = _as_float(row.get("sales_rolling_std_28", 0.0)) / max(
        _as_float(row.get("sales_rolling_mean_28", fallback)),
        1e-6,
    )
    row["sales_momentum_7_28"] = _as_float(row.get("sales_rolling_mean_7", fallback)) / max(
        _as_float(row.get("sales_rolling_mean_28", fallback)),
        1e-6,
    )
    row["sales_trend_28"] = float(recent_28.diff().mean()) if len(recent_28) > 1 else 0.0
    row["intermittency_score_28"] = zero_count / 28
    for column in OPTIONAL_BUSINESS_COLUMNS:
        row[column] = 0.0
    return row
