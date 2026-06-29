import importlib
from typing import Any

import pandas as pd

from src.config import DATE_COLUMN, GROUP_COLUMNS, LAG_PERIODS, ROLLING_WINDOWS, TARGET

try:
    holidays: Any = importlib.import_module("holidays")
except ImportError:  # pragma: no cover - optional dependency fallback
    holidays = None


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    parsed = pd.to_datetime(dates)
    out = pd.DataFrame(index=parsed.index)
    out["day_of_week"] = parsed.dt.dayofweek.astype(str)
    out["month"] = parsed.dt.month.astype(str)
    out["day_of_month"] = parsed.dt.day
    out["week_of_year"] = parsed.dt.isocalendar().week.astype(int)
    out["is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)
    out["is_month_start"] = parsed.dt.is_month_start.astype(int)
    out["is_month_end"] = parsed.dt.is_month_end.astype(int)
    out["quarter"] = parsed.dt.quarter.astype(str)
    out["year_index"] = parsed.dt.year - parsed.dt.year.min()
    if holidays is None:
        out["is_us_holiday"] = "0"
        out["days_to_nearest_us_holiday"] = 999
        return out

    years = range(int(parsed.dt.year.min()) - 1, int(parsed.dt.year.max()) + 2)
    us_holidays = holidays.country_holidays("US", years=years)
    holiday_dates = pd.to_datetime(list(us_holidays.keys()))
    normalized = parsed.dt.normalize()
    out["is_us_holiday"] = normalized.isin(holiday_dates).astype(int).astype(str)
    out["days_to_nearest_us_holiday"] = normalized.map(
        lambda date: int(abs((holiday_dates - date).days).min()) if len(holiday_dates) else 999
    )
    return out


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = pd.concat([out, calendar_features(out[DATE_COLUMN])], axis=1)
    return out


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    out = df.sort_values([*GROUP_COLUMNS, DATE_COLUMN]).copy()
    lags = lags or LAG_PERIODS
    grouped = out.groupby(GROUP_COLUMNS, sort=False)[TARGET]
    for lag in lags:
        out[f"sales_lag_{lag}"] = grouped.shift(lag)
    if {"sales_lag_7", "sales_lag_14"}.issubset(out.columns):
        out["sales_lag_delta_7_14"] = out["sales_lag_7"] - out["sales_lag_14"]
    if {"sales_lag_7", "sales_lag_28"}.issubset(out.columns):
        out["sales_lag_ratio_7_28"] = out["sales_lag_7"] / out["sales_lag_28"].clip(lower=1e-6)
    lag_7_columns = [f"sales_lag_{lag}" for lag in [7, 14, 21, 28] if f"sales_lag_{lag}" in out.columns]
    if lag_7_columns:
        out["sales_same_weekday_4wk_mean"] = out[lag_7_columns].mean(axis=1)
    return out


def add_rolling_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    out = df.sort_values([*GROUP_COLUMNS, DATE_COLUMN]).copy()
    windows = windows or ROLLING_WINDOWS
    grouped = out.groupby(GROUP_COLUMNS, sort=False)[TARGET]
    shifted = grouped.shift(1)
    for window in windows:
        rolling = shifted.groupby([out[c] for c in GROUP_COLUMNS])
        min_periods = max(2, window // 2)
        out[f"sales_rolling_mean_{window}"] = rolling.transform(
            lambda series: series.rolling(window, min_periods=min_periods).mean()
        )
        if window in {7, 28}:
            out[f"sales_rolling_median_{window}"] = rolling.transform(
                lambda series: series.rolling(window, min_periods=min_periods).median()
            )
            out[f"sales_rolling_std_{window}"] = rolling.transform(
                lambda series: series.rolling(window, min_periods=min_periods).std()
            )
            out[f"sales_rolling_min_{window}"] = rolling.transform(
                lambda series: series.rolling(window, min_periods=min_periods).min()
            )
            out[f"sales_rolling_max_{window}"] = rolling.transform(
                lambda series: series.rolling(window, min_periods=min_periods).max()
            )
    return out


def add_history_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([*GROUP_COLUMNS, DATE_COLUMN]).copy()
    # Expanding means are shifted by one row so today's target never leaks into today's features.
    for column in GROUP_COLUMNS:
        out[f"{column}_historical_mean"] = out.groupby(column)[TARGET].transform(
            lambda series: series.shift(1).expanding(min_periods=7).mean()
        )
    out["store_item_historical_mean"] = out.groupby(GROUP_COLUMNS)[TARGET].transform(
        lambda series: series.shift(1).expanding(min_periods=7).mean()
    )
    for keys, name in [
        (["store", "day_of_week"], "store_day_of_week_historical_mean"),
        (["item", "day_of_week"], "item_day_of_week_historical_mean"),
        (["store", "month"], "store_month_historical_mean"),
        (["item", "month"], "item_month_historical_mean"),
    ]:
        out[name] = out.groupby(keys, sort=False)[TARGET].transform(
            lambda series: series.shift(1).expanding(min_periods=3).mean()
        )
    return out


def add_demand_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([*GROUP_COLUMNS, DATE_COLUMN]).copy()
    grouped = out.groupby(GROUP_COLUMNS, sort=False)[TARGET]
    shifted = grouped.shift(1)
    out["recent_zero_sales_28"] = shifted.groupby([out[c] for c in GROUP_COLUMNS]).transform(
        lambda series: (series == 0).rolling(28, min_periods=7).sum()
    )
    out["sales_cv_28"] = out["sales_rolling_std_28"] / out["sales_rolling_mean_28"].clip(lower=1e-6)
    out["sales_momentum_7_28"] = out["sales_rolling_mean_7"] / out["sales_rolling_mean_28"].clip(lower=1e-6)
    out["sales_trend_28"] = shifted.groupby([out[c] for c in GROUP_COLUMNS]).transform(
        lambda series: series.rolling(28, min_periods=14).apply(
            lambda values: float(pd.Series(values).diff().mean()),
            raw=False,
        )
    )
    out["intermittency_score_28"] = out["recent_zero_sales_28"] / 28
    return out


def build_features(df: pd.DataFrame, drop_missing: bool = True) -> pd.DataFrame:
    featured = add_date_features(df)
    featured = add_lag_features(featured)
    featured = add_rolling_features(featured)
    featured = add_history_aggregates(featured)
    featured = add_demand_pattern_features(featured)
    if drop_missing:
        feature_columns = [c for c in featured.columns if c not in [TARGET]]
        featured = featured.dropna(subset=feature_columns)
    return featured.reset_index(drop=True)


def add_forecast_horizon_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1 day.")
    out = df.sort_values([*GROUP_COLUMNS, DATE_COLUMN]).copy()
    out["forecast_horizon"] = horizon
    out["forecast_date"] = out.groupby(GROUP_COLUMNS)[DATE_COLUMN].shift(-horizon)
    out[TARGET] = out.groupby(GROUP_COLUMNS)[TARGET].shift(-horizon)
    return out.dropna(subset=[TARGET, "forecast_date"]).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {DATE_COLUMN, TARGET}
    return [column for column in df.columns if column not in excluded]
