import pandas as pd

from src.features.build_features import build_features
from src.validation.backtest import rolling_origin_backtest


def test_rolling_origin_backtest_returns_fold_metrics():
    rows = []
    for store in ["1", "2"]:
        for item in ["1", "2"]:
            for index, date in enumerate(pd.date_range("2024-01-01", periods=120)):
                rows.append({"date": date, "store": store, "item": item, "sales": 20 + index % 7})
    features = build_features(pd.DataFrame(rows))

    report = rolling_origin_backtest(features, folds=1, horizon_days=7, step_days=7)

    assert len(report) == 1
    assert report["rmse"].iloc[0] >= 0
