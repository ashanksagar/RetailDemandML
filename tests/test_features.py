import pandas as pd

from src.features.build_features import build_features
from src.features.feature_store import build_feature_store, build_online_feature_row


def test_build_features_uses_past_sales_only_for_lags():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40),
            "store": ["1"] * 40,
            "item": ["1"] * 40,
            "sales": list(range(40)),
        }
    )

    features = build_features(df, drop_missing=False)

    row = features.loc[features["date"] == pd.Timestamp("2024-01-29")].iloc[0]
    assert row["sales_lag_7"] == 21
    assert row["sales_lag_14"] == 14
    assert row["sales_lag_28"] == 0


def test_feature_store_builds_business_prediction_features(tmp_path):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40),
            "store": ["1"] * 40,
            "item": ["1"] * 40,
            "sales": list(range(40)),
        }
    )

    store = build_feature_store(df, path=tmp_path / "feature_store.joblib")
    row = build_online_feature_row("1", "1", "2024-02-10", store)

    assert row["store"] == "1"
    assert row["item"] == "1"
    assert row["sales_lag_7"] == 33
    assert row["sales_rolling_mean_7"] == 36
