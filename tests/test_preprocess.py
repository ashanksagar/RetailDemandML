import pandas as pd

from src.data.preprocess import chronological_split, normalize_schema


def test_normalize_schema_accepts_common_aliases():
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "Store": [1],
            "Product": [10],
            "Quantity": [5],
        }
    )

    normalized = normalize_schema(raw)

    assert list(normalized.columns) == ["date", "store", "item", "sales"]
    assert normalized["store"].iloc[0] == "1"
    assert normalized["item"].iloc[0] == "10"


def test_chronological_split_orders_by_future_dates():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30),
            "store": ["1"] * 30,
            "item": ["1"] * 30,
            "sales": range(30),
        }
    )

    train, validation, test = chronological_split(df, validation_days=5, test_days=5)

    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()
