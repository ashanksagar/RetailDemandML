import pandas as pd
import pytest

from src.data.validate import validate_sales_data


def test_validation_rejects_duplicate_keys_and_negative_sales():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "store": ["1", "1"],
            "item": ["1", "1"],
            "sales": [10.0, -1.0],
        }
    )

    report = validate_sales_data(df)

    assert not report.passed
    with pytest.raises(ValueError):
        report.raise_for_errors()
