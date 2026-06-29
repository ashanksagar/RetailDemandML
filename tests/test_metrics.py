import pytest

from src.models.evaluate import regression_metrics


def test_regression_metrics_are_computed():
    metrics = regression_metrics([10, 20, 30], [12, 18, 33])

    assert metrics["mae"] == pytest.approx(7 / 3)
    assert metrics["rmse"] > 0
    assert metrics["mape"] > 0
    assert metrics["smape"] > 0
