import pandas as pd

from src.monitoring.drift import detect_drift, population_stability_index, split_reference_current


def test_population_stability_index_detects_numeric_shift():
    reference = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])
    current = pd.Series([10, 11, 12, 13, 14, 15, 16, 17])

    assert population_stability_index(reference, current) > 0


def test_detect_drift_flags_shifted_numeric_feature():
    reference = pd.DataFrame({"sales_lag_7": [10, 11, 12, 13, 14], "store": ["1"] * 5})
    current = pd.DataFrame({"sales_lag_7": [100, 101, 102, 103, 104], "store": ["1"] * 5})

    findings, summary = detect_drift(reference, current)

    assert not summary.empty
    assert any(finding.feature == "sales_lag_7" for finding in findings)


def test_split_reference_current_returns_non_empty_partitions():
    reference, current = split_reference_current(pd.DataFrame({"x": range(10)}), current_fraction=0.3)

    assert len(reference) == 7
    assert len(current) == 3
