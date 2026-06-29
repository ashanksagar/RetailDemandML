import pandas as pd

from src.api.dashboard import _bar_table, _line_svg, _table


def test_dashboard_table_helpers_render_html():
    rows = [{"model": "ridge", "rmse": 5.1, "smape": 0.06}]

    assert "<table>" in _table(rows, ["model", "rmse"])
    assert "sales_lag_7" in _bar_table(
        [{"feature": "sales_lag_7", "importance": 1.2}], "feature", "importance"
    )


def test_dashboard_line_svg_renders_for_backtest_rows():
    rows = pd.DataFrame({"rmse": [5.4, 5.2], "fold": [1, 2]}).to_dict("records")

    assert "<svg" in _line_svg(rows)
