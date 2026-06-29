# RetailDemandML EDA Report

## Dataset Summary

- Rows: 913,000
- Date range: 2013-01-01 to 2017-12-31
- Stores: 10
- Items: 50
- Mean sales: 52.25
- Median sales: 47.00
- Zero-sales rate: 0.00%
- Duplicate date/store/item keys: 0
- Missing values: 0

## Modeling Implications

- Chronological validation is required because store/item demand is autocorrelated.
- Weekly seasonality should be represented with day-of-week and lag features.
- Store and item effects are material, so models need categorical and hierarchical features.
- Lag and rolling features must be shifted to avoid target leakage.
- The public dataset does not include price, promotions, or stockouts, so these remain documented optional inputs.

## Generated Figures

- `reports/figures/eda_total_daily_sales.png`
- `reports/figures/eda_sales_by_day_of_week.png`
- `reports/figures/eda_sales_by_month.png`
- `reports/figures/eda_top_items.png`
- `reports/figures/eda_store_comparison.png`
- `reports/figures/eda_store_item_heatmap.png`
- `reports/figures/eda_example_series.png`
- `reports/figures/eda_lag_correlations.png`
- `reports/figures/eda_split_timeline.png`
- `reports/figures/model_comparison_rmse.png`
