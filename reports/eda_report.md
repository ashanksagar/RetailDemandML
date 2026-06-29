# RetailDemandML EDA Report

## Dataset Summary

- Rows: 35,040
- Date range: 2021-01-01 to 2022-12-31
- Stores: 4
- Items: 12
- Mean sales: 71.35
- Median sales: 71.00
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

- `C:/Users/dogem/RetailDemandML/reports/figures/eda_total_daily_sales.png`
- `C:/Users/dogem/RetailDemandML/reports/figures/eda_sales_by_day_of_week.png`
- `C:/Users/dogem/RetailDemandML/reports/figures/eda_sales_by_month.png`
- `C:/Users/dogem/RetailDemandML/reports/figures/eda_top_items.png`
- `C:/Users/dogem/RetailDemandML/reports/figures/eda_store_comparison.png`
- `C:/Users/dogem/RetailDemandML/reports/figures/eda_store_item_heatmap.png`
- `C:/Users/dogem/RetailDemandML/reports/figures/eda_example_series.png`
- `C:/Users/dogem/RetailDemandML/reports/figures/eda_lag_correlations.png`
- `C:/Users/dogem/RetailDemandML/reports/figures/eda_split_timeline.png`
- `C:/Users/dogem/RetailDemandML/reports/figures/model_comparison_rmse.png`
