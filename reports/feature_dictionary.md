# RetailDemandML Feature Dictionary

All target-derived features are shifted so the current day's sales never leak into the current row.

## Calendar

- `day_of_week`
- `month`
- `day_of_month`
- `week_of_year`
- `is_weekend`
- `is_month_start`
- `is_month_end`
- `quarter`
- `year_index`
- `is_us_holiday`
- `days_to_nearest_us_holiday`

## Lag

- `sales_lag_*`
- `sales_lag_delta_7_14`
- `sales_lag_ratio_7_28`
- `sales_same_weekday_4wk_mean`

## Rolling

- `sales_rolling_mean_*`
- `sales_rolling_median_*`
- `sales_rolling_std_*`
- `sales_rolling_min_*`
- `sales_rolling_max_*`

## Hierarchical

- `store_historical_mean`
- `item_historical_mean`
- `store_item_historical_mean`
- `store_day_of_week_historical_mean`
- `item_day_of_week_historical_mean`
- `store_month_historical_mean`
- `item_month_historical_mean`

## Demand Pattern

- `recent_zero_sales_28`
- `sales_cv_28`
- `sales_momentum_7_28`
- `sales_trend_28`
- `intermittency_score_28`

## Optional Business Inputs

- `price`
- `promotion_flag`
- `stockout_flag`
- `inventory_on_hand`
