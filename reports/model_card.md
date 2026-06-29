# RetailDemandML Model Card

## Intended Use

Forecast daily retail item demand by store and item for demand planning, inventory analysis, and
forecasting workflow evaluation. It is not intended for automated replenishment without human review,
promotion data, price data, and production monitoring.

## Data Windows

| Split | Start | End | Rows |
| --- | --- | --- | ---: |
| Train | 2013-02-26 | 2017-07-04 | 793,230 |
| Validation | 2017-07-05 | 2017-10-02 | 45,000 |
| Test | 2017-10-03 | 2017-12-31 | 45,000 |

## Selected Model

Best current production candidate: `hist_gradient_boosting`.

## Test Metrics

```json
{'baseline_test': {'mae': 10.072503384078884, 'rmse': 13.338401634591396, 'mape': 0.20799769051654488, 'smape': 0.19647701086971558}, 'xgboost_validation': {'mae': 6.566386461957296, 'rmse': 8.517973043957063, 'mape': 0.11490427538838206, 'smape': 0.11144469820108871}, 'production_test': {'mae': 5.9484640253177465, 'rmse': 7.7146247090118285, 'mape': 0.13039537756492023, 'smape': 0.125311059240068}, 'selected_model': {'name': 'hist_gradient_boosting'}}
```

## Model Comparison

```text
                 model       mae      rmse     mape    smape
hist_gradient_boosting  6.555544  8.502609 0.114641 0.111287
               xgboost  6.566386  8.517973 0.114904 0.111445
                 ridge  6.658030  8.636433 0.116188 0.112755
         random_forest  6.670060  8.657129 0.115972 0.112827
      moving_average_7  9.823610 12.888990 0.166442 0.156629
      seasonal_naive_7 10.072503 13.338402 0.207998 0.196477
```

## Rolling Backtest

```text
  model  fold  train_end test_start   test_end     rmse    smape
xgboost     1 2017-09-10 2017-09-11 2017-10-08 8.242622 0.119395
xgboost     2 2017-10-08 2017-10-09 2017-11-05 8.189529 0.122829
xgboost     3 2017-11-05 2017-11-06 2017-12-03 8.416166 0.122044
```

## Registry

```json
{
  "models": [
    {
      "version": 1,
      "name": "ridge",
      "artifact_path": "C:\\Users\\dogem\\RetailDemandML\\models\\production_model.joblib",
      "registered_at_utc": "2026-06-28T02:19:37.741779+00:00",
      "metrics": {
        "mae": 4.070067457411941,
        "rmse": 5.126078766297085,
        "mape": 0.0618917574371437,
        "smape": 0.06073176318917653
      },
      "validation_metrics": {
        "mae": 4.228879082202911,
        "rmse": 5.33510549686162,
        "mape": 0.06897194288533282,
        "smape": 0.06816611982431779
      },
      "baseline_metrics": {
        "mae": 6.559469424426993,
        "rmse": 8.055865420490674,
        "mape": 0.10126507006598895,
        "smape": 0.09727742315638765
      },
      "dataset_sha256": "79e3edff06e76b79af709533737d296e239b8940768bd8c0df3ae93f4c554c09",
      "dataset_rows": 35040,
      "dataset_date_range": {
        "min": "2021-01-01",
        "max": "2022-12-31"
      },
      "status": "archived",
      "production_artifact_path": "C:\\Users\\dogem\\RetailDemandML\\models\\registry\\production\\production_model.joblib"
    },
    {
      "version": 2,
      "name": "ridge",
      "artifact_path": "C:\\Users\\dogem\\RetailDemandML\\models\\production_model.joblib",
      "registered_at_utc": "2026-06-28T02:21:57.777154+00:00",
      "metrics": {
        "mae": 4.070067457411941,
        "rmse": 5.126078766297085,
        "mape": 0.0618917574371437,
        "smape": 0.06073176318917653
      },
      "validation_metrics": {
        "mae": 4.228879082202911,
        "rmse": 5.33510549686162,
        "mape": 0.06897194288533282,
        "smape": 0.06816611982431779
      },
      "baseline_metrics": {
        "mae": 6.559469424426993,
        "rmse": 8.055865420490674,
        "mape": 0.10126507006598895,
        "smape": 0.09727742315638765
      },
      "dataset_sha256": "79e3edff06e76b79af709533737d296e239b8940768bd8c0df3ae93f4c554c09",
      "dataset_rows": 35040,
      "dataset_date_range": {
        "min": "2021-01-01",
        "max": "2022-12-31"
      },
      "status": "production",
      "production_artifact_path": "C:\\Users\\dogem\\RetailDemandML\\models\\registry\\production\\production_model.joblib"
    },
    {
      "version": 3,
      "name": "ridge",
      "artifact_path": "C:\\Users\\dogem\\RetailDemandML\\models\\production_model.joblib",
      "registered_at_utc": "2026-06-28T02:28:10.308789+00:00",
      "metrics": {
        "mae": 4.070067457411941,
        "rmse": 5.126078766297085,
        "mape": 0.0618917574371437,
        "smape": 0.06073176318917653
      },
      "validation_metrics": {
        "mae": 4.228879082202911,
        "rmse": 5.33510549686162,
        "mape": 0.06897194288533282,
        "smape": 0.06816611982431779
      },
      "baseline_metrics": {
        "mae": 6.559469424426993,
        "rmse": 8.055865420490674,
        "mape": 0.10126507006598895,
        "smape": 0.09727742315638765
      },
      "dataset_sha256": "79e3edff06e76b79af709533737d296e239b8940768bd8c0df3ae93f4c554c09",
      "dataset_rows": 35040,
      "dataset_date_range": {
        "min": "2021-01-01",
        "max": "2022-12-31"
      },
      "status": "candidate"
    },
    {
      "version": 4,
      "name": "ridge",
      "artifact_path": "C:\\Users\\dogem\\RetailDemandML\\models\\production_model.joblib",
      "registered_at_utc": "2026-06-28T03:24:07.429848+00:00",
      "metrics": {
        "mae": 4.070067457411941,
        "rmse": 5.126078766297085,
        "mape": 0.0618917574371437,
        "smape": 0.06073176318917653
      },
      "validation_metrics": {
        "mae": 4.228879082202911,
        "rmse": 5.33510549686162,
        "mape": 0.06897194288533282,
        "smape": 0.06816611982431779
      },
      "baseline_metrics": {
        "mae": 6.559469424426993,
        "rmse": 8.055865420490674,
        "mape": 0.10126507006598895,
        "smape": 0.09727742315638765
      },
      "dataset_sha256": "79e3edff06e76b79af709533737d296e239b8940768bd8c0df3ae93f4c554c09",
      "dataset_rows": 35040,
      "dataset_date_range": {
        "min": "2021-01-01",
        "max": "2022-12-31"
      },
      "status": "candidate"
    },
    {
      "version": 5,
      "name": "hist_gradient_boosting",
      "artifact_path": "C:\\Users\\dogem\\RetailDemandML\\models\\production_model.joblib",
      "registered_at_utc": "2026-06-29T05:11:24.505370+00:00",
      "metrics": {
        "mae": 5.9484640253177465,
        "rmse": 7.7146247090118285,
        "mape": 0.13039537756492023,
        "smape": 0.125311059240068
      },
      "validation_metrics": {
        "mae": 6.566386461957296,
        "rmse": 8.517973043957063,
        "mape": 0.11490427538838206,
        "smape": 0.11144469820108871
      },
      "baseline_metrics": {
        "mae": 10.072503384078884,
        "rmse": 13.338401634591396,
        "mape": 0.20799769051654488,
        "smape": 0.19647701086971558
      },
      "dataset_sha256": "038f25690a65149c94f86ddd3deceda20c037a5cfd754cafdfc539a72992f2ed",
      "dataset_rows": 913000,
      "dataset_date_range": {
        "min": "2013-01-01",
        "max": "2017-12-31"
      },
      "status": "candidate"
    }
  ],
  "production_version": 2
}
```

## Latest Promotion Decision

```json
{"candidate_version": 5, "previous_production_version": 2, "promoted": false, "reason": "rmse improvement -50.50% below threshold 0.00%", "decided_at_utc": "2026-06-29T05:11:24.517672+00:00"}
```

## Leakage Controls

- Chronological train/validation/test splits.
- Lag and rolling features are shifted so today's target is never included in today's features.
- Encoders are fitted inside model pipelines on training partitions only.
- API feature generation uses the saved historical feature-state artifact created from training data.

## Known Limitations

- Public Store Item Demand data does not include price, promotion, stockout, or competitor signals.
- Cold-start stores/items fall back to global and aggregate history.
- Prediction intervals are residual-based and should be recalibrated on production traffic.
