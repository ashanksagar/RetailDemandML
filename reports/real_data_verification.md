# Real Kaggle Dataset Verification

Status: `verified_real_kaggle_data`
Likely sample data: `False`

## Checks

- PASS `kaggle_cli`: C:\Users\dogem\AppData\Local\Programs\Python\Python313\Scripts\kaggle.EXE
- PASS `kaggle_credentials`: Credential source(s) present: access_token.
- PASS `train_csv_present`: Found C:\Users\dogem\RetailDemandML\data\raw\train.csv.
- PASS `required_columns`: All required columns present.
- PASS `real_dataset_row_count`: Rows=913,000; expected at least 900,000 for Kaggle train.csv.
- PASS `store_count`: Stores=10; expected 10.
- PASS `item_count`: Items=50; expected 50.
- PASS `date_range`: Date range=2013-01-01 to 2017-12-31; expected 2013-01-01 to 2017-12-31.
- PASS `non_negative_sales`: Sales are non-negative.

## Next Steps

- Run `python scripts/run_pipeline.py --config configs/default.yaml` and update README metrics.
