import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    BACKTEST_PATH,
    DATE_COLUMN,
    RAW_TRAIN_PATH,
    SAMPLE_TRAIN_PATH,
    TARGET,
    ensure_directories,
    load_project_config,
)
from src.data.ingest import generate_sample_data, load_raw_sales
from src.data.preprocess import chronological_split, normalize_schema
from src.features.build_features import build_features
from src.models.evaluate import regression_metrics
from src.models.train_xgboost import build_xgboost_pipeline


def rolling_origin_backtest(
    features: pd.DataFrame,
    folds: int = 3,
    horizon_days: int = 28,
    step_days: int = 28,
) -> pd.DataFrame:
    ensure_directories()
    unique_dates = pd.Series(sorted(features[DATE_COLUMN].unique()))
    if len(unique_dates) < horizon_days * folds + step_days:
        raise ValueError("Not enough history for requested rolling backtest.")

    rows = []
    feature_columns = [column for column in features.columns if column not in {DATE_COLUMN, TARGET}]
    max_date = unique_dates.iloc[-1]
    first_test_start = max_date - pd.Timedelta(days=(folds * step_days) + horizon_days - 1)

    for fold in range(folds):
        test_start = first_test_start + pd.Timedelta(days=fold * step_days)
        test_end = test_start + pd.Timedelta(days=horizon_days - 1)
        train = features[features[DATE_COLUMN] < test_start].copy()
        test = features[(features[DATE_COLUMN] >= test_start) & (features[DATE_COLUMN] <= test_end)].copy()
        if train.empty or test.empty:
            continue

        model = build_xgboost_pipeline(
            feature_columns,
            params={"n_estimators": 100, "max_depth": 5, "learning_rate": 0.06},
        )
        model.fit(train[feature_columns], train[TARGET])
        predictions = model.predict(test[feature_columns])
        metrics = regression_metrics(test[TARGET], predictions)
        rows.append(
            {
                "model": "xgboost",
                "fold": fold + 1,
                "train_end": train[DATE_COLUMN].max().date(),
                "test_start": test_start.date(),
                "test_end": test_end.date(),
                "rows": len(test),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def write_backtest_report(backtest: pd.DataFrame, path: Path = BACKTEST_PATH) -> None:
    ensure_directories()
    backtest.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    raw_path = SAMPLE_TRAIN_PATH if args.sample else RAW_TRAIN_PATH
    if args.sample:
        raw_path = generate_sample_data()
    config = load_project_config(args.config)
    raw = load_raw_sales(raw_path)
    clean = normalize_schema(raw)
    features = build_features(clean)
    train, validation, test = chronological_split(
        features,
        validation_days=config.data.validation_days,
        test_days=config.data.test_days,
    )
    features = pd.concat([train, validation, test]).sort_values(DATE_COLUMN)
    report = rolling_origin_backtest(
        features,
        folds=config.backtest.folds,
        horizon_days=config.backtest.horizon_days,
        step_days=config.backtest.step_days,
    )
    write_backtest_report(report)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
