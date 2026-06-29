import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

from src.config import (
    BACKTEST_PATH,
    BASELINE_PATH,
    CLEAN_PATH,
    DATA_METADATA_PATH,
    FEATURES_PATH,
    MODEL_PATH,
    MODEL_COMPARISON_PATH,
    METRICS_PATH,
    MLFLOW_EXPERIMENT,
    PREDICTIONS_PATH,
    TEST_DAYS,
    VALIDATION_DAYS,
    ensure_directories,
    load_project_config,
    ProjectConfig,
)
from src.data.ingest import generate_sample_data, load_raw_sales, write_dataset_metadata
from src.data.preprocess import chronological_split, normalize_schema
from src.data.validate import validate_sales_data
from src.features.build_features import build_features, get_feature_columns
from src.features.feature_store import build_feature_store
from src.models.compare import compare_sklearn_models, write_model_comparison
from src.models.compare import sklearn_model_candidates
from src.models.evaluate import regression_metrics, write_metrics
from src.models.registry import promote_candidate, register_candidate
from src.models.reports import residual_prediction_interval, sliced_metrics, write_model_card
from src.models.train_baseline import evaluate_baseline
from src.models.train_xgboost import train_xgboost
from src.validation.backtest import rolling_origin_backtest, write_backtest_report
from src.reports.feature_dictionary import write_feature_dictionary
from src.reports.feature_importance import write_feature_importance


def prepare_datasets(
    sample: bool = False, config: ProjectConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_directories()
    config = config or load_project_config()
    if sample:
        generate_sample_data()
    write_dataset_metadata()
    raw = load_raw_sales()
    clean = normalize_schema(raw)
    validate_sales_data(clean, require_daily_continuity=True).raise_for_errors()
    clean.to_csv(CLEAN_PATH, index=False)
    features = build_features(clean)
    features.to_csv(FEATURES_PATH, index=False)
    build_feature_store(clean)
    return chronological_split(
        features,
        validation_days=config.data.validation_days or VALIDATION_DAYS,
        test_days=config.data.test_days or TEST_DAYS,
    )


def run_pipeline(sample: bool = False, config_path: Path | None = None) -> dict[str, dict[str, float]]:
    config = load_project_config(config_path)
    train, validation, test = prepare_datasets(sample=sample, config=config)

    baseline_predictions, baseline_metrics = evaluate_baseline(pd.concat([train, validation]), test)
    baseline_predictions.to_csv(BASELINE_PATH, index=False)

    params = config.model.xgboost.model_dump()
    xgboost_model, validation_metrics = train_xgboost(train, validation, params=params)
    feature_columns = get_feature_columns(test)

    comparison = compare_sklearn_models(train, validation)
    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                [
                    {"model": "seasonal_naive_7", **baseline_metrics},
                    {"model": "xgboost", **validation_metrics},
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values("rmse")
    write_model_comparison(comparison)

    trainable_models = set(sklearn_model_candidates(feature_columns)).union({"xgboost"})
    selected_model_name = next(
        name for name in comparison["model"].tolist() if name in trainable_models
    )
    production_train = pd.concat([train, validation]).sort_values("date")
    if selected_model_name == "xgboost":
        production_model = xgboost_model
        validation_predictions = xgboost_model.predict(validation[feature_columns])
    else:
        production_model = sklearn_model_candidates(feature_columns)[selected_model_name]
        production_model.fit(production_train[feature_columns], production_train["sales"])
        validation_model = sklearn_model_candidates(feature_columns)[selected_model_name]
        validation_model.fit(train[feature_columns], train["sales"])
        validation_predictions = validation_model.predict(validation[feature_columns])

    import joblib

    joblib.dump(
        {
            "model": production_model,
            "feature_columns": feature_columns,
            "model_name": selected_model_name,
        },
        MODEL_PATH,
    )
    write_feature_importance(MODEL_PATH)

    predictions = test.copy()
    predictions["prediction"] = production_model.predict(test[feature_columns])
    intervals = residual_prediction_interval(
        validation["sales"],
        pd.Series(validation_predictions),
        predictions["prediction"],
    )
    predictions["prediction_p10"] = intervals["prediction_p10"].clip(lower=0)
    predictions["prediction_p90"] = intervals["prediction_p90"].clip(lower=0)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    test_metrics = regression_metrics(predictions["sales"], predictions["prediction"])

    full_features = pd.concat([train, validation, test]).sort_values("date")
    backtest = rolling_origin_backtest(
        full_features,
        folds=config.backtest.folds,
        horizon_days=config.backtest.horizon_days,
        step_days=config.backtest.step_days,
    )
    write_backtest_report(backtest)
    sliced = sliced_metrics(predictions)
    sliced.to_csv("reports/sliced_metrics.csv", index=False)

    all_metrics = {
        "baseline_test": baseline_metrics,
        "xgboost_validation": validation_metrics,
        "production_test": test_metrics,
        "selected_model": {"name": selected_model_name},
    }
    write_feature_dictionary()
    write_metrics(all_metrics, METRICS_PATH)
    candidate = register_candidate()
    promotion = promote_candidate(candidate_version=candidate["version"])
    write_model_card(train, validation, test, all_metrics)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="pipeline_artifacts"):
        mlflow.log_params(
            {
                "validation_days": config.data.validation_days,
                "test_days": config.data.test_days,
                "backtest_folds": config.backtest.folds,
                "backtest_horizon_days": config.backtest.horizon_days,
            }
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_param("registered_model_version", candidate["version"])
        mlflow.log_param("promotion_decision", promotion["promoted"])
        for artifact in [
            METRICS_PATH,
            MODEL_COMPARISON_PATH,
            BACKTEST_PATH,
            DATA_METADATA_PATH,
            Path("reports/sliced_metrics.csv"),
            Path("reports/feature_dictionary.md"),
        ]:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Generate sample data before training")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    metrics = run_pipeline(sample=args.sample, config_path=args.config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
