from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.config import (
    CATEGORICAL_COLUMNS,
    MLFLOW_EXPERIMENT,
    RANDOM_STATE,
    TARGET,
    XGBOOST_MODEL_PATH,
    ensure_directories,
)
from src.features.build_features import get_feature_columns
from src.models.evaluate import regression_metrics


def build_xgboost_pipeline(feature_columns: list[str], params: dict | None = None) -> Pipeline:
    params = params or {}
    categorical = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=params.get("n_estimators", 250),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_path: Path = XGBOOST_MODEL_PATH,
    params: dict | None = None,
    log_mlflow: bool = True,
) -> tuple[Pipeline, dict[str, float]]:
    ensure_directories()
    feature_columns = get_feature_columns(train)
    pipeline = build_xgboost_pipeline(feature_columns, params=params)
    x_train = train[feature_columns]
    y_train = train[TARGET]
    x_val = validation[feature_columns]
    y_val = validation[TARGET]

    if log_mlflow:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="xgboost_demand_forecast"):
            mlflow.log_params(params or {})
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_val)
            metrics = regression_metrics(y_val, predictions)
            mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
    else:
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_val)
        metrics = regression_metrics(y_val, predictions)

    joblib.dump({"model": pipeline, "feature_columns": feature_columns}, model_path)
    return pipeline, metrics
