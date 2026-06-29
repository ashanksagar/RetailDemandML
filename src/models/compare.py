from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLUMNS, MODEL_COMPARISON_PATH, RANDOM_STATE, TARGET, ensure_directories
from src.features.build_features import get_feature_columns
from src.models.evaluate import regression_metrics


def _preprocessor(feature_columns: list[str], scale_numeric: bool = False) -> ColumnTransformer:
    categorical = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", numeric_transformer, numeric),
        ]
    )


def sklearn_model_candidates(feature_columns: list[str]) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline(
            [
                ("preprocessor", _preprocessor(feature_columns, scale_numeric=True)),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", _preprocessor(feature_columns)),
                ("model", HistGradientBoostingRegressor(max_iter=150, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", _preprocessor(feature_columns)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def moving_average_predictions(df: pd.DataFrame, window: int = 7) -> pd.Series:
    column = f"sales_rolling_mean_{window}"
    if column not in df:
        raise ValueError(f"Missing {column}; build rolling features before comparison.")
    return df[column].fillna(df[column].mean())


def compare_sklearn_models(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    feature_columns = get_feature_columns(train)
    rows = []
    x_train = train[feature_columns]
    y_train = train[TARGET]
    x_test = test[feature_columns]
    y_test = test[TARGET]

    moving_average = moving_average_predictions(test)
    rows.append({"model": "moving_average_7", **regression_metrics(y_test, moving_average)})

    for name, model in sklearn_model_candidates(feature_columns).items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rows.append({"model": name, **regression_metrics(y_test, predictions)})

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def write_model_comparison(comparison: pd.DataFrame, path: Path = MODEL_COMPARISON_PATH) -> None:
    ensure_directories()
    comparison.to_csv(path, index=False)
