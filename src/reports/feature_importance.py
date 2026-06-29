import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import (
    CATEGORICAL_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    FEATURES_PATH,
    MODEL_PATH,
    TARGET,
    ensure_directories,
)


def _feature_names(model, fallback_columns: list[str]) -> list[str]:
    preprocessor = model.named_steps.get("preprocessor") if hasattr(model, "named_steps") else None
    if preprocessor is None:
        return fallback_columns
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return fallback_columns


def _importance_values(model) -> np.ndarray | None:
    estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_).ravel()
        return np.abs(values)
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_).ravel()
    return None


def write_feature_importance(
    model_path: Path = MODEL_PATH,
    output_path: Path = FEATURE_IMPORTANCE_PATH,
    top_n: int | None = None,
    features_path: Path = FEATURES_PATH,
    permutation_limit: int = 2000,
) -> pd.DataFrame:
    ensure_directories()
    artifact = joblib.load(model_path)
    model = artifact["model"]
    fallback_columns = artifact.get("feature_columns", [])
    values = _importance_values(model)
    names = _feature_names(model, fallback_columns)
    if values is None:
        if not features_path.exists() or not fallback_columns:
            importance = pd.DataFrame(columns=["feature", "importance", "rank"])
            importance.to_csv(output_path, index=False)
            return importance
        data = pd.read_csv(features_path).tail(permutation_limit)
        for column in CATEGORICAL_COLUMNS:
            if column in data.columns:
                data[column] = data[column].astype(str)
        result = permutation_importance(
            model,
            data[fallback_columns],
            data[TARGET],
            n_repeats=3,
            random_state=42,
            scoring="neg_root_mean_squared_error",
        )
        names = fallback_columns
        values = np.maximum(result.importances_mean, 0)
    if len(names) != len(values):
        names = [f"feature_{index}" for index in range(len(values))]
    importance = (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance["rank"] = importance.index + 1
    if top_n is not None:
        importance = importance.head(top_n)
    importance.to_csv(output_path, index=False)
    return importance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=None)
    args = parser.parse_args()
    importance = write_feature_importance(top_n=args.top_n)
    print(importance.head(20).to_string(index=False))
    print(f"Wrote feature importance to {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
