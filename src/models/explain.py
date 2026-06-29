import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import FEATURES_PATH, FIGURES_DIR, MODEL_PATH, ensure_directories


def generate_shap_summary(model_path: Path = MODEL_PATH, features_path: Path = FEATURES_PATH, limit: int = 500) -> Path:
    import matplotlib.pyplot as plt
    import shap

    ensure_directories()
    artifact = joblib.load(model_path)
    pipeline = artifact["model"]
    feature_columns = artifact["feature_columns"]
    df = pd.read_csv(features_path).head(limit)
    x = df[feature_columns]
    transformed = pipeline.named_steps["preprocessor"].transform(x)
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)
    output_path = FIGURES_DIR / "shap_summary.png"
    shap.summary_plot(values, transformed, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    path = generate_shap_summary(limit=args.limit)
    print(f"Wrote SHAP summary to {path}")


if __name__ == "__main__":
    main()
