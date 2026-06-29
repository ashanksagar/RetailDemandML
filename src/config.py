from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"

RAW_TRAIN_PATH = RAW_DATA_DIR / "train.csv"
RAW_TEST_PATH = RAW_DATA_DIR / "test.csv"
CLEAN_PATH = PROCESSED_DATA_DIR / "clean_sales.csv"
FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "predictions.csv"
FORECAST_PATH = PROCESSED_DATA_DIR / "forecast_next_28_days.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
BACKTEST_PATH = REPORTS_DIR / "backtest_results.csv"
MODEL_CARD_PATH = REPORTS_DIR / "model_card.md"
DATA_METADATA_PATH = REPORTS_DIR / "data_metadata.json"
REAL_DATA_VERIFICATION_PATH = REPORTS_DIR / "real_data_verification.json"
REAL_DATA_VERIFICATION_MARKDOWN_PATH = REPORTS_DIR / "real_data_verification.md"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.json"
DRIFT_SUMMARY_PATH = REPORTS_DIR / "drift_summary.csv"
DRIFT_MARKDOWN_PATH = REPORTS_DIR / "drift_report.md"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
MODEL_PATH = MODELS_DIR / "production_model.joblib"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
BASELINE_PATH = MODELS_DIR / "baseline_predictions.csv"
FEATURE_STORE_PATH = MODELS_DIR / "feature_store.joblib"
REGISTRY_DIR = MODELS_DIR / "registry"
REGISTRY_PATH = REGISTRY_DIR / "model_registry.json"
PROMOTIONS_PATH = REGISTRY_DIR / "promotions.jsonl"
REGISTRY_PRODUCTION_DIR = REGISTRY_DIR / "production"
REGISTRY_PRODUCTION_MODEL_PATH = REGISTRY_PRODUCTION_DIR / "production_model.joblib"

TARGET = "sales"
DATE_COLUMN = "date"
GROUP_COLUMNS = ["store", "item"]
CATEGORICAL_COLUMNS = [
    "store",
    "item",
    "day_of_week",
    "month",
    "quarter",
    "is_us_holiday",
]
OPTIONAL_BUSINESS_COLUMNS = ["price", "promotion_flag", "stockout_flag", "inventory_on_hand"]
LAG_PERIODS = [1, 7, 14, 28, 56]
ROLLING_WINDOWS = [7, 14, 28, 56]
VALIDATION_DAYS = 90
TEST_DAYS = 90
RANDOM_STATE = 42
MLFLOW_EXPERIMENT = "retail-demand-forecasting"


def ensure_directories() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        REGISTRY_DIR,
        REGISTRY_PRODUCTION_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


class DataConfig(BaseModel):
    validation_days: int = VALIDATION_DAYS
    test_days: int = TEST_DAYS


class FeatureConfig(BaseModel):
    lags: list[int] = Field(default_factory=lambda: LAG_PERIODS.copy())
    rolling_windows: list[int] = Field(default_factory=lambda: ROLLING_WINDOWS.copy())
    enable_hierarchical_features: bool = True
    enable_demand_pattern_features: bool = True


class XGBoostConfig(BaseModel):
    n_estimators: int = 250
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9


class ModelConfig(BaseModel):
    forecast_horizon: int = 1
    random_state: int = RANDOM_STATE
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)


class BacktestConfig(BaseModel):
    folds: int = 3
    horizon_days: int = 28
    step_days: int = 28


class RegistryConfig(BaseModel):
    primary_metric: str = "rmse"
    minimum_relative_improvement: float = 0.0
    required_backtest_folds: int = 1
    max_slice_rmse_regression: float = 0.10


class MonitoringConfig(BaseModel):
    psi_threshold: float = 0.20
    mean_shift_threshold: float = 2.0
    missing_rate_shift_threshold: float = 0.05
    new_category_rate_threshold: float = 0.01


class ProjectConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


def load_project_config(path: Path | None = None) -> ProjectConfig:
    config_path = path or CONFIGS_DIR / "default.yaml"
    if not config_path.exists():
        return ProjectConfig()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return ProjectConfig.model_validate(raw)
