# RetailDemandML Project Plan

## Objective

RetailDemandML is a production-style demand forecasting platform for retail sales. It demonstrates the full machine learning lifecycle: data ingestion, time-aware preprocessing, feature engineering, model training, experiment tracking, evaluation, explainability, inference serving, containerization, and automated tests.

## Dataset Strategy

The MVP uses the public Kaggle Store Item Demand Forecasting Challenge dataset when available:

- `train.csv`
- `test.csv` optional for competition-style inference

Because Kaggle downloads require user credentials, the project supports two paths:

1. Manual dataset placement in `data/raw/train.csv`.
2. Local development sample generation with `make sample-data`.

The sample data is synthetic but shaped like the public dataset so the full pipeline can run on a normal laptop without external credentials.

## Milestones

### Milestone 1: Foundation

- Create project structure.
- Add configuration, dependency metadata, Makefile, Dockerfile, and docker-compose.
- Document architecture and task status.

### Milestone 2: Data Pipeline

- Implement raw data validation.
- Normalize columns and data types.
- Create chronological train/validation/test splits.
- Support sample mode for fast local iteration.

### Milestone 3: Feature Engineering

- Add calendar features.
- Add lag features.
- Add rolling mean features.
- Add store/item aggregate history features without using future targets.

### Milestone 4: Modeling

- Implement seasonal naive baseline.
- Implement XGBoost model.
- Log params, metrics, and artifacts to MLflow.
- Save production model artifact.

### Milestone 5: Evaluation and Explainability

- Compute MAE, RMSE, MAPE, and SMAPE.
- Save metrics to `reports/metrics.json`.
- Generate SHAP summary artifacts when dependencies and data permit.

### Milestone 6: Serving

- Provide FastAPI prediction endpoint.
- Use Pydantic request and response schemas.
- Load trained model artifacts from disk.

### Milestone 7: Quality

- Add focused pytest coverage for preprocessing, feature engineering, metrics, and API health.
- Keep functions modular and testable.

## Implementation Status

Implemented in the current MVP:

- Project skeleton.
- Manual or synthetic dataset ingestion.
- Preprocessing and chronological split.
- Time series feature engineering.
- Baseline and XGBoost training.
- Evaluation metrics.
- MLflow logging hooks.
- FastAPI service.
- Docker and Makefile commands.
- Basic tests.
- Data validation, metadata, and Pandera schema support.
- Rolling-origin backtesting.
- Model comparison and production model selection.
- Feature-state artifact for business-input API serving.
- Batch next-28-day forecasts.
- Sliced metrics, residual intervals, and model card.
- CI, linting, and static type checking.
- Kaggle CLI download/unzip workflow with dataset metadata.
- Real Kaggle dataset verification report for public-result readiness.
- Reproducible EDA script with saved figures and markdown report.
- EDA notebook narrative pointing to generated artifacts.
- Expanded final feature engineering with feature dictionary.
- Local model registry with promotion decisions and API metadata.
- Drift monitoring report with configurable thresholds.
- Production API documentation with examples and model health metadata.
- Docker hardening and broader CI workflow.

Pending future enhancements:

- Drift monitoring dashboard.
- External alert routing for drift warnings.
- Full verified Kaggle training run after credentials are available.
- Real external promotion, price, stockout, and inventory sources.
