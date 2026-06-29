# RetailDemandML ML Finish Implementation Plan

## Goal

Finish the machine learning portion of RetailDemandML so the project reads like a serious production forecasting system built on real historical retail data. The final result should make a senior ML engineer see careful data work, thoughtful forecasting validation, strong diagnostics, explainable modeling, and a reproducible model lifecycle.

## Scope

This plan covers:

1. Real Kaggle dataset integration.
2. EDA and rich visualizations.
3. Final feature engineering.
4. Model registry and promotion workflow.

Implementation status: completed in the current workspace. Remaining polish is mainly notebook narrative and external business-signal data sources.

Out of scope for this phase:

- Cloud deployment.
- Streaming ingestion.
- Real-time feature store infrastructure.
- External monitoring services.

## Phase 1: Real Kaggle Dataset

### Objective

Replace sample-data-first development with a reproducible real-data workflow using the Kaggle Store Item Demand Forecasting dataset while preserving sample mode for fast local tests.

### Implementation Tasks

- Improve `src/data/ingest.py` Kaggle support:
  - Download the Kaggle competition archive when `kaggle` CLI credentials exist.
  - Automatically unzip `train.csv` and optional `test.csv`.
  - Validate downloaded file names and expected columns.
  - Write dataset metadata with source, row count, date range, store count, item count, SHA256, and creation time.

- Add a dataset preparation CLI:
  - `python -m src.data.ingest --kaggle --unzip`
  - `python -m src.data.ingest --metadata`
  - `python -m src.data.ingest --sample`

- Add Makefile commands:
  - `make data`
  - `make data-metadata`

- Add validation checks:
  - Required columns.
  - No negative sales.
  - Unique `(date, store, item)`.
  - Daily continuity by store/item.
  - Expected date range.
  - Expected count of stores and items.

- Update docs:
  - Add Kaggle credential setup instructions.
  - Explain exact dataset used.
  - Explain fallback sample mode.

### Files To Touch

- `src/data/ingest.py`
- `src/data/validate.py`
- `src/config.py`
- `Makefile`
- `README.md`
- `PROJECT_PLAN.md`
- `TASKS.md`

### Acceptance Criteria

- Running `make data` downloads or prepares the Kaggle dataset.
- `reports/data_metadata.json` contains source, hash, rows, columns, date range, store count, and item count.
- `make train` works on the real Kaggle dataset.
- `make test` still works without Kaggle credentials.

## Phase 2: EDA

### Objective

Create a polished EDA layer that demonstrates real understanding of retail demand behavior, time-series structure, and modeling risks.

### Implementation Tasks

- Expand `notebooks/01_eda.ipynb` into a real analysis notebook:
  - Dataset overview.
  - Missingness and duplicate checks.
  - Date coverage.
  - Store/item counts.
  - Sales distribution.
  - Store-level demand variation.
  - Item-level demand variation.
  - Seasonality by day of week, month, and year.
  - Trend examples for top stores/items.
  - Intermittent demand analysis.
  - Autocorrelation and lag signal checks.
  - Leakage risk notes.
  - Modeling implications.

- Add a script version for reproducibility:
  - `src/reports/eda.py`
  - `python -m src.reports.eda`

- Save EDA artifacts under:
  - `reports/figures/`
  - `reports/eda_summary.json`
  - `reports/eda_report.md`

### Rich Visualizations

Generate clean, resume-quality charts:

- Total daily sales over time.
- Sales by day of week.
- Sales by month.
- Top items by average demand.
- Store demand comparison.
- Heatmap of store/item average demand.
- Rolling average examples for selected store/item pairs.
- Lag correlation plot.
- Forecast split timeline.
- Missingness and duplicate summary.
- Model comparison chart after training.
- Backtest fold metric chart.
- SHAP feature importance chart.

Use consistent styling:

- Clear titles.
- Axis labels.
- High DPI.
- Saved PNG files.
- No notebook-only charts.

### Files To Add

- `src/reports/__init__.py`
- `src/reports/eda.py`
- `src/reports/plots.py`
- `reports/eda_report.md`
- Updated `notebooks/01_eda.ipynb`

### Acceptance Criteria

- `python -m src.reports.eda` runs headlessly.
- At least 10 visual artifacts are saved to `reports/figures/`.
- `reports/eda_report.md` summarizes findings and modeling implications.
- README includes a short EDA preview section.

## Phase 3: Final Feature Engineering

### Objective

Move from good MVP features to a final forecasting feature set that captures demand structure while staying leakage-safe.

### Feature Groups

#### Calendar Features

- Day of week.
- Month.
- Week of year.
- Day of month.
- Weekend flag.
- US holiday flag.
- Days to nearest US holiday.
- Month start/end.
- Quarter.
- Year trend index.

#### Lag Features

- Sales lag 1, 7, 14, 28, 56.
- Same day previous week.
- Same day previous 4 weeks average.
- Lag deltas:
  - `sales_lag_7 - sales_lag_14`
  - `sales_lag_7 / sales_lag_28`

#### Rolling Features

For each store/item:

- Rolling mean: 7, 14, 28, 56.
- Rolling median: 7, 28.
- Rolling std: 7, 28.
- Rolling min/max: 7, 28.

All rolling features must use shifted targets only.

#### Hierarchical Aggregate Features

Leakage-safe expanding or shifted rolling aggregates:

- Store historical mean.
- Item historical mean.
- Store-item historical mean.
- Store/day-of-week historical mean.
- Item/day-of-week historical mean.
- Store/month historical mean.
- Item/month historical mean.

#### Demand Pattern Features

- Intermittency score.
- Recent zero-sales count.
- Coefficient of variation.
- Recent trend slope.
- Sales momentum ratio.

#### Optional External-Like Features

The Kaggle dataset does not include price, promotion, stockout, or inventory. Add documented placeholder support:

- `price`
- `promotion_flag`
- `stockout_flag`
- `inventory_on_hand`

These should be optional columns, validated if present, and ignored if absent.

### Implementation Tasks

- Refactor `src/features/build_features.py` into clearly separated feature builders:
  - `calendar_features`
  - `lag_features`
  - `rolling_features`
  - `hierarchical_features`
  - `demand_pattern_features`
  - `optional_business_features`

- Add feature configuration in YAML:
  - lags.
  - rolling windows.
  - hierarchy toggles.
  - optional feature toggles.

- Add feature documentation:
  - `reports/feature_dictionary.md`

- Add tests for leakage:
  - Rolling features do not include current target.
  - Expanding features are shifted.
  - Split boundaries do not leak future rows.

### Files To Touch

- `src/features/build_features.py`
- `src/features/feature_store.py`
- `src/config.py`
- `configs/default.yaml`
- `configs/sample.yaml`
- `tests/test_features.py`

### Files To Add

- `reports/feature_dictionary.md`
- Possibly `src/features/transformers.py`

### Acceptance Criteria

- Feature build remains deterministic.
- Feature build works on sample data and real Kaggle data.
- Leakage tests pass.
- Model comparison improves or explains why added features are neutral.
- Feature dictionary documents every engineered feature.

## Phase 4: Model Registry

### Objective

Add a practical local MLflow model registry workflow that tracks candidate models, promotes a selected production model, and makes serving read from the promoted artifact.

### Implementation Tasks

- Add registry module:
  - `src/models/registry.py`

- Registry responsibilities:
  - Register trained candidate model.
  - Store model name, version, metrics, params, feature list, dataset hash, and run ID.
  - Compare candidate against current production model.
  - Promote only if candidate beats configured metric threshold.
  - Write a promotion record.

- Add local registry artifacts:
  - `models/registry/model_registry.json`
  - `models/registry/promotions.jsonl`
  - `models/registry/production/production_model.joblib`

- Add commands:
  - `make register`
  - `make promote`
  - `make registry`

- Add configurable promotion rules:
  - Primary metric: RMSE or SMAPE.
  - Minimum relative improvement.
  - Required backtest fold count.
  - Maximum allowed metric regression on any critical slice.

- Update API:
  - Load from promoted registry artifact by default.
  - Include model name, version, promotion time, and dataset hash in `/health`.
  - Include model version in prediction responses.

- Update model card:
  - Include registry version.
  - Include promotion decision.
  - Include challenger vs champion metrics.

### Files To Add

- `src/models/registry.py`
- `tests/test_registry.py`
- `models/registry/.gitkeep`

### Files To Touch

- `scripts/run_pipeline.py`
- `src/api/main.py`
- `src/api/schemas.py`
- `src/models/reports.py`
- `src/config.py`
- `Makefile`
- `README.md`

### Acceptance Criteria

- Training produces a candidate model.
- Registry can compare challenger vs champion.
- Promotion writes a versioned registry record.
- API serves the promoted model.
- `/health` reports active model metadata.
- Tests cover promotion and non-promotion paths.

## Phase 5: Final Reporting Polish

### Objective

Make the project presentation feel complete and credible.

### Implementation Tasks

- Update README with:
  - Real Kaggle dataset stats.
  - EDA highlights.
  - Model comparison table.
  - Backtest summary.
  - Final selected model.
  - Registry/promotion summary.
  - Key charts embedded or linked.

- Update `MODEL_CARD.md` or generated `reports/model_card.md`:
  - Intended use.
  - Dataset version/hash.
  - Train/validation/test windows.
  - Feature groups.
  - Candidate models.
  - Backtest results.
  - Slice performance.
  - Limitations.
  - Monitoring plan.

- Add a concise architecture image or Mermaid diagram.

### Acceptance Criteria

- A reviewer can understand the data, model, validation method, and production path from README alone.
- Reports are generated by code, not manually copied.
- Resume bullets reflect measured results on real data.

## Recommended Build Order

1. Real Kaggle dataset ingestion and metadata.
2. EDA script and saved figures.
3. Final feature engineering and leakage tests.
4. Retrain and compare all models on real data.
5. Generate final visual reports.
6. Implement model registry and promotion.
7. Update API to serve promoted model metadata.
8. Update README, model card, architecture, and tasks.
9. Run final verification suite.

## Final Verification Checklist

Run:

```bash
make setup
make data
make train
make backtest
make predict-batch
make explain
make lint
make typecheck
make test
```

Expected final artifacts:

```text
data/raw/train.csv
reports/data_metadata.json
reports/eda_summary.json
reports/eda_report.md
reports/figures/*.png
reports/feature_dictionary.md
reports/model_comparison.csv
reports/backtest_results.csv
reports/sliced_metrics.csv
reports/model_card.md
models/registry/model_registry.json
models/registry/promotions.jsonl
models/registry/production/production_model.joblib
```

## Definition Of Done

The ML work is finished when:

- The real Kaggle dataset is used by default for full training.
- EDA is reproducible and generates polished visual artifacts.
- Feature engineering is documented, tested, and leakage-safe.
- Models are compared with chronological validation and rolling backtests.
- A production model is registered and promoted through an explicit workflow.
- The API serves the promoted model and exposes model metadata.
- README and model card show real dataset results and measured performance.
- `make lint`, `make typecheck`, and `make test` pass.
