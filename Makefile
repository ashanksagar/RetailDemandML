.PHONY: setup data data-metadata verify-real-data sample-data eda features-doc feature-importance drift train train-sample evaluate backtest backtest-sample predict-batch register promote registry serve test lint typecheck ci tune tune-sample explain clean

setup:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

sample-data:
	python -m src.data.ingest --sample

data:
	python -m src.data.ingest --kaggle --unzip

data-metadata:
	python -m src.data.ingest --metadata

verify-real-data:
	python -m src.data.verify_real_dataset

eda:
	python -m src.reports.eda

features-doc:
	python -m src.reports.feature_dictionary

feature-importance:
	python -m src.reports.feature_importance

drift:
	python -m src.monitoring.drift

train:
	python scripts/run_pipeline.py --config configs/default.yaml

train-sample:
	python scripts/run_pipeline.py --sample

evaluate:
	python -m src.models.evaluate --predictions data/processed/predictions.csv

backtest:
	python -m src.validation.backtest --config configs/default.yaml

backtest-sample:
	python -m src.validation.backtest --sample

predict-batch:
	python -m src.pipelines.predict_batch --days 28

register:
	python -m src.models.registry register

promote:
	python -m src.models.registry promote

registry:
	python -m src.models.registry status

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest

lint:
	ruff check src scripts tests

typecheck:
	mypy src scripts

ci: lint typecheck test

tune:
	python -m src.models.tune --n-trials 20

tune-sample:
	python -m src.models.tune --n-trials 20 --sample

explain:
	python -m src.models.explain

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['data/processed', 'models', 'mlruns', 'reports']]"
