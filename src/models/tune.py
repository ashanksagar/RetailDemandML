import argparse

import optuna

from scripts.run_pipeline import prepare_datasets
from src.models.train_xgboost import train_xgboost


def objective(trial: optuna.Trial) -> float:
    train, validation, _test = prepare_datasets(sample=False)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
    }
    _model, metrics = train_xgboost(train, validation, params=params, log_mlflow=True)
    return metrics["rmse"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--sample", action="store_true", help="Generate sample data first if raw data is missing")
    args = parser.parse_args()
    if args.sample:
        from src.data.ingest import generate_sample_data

        generate_sample_data()
    study = optuna.create_study(direction="minimize", study_name="retail-demand-xgboost")
    study.optimize(objective, n_trials=args.n_trials)
    print(f"Best RMSE: {study.best_value:.4f}")
    print(study.best_params)


if __name__ == "__main__":
    main()
