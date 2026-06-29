import argparse

import joblib
import pandas as pd

from src.config import FORECAST_PATH, MODEL_PATH, ensure_directories
from src.features.feature_store import build_online_feature_row, load_feature_store


def forecast_next_days(days: int = 28) -> pd.DataFrame:
    ensure_directories()
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    feature_store = load_feature_store()
    last_date = pd.Timestamp(feature_store["last_date"])
    rows = []

    for store, item in sorted(feature_store["states"]):
        for offset in range(1, days + 1):
            forecast_date = last_date + pd.Timedelta(days=offset)
            features = build_online_feature_row(store, item, forecast_date, feature_store)
            prediction = float(model.predict(pd.DataFrame([features])[feature_columns])[0])
            rows.append(
                {
                    "store": store,
                    "item": item,
                    "forecast_date": forecast_date.date(),
                    "horizon": offset,
                    "prediction": max(0.0, prediction),
                }
            )

    forecast = pd.DataFrame(rows)
    forecast.to_csv(FORECAST_PATH, index=False)
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=28)
    args = parser.parse_args()
    forecast = forecast_next_days(days=args.days)
    print(forecast.head(20).to_string(index=False))
    print(f"Wrote {len(forecast):,} forecasts to {FORECAST_PATH}")


if __name__ == "__main__":
    main()
