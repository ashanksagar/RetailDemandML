from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FeaturePredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "store": "1",
                "item": "1",
                "day_of_week": "6",
                "month": "1",
                "day_of_month": 1,
                "week_of_year": 52,
                "is_weekend": 1,
                "is_month_start": 1,
                "is_month_end": 0,
                "quarter": "1",
                "year_index": 0,
                "is_us_holiday": "1",
                "days_to_nearest_us_holiday": 0,
                "sales_lag_1": 62,
                "sales_lag_7": 69,
                "sales_lag_14": 59,
                "sales_lag_28": 64,
                "sales_lag_56": 66,
                "sales_rolling_mean_7": 55.1,
                "sales_rolling_mean_28": 51.6,
                "store_historical_mean": 65.6,
                "item_historical_mean": 58.3,
            }
        },
    )

    store: str = Field(description="Store identifier.")
    item: str = Field(description="Item or product identifier.")
    day_of_week: str = Field(examples=["0"])
    month: str = Field(examples=["1"])
    day_of_month: int
    week_of_year: int
    is_weekend: int
    sales_lag_7: float
    sales_lag_14: float
    sales_lag_28: float
    sales_rolling_mean_7: float
    sales_rolling_mean_28: float
    store_historical_mean: float
    item_historical_mean: float


class DemandPredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "store": "1",
                "item": "1",
                "forecast_date": "2023-01-01",
            }
        }
    )

    store: str = Field(examples=["1"], description="Store identifier.")
    item: str = Field(examples=["1"], description="Item or product identifier.")
    forecast_date: str = Field(examples=["2023-01-01"], description="Forecast date in YYYY-MM-DD format.")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 60.3,
                "prediction_p10": None,
                "prediction_p90": None,
                "model_version": "ridge",
                "features_used": {"store": "1", "item": "1", "sales_lag_7": 69},
            }
        }
    )

    prediction: float = Field(description="Non-negative point forecast for expected units sold.")
    prediction_p10: float | None = None
    prediction_p90: float | None = None
    model_version: str = Field(default="local-model", description="Active model name or registry version.")
    features_used: dict[str, Any] | None = Field(default=None, description="Materialized features used by /predict.")


class HealthResponse(BaseModel):
    status: str
    active_model: dict[str, Any] | None = None
    registry_enabled: bool
    model_loaded: bool
    feature_store_loaded: bool
