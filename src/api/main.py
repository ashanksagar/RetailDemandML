import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import DemandPredictionRequest, FeaturePredictionRequest, HealthResponse, PredictionResponse
from src.api.dashboard import router as dashboard_router
from src.config import FEATURE_STORE_PATH, MODEL_PATH, REGISTRY_PRODUCTION_MODEL_PATH
from src.features.feature_store import build_online_feature_row, load_feature_store
from src.models.registry import active_production_metadata

app = FastAPI(
    title="RetailDemandML Forecasting API",
    version="0.2.0",
    summary="Production-style retail demand forecasting service.",
    description=(
        "RetailDemandML serves daily store/item demand forecasts from a promoted model registry "
        "artifact. The business endpoint accepts store, item, and forecast date, then materializes "
        "the lag, rolling, holiday, and historical aggregate features from the saved feature-state artifact."
    ),
    contact={"name": "RetailDemandML"},
    license_info={"name": "MIT"},
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "health", "description": "Service and model registry status."},
        {"name": "prediction", "description": "Business and feature-level prediction endpoints."},
        {"name": "dashboard", "description": "Internal ML operations dashboard."},
    ],
)

app.include_router(dashboard_router)


@lru_cache(maxsize=1)
def load_model() -> dict:
    default_model_path = REGISTRY_PRODUCTION_MODEL_PATH if REGISTRY_PRODUCTION_MODEL_PATH.exists() else MODEL_PATH
    model_path = Path(os.getenv("MODEL_PATH", default_model_path))
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run `make train` first.")
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_serving_feature_store() -> dict:
    try:
        return load_feature_store(FEATURE_STORE_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc} Run `make train` before serving business predictions.") from exc


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Check service and model status",
)
def health() -> HealthResponse:
    metadata = active_production_metadata()
    model_loaded = True
    feature_store_loaded = True
    try:
        load_model()
    except FileNotFoundError:
        model_loaded = False
    try:
        load_serving_feature_store()
    except FileNotFoundError:
        feature_store_loaded = False
    return HealthResponse(
        status="ok",
        active_model=metadata,
        registry_enabled=REGISTRY_PRODUCTION_MODEL_PATH.exists(),
        model_loaded=model_loaded,
        feature_store_loaded=feature_store_loaded,
    )


@app.post(
    "/score-features",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Score a fully materialized feature row",
    description=(
        "Low-level scoring endpoint for debugging or external feature-store integrations. "
        "Most application callers should use /predict instead."
    ),
)
def score_features(request: FeaturePredictionRequest) -> PredictionResponse:
    try:
        artifact = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = pd.DataFrame([request.model_dump()])
    feature_columns = artifact["feature_columns"]
    missing = set(feature_columns).difference(row.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {sorted(missing)}")

    prediction = float(artifact["model"].predict(row[feature_columns])[0])
    return PredictionResponse(
        prediction=max(0.0, prediction),
        model_version=str(artifact.get("model_name", "local-model")),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict demand from business inputs",
    description=(
        "Predict daily demand for a store/item/date. The API builds serving-time features from "
        "the saved feature-state artifact to reduce training-serving skew."
    ),
)
def predict(request: DemandPredictionRequest) -> PredictionResponse:
    try:
        artifact = load_model()
        feature_store = load_serving_feature_store()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    features = build_online_feature_row(
        request.store,
        request.item,
        request.forecast_date,
        feature_store,
    )
    feature_columns = artifact["feature_columns"]
    row = pd.DataFrame([features])
    prediction = float(artifact["model"].predict(row[feature_columns])[0])
    return PredictionResponse(
        prediction=max(0.0, prediction),
        model_version=str(artifact.get("model_name", "local-model")),
        features_used=features,
    )
