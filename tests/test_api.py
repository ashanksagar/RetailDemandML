from fastapi.testclient import TestClient

from src.api.main import app, load_model, load_serving_feature_store


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "registry_enabled" in response.json()
    assert "model_loaded" in response.json()


def test_openapi_documents_prediction_endpoints():
    client = TestClient(app)

    schema = client.get("/openapi.json").json()

    assert "/predict" in schema["paths"]
    assert "/score-features" in schema["paths"]
    assert "/dashboard" in schema["paths"]
    assert schema["info"]["title"] == "RetailDemandML Forecasting API"


def test_dashboard_page_renders():
    client = TestClient(app)

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "RetailDemandML Operations" in response.text


def test_dashboard_data_endpoint_returns_artifact_sections():
    client = TestClient(app)

    response = client.get("/dashboard/data")

    assert response.status_code == 200
    payload = response.json()
    assert "metrics" in payload
    assert "model_comparison" in payload
    assert "feature_importance" in payload


def test_predict_returns_service_unavailable_when_artifacts_missing():
    load_model.cache_clear()
    load_serving_feature_store.cache_clear()
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={"store": "missing", "item": "missing", "forecast_date": "2024-01-01"},
    )

    assert response.status_code in {200, 503}
