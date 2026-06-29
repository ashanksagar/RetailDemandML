import json

import joblib

from src.models import registry


def test_registry_promotes_first_candidate(tmp_path, monkeypatch):
    model_path = tmp_path / "production_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    metadata_path = tmp_path / "data_metadata.json"
    registry_path = tmp_path / "model_registry.json"
    promotions_path = tmp_path / "promotions.jsonl"
    promoted_model_path = tmp_path / "registry" / "production" / "production_model.joblib"

    promoted_model_path.parent.mkdir(parents=True)
    joblib.dump({"model_name": "ridge"}, model_path)
    metrics_path.write_text(
        json.dumps(
            {
                "selected_model": {"name": "ridge"},
                "production_test": {"rmse": 5.0, "mae": 4.0},
            }
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(json.dumps({"sha256": "abc", "rows": 10}), encoding="utf-8")

    monkeypatch.setattr(registry, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(registry, "PROMOTIONS_PATH", promotions_path)
    monkeypatch.setattr(registry, "REGISTRY_PRODUCTION_MODEL_PATH", promoted_model_path)

    candidate = registry.register_candidate(
        model_path=model_path,
        metrics_path=metrics_path,
        data_metadata_path=metadata_path,
    )
    decision = registry.promote_candidate(candidate_version=candidate["version"])

    assert decision["promoted"] is True
    assert promoted_model_path.exists()
    assert registry.active_production_metadata()["version"] == candidate["version"]
