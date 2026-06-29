from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import (
    DATA_METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PROMOTIONS_PATH,
    REGISTRY_PATH,
    REGISTRY_PRODUCTION_MODEL_PATH,
    ensure_directories,
    load_project_config,
)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    ensure_directories()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_registry() -> dict:
    return _read_json(REGISTRY_PATH, {"models": [], "production_version": None})


def active_production_metadata() -> dict | None:
    registry = load_registry()
    production_version = registry.get("production_version")
    if production_version is None:
        return None
    for model in registry.get("models", []):
        if model.get("version") == production_version:
            return model
    return None


def latest_candidate() -> dict | None:
    registry = load_registry()
    candidates = registry.get("models", [])
    return candidates[-1] if candidates else None


def register_candidate(
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    data_metadata_path: Path = DATA_METADATA_PATH,
) -> dict:
    ensure_directories()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact at {model_path}. Run `make train` first.")
    metrics = _read_json(metrics_path, {})
    data_metadata = _read_json(data_metadata_path, {})
    registry = load_registry()
    version = len(registry.get("models", [])) + 1
    selected = metrics.get("selected_model", {})
    candidate = {
        "version": version,
        "name": selected.get("name", "production_model"),
        "artifact_path": str(model_path),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics.get("production_test", {}),
        "validation_metrics": metrics.get("xgboost_validation", {}),
        "baseline_metrics": metrics.get("baseline_test", {}),
        "dataset_sha256": data_metadata.get("sha256"),
        "dataset_rows": data_metadata.get("rows"),
        "dataset_date_range": data_metadata.get("date_range"),
        "status": "candidate",
    }
    registry.setdefault("models", []).append(candidate)
    _write_json(REGISTRY_PATH, registry)
    return candidate


def _metric_value(model: dict | None, metric: str) -> float | None:
    if model is None:
        return None
    value = model.get("metrics", {}).get(metric)
    return float(value) if value is not None else None


def should_promote(candidate: dict, champion: dict | None, metric: str, min_improvement: float) -> tuple[bool, str]:
    candidate_value = _metric_value(candidate, metric)
    champion_value = _metric_value(champion, metric)
    if candidate_value is None:
        return False, f"candidate missing metric {metric}"
    if champion_value is None:
        return True, "no current production champion"
    if champion is None:
        return True, "no current production champion"
    if candidate.get("dataset_sha256") and candidate.get("dataset_sha256") != champion.get("dataset_sha256"):
        return True, "candidate was trained on a new verified dataset version"
    improvement = (champion_value - candidate_value) / max(champion_value, 1e-8)
    if improvement > min_improvement:
        return True, f"{metric} improved by {improvement:.2%}"
    return False, f"{metric} improvement {improvement:.2%} below threshold {min_improvement:.2%}"


def promote_candidate(
    candidate_version: int | None = None,
    primary_metric: str | None = None,
    min_improvement: float | None = None,
) -> dict:
    ensure_directories()
    config = load_project_config()
    metric = primary_metric or config.registry.primary_metric
    threshold = (
        config.registry.minimum_relative_improvement
        if min_improvement is None
        else min_improvement
    )
    registry = load_registry()
    models = registry.get("models", [])
    if not models:
        raise ValueError("No registered candidates. Run `make register` first.")
    candidate = (
        next((model for model in models if model.get("version") == candidate_version), None)
        if candidate_version is not None
        else models[-1]
    )
    if candidate is None:
        raise ValueError(f"Candidate version {candidate_version} not found.")

    champion = active_production_metadata()
    promote, reason = should_promote(candidate, champion, metric, threshold)
    decision = {
        "candidate_version": candidate["version"],
        "previous_production_version": registry.get("production_version"),
        "promoted": promote,
        "reason": reason,
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if promote:
        source = Path(candidate["artifact_path"])
        shutil.copy2(source, REGISTRY_PRODUCTION_MODEL_PATH)
        registry["production_version"] = candidate["version"]
        for model in models:
            if model.get("version") == candidate["version"]:
                model["status"] = "production"
                model["production_artifact_path"] = str(REGISTRY_PRODUCTION_MODEL_PATH)
            elif model.get("status") == "production":
                model["status"] = "archived"
        _write_json(REGISTRY_PATH, registry)

    with PROMOTIONS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(decision) + "\n")
    return decision


def registry_status() -> dict:
    registry = load_registry()
    return {
        "registry_path": str(REGISTRY_PATH),
        "production_model_path": str(REGISTRY_PRODUCTION_MODEL_PATH),
        "production": active_production_metadata(),
        "candidate_count": len(registry.get("models", [])),
        "latest_candidate": latest_candidate(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["register", "promote", "status"])
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args()

    if args.command == "register":
        print(json.dumps(register_candidate(), indent=2))
    elif args.command == "promote":
        print(json.dumps(promote_candidate(candidate_version=args.version), indent=2))
    else:
        print(json.dumps(registry_status(), indent=2))


if __name__ == "__main__":
    main()
