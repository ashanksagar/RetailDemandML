from pathlib import Path


def test_dockerfile_uses_non_root_user_and_healthcheck():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile


def test_dockerignore_excludes_large_generated_artifacts():
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert "mlruns/" in dockerignore
    assert "models/*" in dockerignore
    assert "data/processed/*" in dockerignore
