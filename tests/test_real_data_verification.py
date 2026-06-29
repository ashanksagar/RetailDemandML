from pathlib import Path

import pandas as pd

from src.data import verify_real_dataset
from src.data.ingest import generate_sample_data
from src.data.verify_real_dataset import _check_credentials, verify_dataframe


def test_real_data_verification_flags_sample_sized_data():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=10),
            "store": ["1"] * 10,
            "item": ["1"] * 10,
            "sales": [1] * 10,
        }
    )

    checks = {check.name: check for check in verify_dataframe(df)}

    assert not checks["real_dataset_row_count"].passed
    assert not checks["store_count"].passed
    assert not checks["item_count"].passed


def test_kaggle_api_token_environment_variable_counts_as_credentials(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "token")
    monkeypatch.setattr(verify_real_dataset, "kaggle_access_token_path", lambda: Path("missing_access_token"))
    monkeypatch.setattr(verify_real_dataset, "kaggle_credentials_path", lambda: Path("missing_kaggle_json"))

    check = _check_credentials()

    assert check.passed
    assert "KAGGLE_API_TOKEN" in check.detail


def test_access_token_file_counts_as_credentials(tmp_path, monkeypatch):
    access_token = tmp_path / "access_token"
    access_token.write_text("token", encoding="utf-8")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(verify_real_dataset, "kaggle_access_token_path", lambda: access_token)
    monkeypatch.setattr(verify_real_dataset, "kaggle_credentials_path", lambda: tmp_path / "kaggle.json")

    check = _check_credentials()

    assert check.passed
    assert "access_token" in check.detail


def test_sample_generation_refuses_to_overwrite_existing_train_csv(tmp_path):
    train_path = tmp_path / "train.csv"
    train_path.write_text("date,store,item,sales\n2013-01-01,1,1,10\n", encoding="utf-8")

    try:
        generate_sample_data(path=train_path)
    except FileExistsError as exc:
        assert "Refusing to overwrite canonical Kaggle data" in str(exc)
    else:
        raise AssertionError("sample generation should not overwrite existing train.csv")

    assert train_path.read_text(encoding="utf-8") == "date,store,item,sales\n2013-01-01,1,1,10\n"


def test_sample_generation_uses_explicit_sample_file(tmp_path):
    sample_path = tmp_path / "sample_train.csv"

    path = generate_sample_data(path=sample_path, days=3, stores=1, items=1)

    assert path == sample_path
    assert sample_path.exists()
    assert len(pd.read_csv(sample_path)) == 3
