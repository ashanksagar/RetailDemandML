import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    DATE_COLUMN,
    GROUP_COLUMNS,
    PROJECT_ROOT,
    RAW_TRAIN_PATH,
    REAL_DATA_VERIFICATION_MARKDOWN_PATH,
    REAL_DATA_VERIFICATION_PATH,
    TARGET,
    ensure_directories,
)
from src.data.ingest import file_sha256

EXPECTED_MIN_ROWS = 900_000
EXPECTED_STORES = 10
EXPECTED_ITEMS = 50
EXPECTED_START = "2013-01-01"
EXPECTED_END = "2017-12-31"


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _report_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else path.name


def kaggle_credentials_path() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def kaggle_access_token_path() -> Path:
    return Path.home() / ".kaggle" / "access_token"


def _credential_sources() -> list[tuple[str, bool, str]]:
    env_present = bool(os.getenv("KAGGLE_API_TOKEN"))
    access_token = kaggle_access_token_path()
    kaggle_json = kaggle_credentials_path()
    return [
        (
            "KAGGLE_API_TOKEN",
            env_present,
            "KAGGLE_API_TOKEN environment variable is set."
            if env_present
            else "KAGGLE_API_TOKEN environment variable is not set.",
        ),
        (
            "access_token",
            access_token.exists(),
            "Kaggle access_token file is present."
            if access_token.exists()
            else "Kaggle access_token file is not present.",
        ),
        (
            "kaggle_json",
            kaggle_json.exists(),
            "Kaggle kaggle.json file is present."
            if kaggle_json.exists()
            else "Kaggle kaggle.json file is not present.",
        ),
    ]


def _check_cli() -> CheckResult:
    path = shutil.which("kaggle")
    return CheckResult(
        name="kaggle_cli",
        passed=path is not None,
        detail="Kaggle CLI is available on PATH." if path else "Kaggle CLI is not installed or not on PATH.",
    )


def _check_credentials() -> CheckResult:
    sources = _credential_sources()
    present = [name for name, exists, _detail in sources if exists]
    details = "; ".join(detail for _name, _exists, detail in sources)
    return CheckResult(
        name="kaggle_credentials",
        passed=bool(present),
        detail=f"Credential source(s) present: {', '.join(present)}."
        if present
        else f"No Kaggle credential source found. Checked: {details}",
    )


def _load_train() -> tuple[pd.DataFrame | None, CheckResult]:
    if not RAW_TRAIN_PATH.exists():
        return None, CheckResult("train_csv_present", False, f"Missing {_report_path(RAW_TRAIN_PATH)}.")
    try:
        return pd.read_csv(RAW_TRAIN_PATH), CheckResult(
            "train_csv_present", True, f"Found {_report_path(RAW_TRAIN_PATH)}."
        )
    except Exception as exc:
        return None, CheckResult("train_csv_readable", False, str(exc))


def verify_dataframe(df: pd.DataFrame) -> list[CheckResult]:
    checks: list[CheckResult] = []
    required = {DATE_COLUMN, *GROUP_COLUMNS, TARGET}
    missing = sorted(required.difference(df.columns))
    checks.append(
        CheckResult(
            "required_columns",
            not missing,
            "All required columns present." if not missing else f"Missing columns: {missing}",
        )
    )
    if missing:
        return checks

    dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    row_count = len(df)
    store_count = df[GROUP_COLUMNS[0]].nunique()
    item_count = df[GROUP_COLUMNS[1]].nunique()
    date_min = str(dates.min().date()) if not dates.isna().all() else "unknown"
    date_max = str(dates.max().date()) if not dates.isna().all() else "unknown"

    checks.extend(
        [
            CheckResult(
                "real_dataset_row_count",
                row_count >= EXPECTED_MIN_ROWS,
                f"Rows={row_count:,}; expected at least {EXPECTED_MIN_ROWS:,} for Kaggle train.csv.",
            ),
            CheckResult(
                "store_count",
                store_count == EXPECTED_STORES,
                f"Stores={store_count}; expected {EXPECTED_STORES}.",
            ),
            CheckResult(
                "item_count",
                item_count == EXPECTED_ITEMS,
                f"Items={item_count}; expected {EXPECTED_ITEMS}.",
            ),
            CheckResult(
                "date_range",
                date_min == EXPECTED_START and date_max == EXPECTED_END,
                f"Date range={date_min} to {date_max}; expected {EXPECTED_START} to {EXPECTED_END}.",
            ),
            CheckResult(
                "non_negative_sales",
                bool((pd.to_numeric(df[TARGET], errors="coerce") >= 0).all()),
                "Sales are non-negative.",
            ),
        ]
    )
    return checks


def verification_payload() -> dict:
    ensure_directories()
    checks = [_check_cli(), _check_credentials()]
    df, train_check = _load_train()
    checks.append(train_check)
    metadata = {}
    if df is not None:
        checks.extend(verify_dataframe(df))
        metadata = {
            "path": _report_path(RAW_TRAIN_PATH),
            "sha256": file_sha256(RAW_TRAIN_PATH),
            "rows": int(len(df)),
            "columns": list(df.columns),
        }

    passed = all(check.passed for check in checks)
    likely_sample_data = bool(df is not None and len(df) < EXPECTED_MIN_ROWS)
    cli_check = next(check for check in checks if check.name == "kaggle_cli")
    credential_check = next(check for check in checks if check.name == "kaggle_credentials")
    next_steps = []
    if not cli_check.passed:
        next_steps.append("Install Kaggle CLI with `python -m pip install kaggle`.")
    if not credential_check.passed:
        next_steps.append(
            "Configure Kaggle credentials using one of: `KAGGLE_API_TOKEN`, "
            "`%USERPROFILE%\\.kaggle\\access_token`, or `%USERPROFILE%\\.kaggle\\kaggle.json`."
        )
    if not passed:
        next_steps.extend(
            [
                "Run `make data` to download and unzip the Kaggle dataset.",
                "Run `make verify-real-data` again.",
            ]
        )
    next_steps.append(
        "Run `python scripts/run_pipeline.py --config configs/default.yaml` and update README metrics."
    )
    return {
        "status": "verified_real_kaggle_data" if passed else "not_verified",
        "passed": passed,
        "likely_sample_data": likely_sample_data,
        "metadata": metadata,
        "checks": [asdict(check) for check in checks],
        "next_steps": next_steps,
    }


def write_markdown(payload: dict) -> None:
    lines = [
        "# Real Kaggle Dataset Verification",
        "",
        f"Status: `{payload['status']}`",
        f"Likely sample data: `{payload['likely_sample_data']}`",
        "",
        "## Checks",
        "",
    ]
    for check in payload["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {marker} `{check['name']}`: {check['detail']}")
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {step}" for step in payload["next_steps"])
    REAL_DATA_VERIFICATION_MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = verification_payload()
    REAL_DATA_VERIFICATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
