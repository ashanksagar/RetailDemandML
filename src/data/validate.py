import importlib
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import DATE_COLUMN, GROUP_COLUMNS, TARGET

try:
    pa_module: Any = importlib.import_module("pandera.pandas")
    PanderaCheck: Any = pa_module.Check
    PanderaColumn: Any = pa_module.Column
    PanderaDataFrameSchema: Any = pa_module.DataFrameSchema
except ImportError:  # pragma: no cover - manual checks still enforce the same contract
    pa_module = None
    PanderaCheck = None
    PanderaColumn = None
    PanderaDataFrameSchema = None


@dataclass(frozen=True)
class DataQualityIssue:
    severity: str
    check: str
    message: str


@dataclass(frozen=True)
class DataQualityReport:
    passed: bool
    issues: list[DataQualityIssue]

    def raise_for_errors(self) -> None:
        errors = [issue for issue in self.issues if issue.severity == "error"]
        if errors:
            details = "; ".join(f"{issue.check}: {issue.message}" for issue in errors)
            raise ValueError(f"Data quality validation failed: {details}")


def validate_sales_data(df: pd.DataFrame, require_daily_continuity: bool = False) -> DataQualityReport:
    issues: list[DataQualityIssue] = []
    required = {DATE_COLUMN, *GROUP_COLUMNS, TARGET}
    missing = sorted(required.difference(df.columns))
    if missing:
        issues.append(DataQualityIssue("error", "required_columns", f"Missing {missing}"))
        return DataQualityReport(False, issues)

    if df[[DATE_COLUMN, *GROUP_COLUMNS]].duplicated().any():
        issues.append(
            DataQualityIssue("error", "duplicate_keys", "Duplicate date/store/item observations found")
        )
    if df[TARGET].isna().any():
        issues.append(DataQualityIssue("error", "target_missing", "Sales contains missing values"))
    if (df[TARGET] < 0).any():
        issues.append(DataQualityIssue("error", "target_non_negative", "Sales contains negative values"))

    parsed_dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    if parsed_dates.isna().any():
        issues.append(DataQualityIssue("error", "date_parse", "Some dates could not be parsed"))

    missing_rate = df.isna().mean().max()
    if missing_rate > 0.05:
        issues.append(
            DataQualityIssue(
                "warning",
                "missingness",
                f"At least one column has {missing_rate:.1%} missing values",
            )
        )

    if require_daily_continuity and not parsed_dates.isna().any():
        continuity_issues = 0
        checked = df.assign(**{DATE_COLUMN: parsed_dates}).sort_values([*GROUP_COLUMNS, DATE_COLUMN])
        for _key, group in checked.groupby(GROUP_COLUMNS):
            expected = pd.date_range(group[DATE_COLUMN].min(), group[DATE_COLUMN].max(), freq="D")
            if len(expected) != group[DATE_COLUMN].nunique():
                continuity_issues += 1
        if continuity_issues:
            issues.append(
                DataQualityIssue(
                    "warning",
                    "daily_continuity",
                    f"{continuity_issues} store/item series have date gaps",
                )
            )

    return DataQualityReport(not any(issue.severity == "error" for issue in issues), issues)


def validate_expected_kaggle_shape(df: pd.DataFrame) -> DataQualityReport:
    issues = list(validate_sales_data(df, require_daily_continuity=True).issues)
    if "store" in df and df["store"].nunique() < 2:
        issues.append(DataQualityIssue("warning", "store_count", "Dataset has fewer than 2 stores"))
    if "item" in df and df["item"].nunique() < 2:
        issues.append(DataQualityIssue("warning", "item_count", "Dataset has fewer than 2 items"))
    if DATE_COLUMN in df:
        dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce").dropna()
        if not dates.empty and (dates.max() - dates.min()).days < 365:
            issues.append(DataQualityIssue("warning", "history_length", "Dataset has less than one year of history"))
    return DataQualityReport(not any(issue.severity == "error" for issue in issues), issues)


def pandera_sales_schema():
    if PanderaDataFrameSchema is None:
        raise ImportError("pandera is not installed. Install project dependencies with `make setup`.")
    return PanderaDataFrameSchema(
        {
            DATE_COLUMN: PanderaColumn(pa_module.DateTime, nullable=False),
            "store": PanderaColumn(str, nullable=False),
            "item": PanderaColumn(str, nullable=False),
            TARGET: PanderaColumn(float, PanderaCheck.ge(0), nullable=False, coerce=True),
        },
        unique=[DATE_COLUMN, "store", "item"],
        coerce=True,
    )


def validate_with_pandera(df: pd.DataFrame) -> pd.DataFrame:
    return pandera_sales_schema().validate(df)
