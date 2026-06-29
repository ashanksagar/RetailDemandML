import pandas as pd

from src.config import DATE_COLUMN, GROUP_COLUMNS, OPTIONAL_BUSINESS_COLUMNS, TARGET

COLUMN_ALIASES = {
    "product": "item",
    "sku": "item",
    "quantity": "sales",
    "demand": "sales",
    "units": "sales",
}


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(columns={c: COLUMN_ALIASES.get(c.lower(), c.lower()) for c in df.columns})
    required = {DATE_COLUMN, *GROUP_COLUMNS, TARGET}
    missing = required.difference(normalized.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    optional_columns = [column for column in OPTIONAL_BUSINESS_COLUMNS if column in normalized.columns]
    normalized = normalized[[DATE_COLUMN, *GROUP_COLUMNS, TARGET, *optional_columns]].copy()
    normalized[DATE_COLUMN] = pd.to_datetime(normalized[DATE_COLUMN])
    normalized[TARGET] = pd.to_numeric(normalized[TARGET], errors="coerce")
    normalized = normalized.dropna(subset=[DATE_COLUMN, TARGET])

    for column in GROUP_COLUMNS:
        normalized[column] = normalized[column].astype(str)
    for column in optional_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0)

    normalized = normalized.sort_values([*GROUP_COLUMNS, DATE_COLUMN]).reset_index(drop=True)
    return normalized


def chronological_split(
    df: pd.DataFrame, validation_days: int, test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_date = df[DATE_COLUMN].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    validation_start = test_start - pd.Timedelta(days=validation_days)

    train = df[df[DATE_COLUMN] < validation_start].copy()
    validation = df[(df[DATE_COLUMN] >= validation_start) & (df[DATE_COLUMN] < test_start)].copy()
    test = df[df[DATE_COLUMN] >= test_start].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition. Use more history or fewer days.")

    return train, validation, test
