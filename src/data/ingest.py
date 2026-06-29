import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_METADATA_PATH,
    DATE_COLUMN,
    GROUP_COLUMNS,
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
    SAMPLE_TRAIN_PATH,
    TARGET,
    ensure_directories,
)


def generate_sample_data(
    path: Path = SAMPLE_TRAIN_PATH,
    days: int = 730,
    stores: int = 4,
    items: int = 12,
    force: bool = False,
) -> Path:
    """Generate a compact retail-like daily sales dataset for local development."""
    ensure_directories()
    if path.name == RAW_TRAIN_PATH.name and path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite canonical Kaggle data at {path}. "
            f"Write sample data to {SAMPLE_TRAIN_PATH} or pass force=True intentionally."
        )
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=days, freq="D")
    rows: list[dict[str, object]] = []

    for store in range(1, stores + 1):
        store_effect = rng.normal(loc=4 * store, scale=1.5)
        for item in range(1, items + 1):
            item_effect = rng.normal(loc=2 * item, scale=2.0)
            trend = np.linspace(0, 6, days)
            weekly = 8 * np.sin(2 * np.pi * np.arange(days) / 7)
            annual = 10 * np.sin(2 * np.pi * np.arange(days) / 365)
            noise = rng.normal(0, 5, days)
            sales = 45 + store_effect + item_effect + trend + weekly + annual + noise
            sales = np.maximum(0, np.round(sales)).astype(int)
            rows.extend(
                {"date": date, "store": store, "item": item, "sales": int(value)}
                for date, value in zip(dates, sales)
            )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def load_raw_sales(path: Path = RAW_TRAIN_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw data at {path}. Add Kaggle train.csv there or run `make sample-data`."
        )
    return pd.read_csv(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_range(df: pd.DataFrame) -> dict[str, str | None]:
    if DATE_COLUMN not in df:
        return {"min": None, "max": None}
    dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce").dropna()
    if dates.empty:
        return {"min": None, "max": None}
    return {"min": str(dates.min().date()), "max": str(dates.max().date())}


def write_dataset_metadata(path: Path = RAW_TRAIN_PATH, metadata_path: Path = DATA_METADATA_PATH) -> dict:
    ensure_directories()
    df = load_raw_sales(path)
    metadata = {
        "source": "kaggle_store_item_demand_or_local_sample",
        "path": str(path),
        "sha256": file_sha256(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "date_range": _date_range(df),
        "store_count": int(df[GROUP_COLUMNS[0]].nunique()) if GROUP_COLUMNS[0] in df else None,
        "item_count": int(df[GROUP_COLUMNS[1]].nunique()) if GROUP_COLUMNS[1] in df else None,
        "target_min": float(df[TARGET].min()) if TARGET in df else None,
        "target_max": float(df[TARGET].max()) if TARGET in df else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_file_present": RAW_TEST_PATH.exists(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _find_latest_kaggle_zip(raw_dir: Path) -> Path | None:
    candidates = sorted(raw_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def unzip_kaggle_archive(zip_path: Path | None = None, raw_dir: Path | None = None) -> list[Path]:
    ensure_directories()
    raw_dir = raw_dir or RAW_TRAIN_PATH.parent
    zip_path = zip_path or _find_latest_kaggle_zip(raw_dir)
    if zip_path is None or not zip_path.exists():
        raise FileNotFoundError("No Kaggle zip archive found in data/raw.")

    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if name in {"train.csv", "test.csv"}:
                target = raw_dir / name
                with archive.open(member) as source, target.open("wb") as destination:
                    destination.write(source.read())
                extracted.append(target)

    if RAW_TRAIN_PATH not in extracted and not RAW_TRAIN_PATH.exists():
        raise FileNotFoundError("Kaggle archive did not contain train.csv.")
    return extracted


def download_from_kaggle(destination: Path = RAW_TRAIN_PATH, unzip: bool = False) -> Path:
    """Download the public Kaggle dataset when the Kaggle CLI is configured."""
    ensure_directories()
    raw_dir = destination.parent
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "demand-forecasting-kernels-only",
            "-p",
            str(raw_dir),
        ],
        check=True,
    )
    if unzip:
        unzip_kaggle_archive(raw_dir=raw_dir)
        write_dataset_metadata(destination)
        return destination
    archive = _find_latest_kaggle_zip(raw_dir)
    if archive is None:
        raise FileNotFoundError("Kaggle command completed but no zip archive was found.")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Generate local sample data")
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=SAMPLE_TRAIN_PATH,
        help=f"Sample output path. Defaults to {SAMPLE_TRAIN_PATH}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow sample generation to overwrite an existing output, including train.csv.",
    )
    parser.add_argument("--kaggle", action="store_true", help="Download via configured Kaggle CLI")
    parser.add_argument("--unzip", action="store_true", help="Unzip train.csv/test.csv from Kaggle archive")
    parser.add_argument("--metadata", action="store_true", help="Write dataset metadata report")
    args = parser.parse_args()

    if args.kaggle:
        path = download_from_kaggle(unzip=args.unzip)
        print(f"Prepared Kaggle data at {path}")
    elif args.unzip:
        paths = unzip_kaggle_archive()
        write_dataset_metadata()
        print(f"Extracted: {', '.join(str(path) for path in paths)}")
    elif args.sample:
        path = generate_sample_data(path=args.sample_output, force=args.force)
        write_dataset_metadata(path)
        print(f"Generated sample data at {path}")
    elif args.metadata:
        metadata = write_dataset_metadata()
        print(json.dumps(metadata, indent=2))
    else:
        df = load_raw_sales()
        print(f"Loaded {len(df):,} rows from {RAW_TRAIN_PATH}")


if __name__ == "__main__":
    main()
