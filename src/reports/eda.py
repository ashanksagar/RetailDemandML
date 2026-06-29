import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_METADATA_PATH,
    MODEL_COMPARISON_PATH,
    REPORTS_DIR,
    VALIDATION_DAYS,
    TEST_DAYS,
    ensure_directories,
)
from src.data.ingest import load_raw_sales, write_dataset_metadata
from src.data.preprocess import normalize_schema
from src.reports.plots import (
    plot_daily_sales,
    plot_day_of_week,
    plot_example_series,
    plot_lag_correlations,
    plot_model_comparison,
    plot_monthly_sales,
    plot_split_timeline,
    plot_store_comparison,
    plot_store_item_heatmap,
    plot_top_items,
)

EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
EDA_REPORT_PATH = REPORTS_DIR / "eda_report.md"


def summarize_sales(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "store_count": int(df["store"].nunique()),
        "item_count": int(df["item"].nunique()),
        "mean_sales": float(df["sales"].mean()),
        "median_sales": float(df["sales"].median()),
        "zero_sales_rate": float((df["sales"] == 0).mean()),
        "duplicate_key_count": int(df[["date", "store", "item"]].duplicated().sum()),
        "missing_value_count": int(df.isna().sum().sum()),
    }
    return summary


def write_eda_report(summary: dict, figures: list[Path]) -> None:
    lines = [
        "# RetailDemandML EDA Report",
        "",
        "## Dataset Summary",
        "",
        f"- Rows: {summary['rows']:,}",
        f"- Date range: {summary['date_min']} to {summary['date_max']}",
        f"- Stores: {summary['store_count']:,}",
        f"- Items: {summary['item_count']:,}",
        f"- Mean sales: {summary['mean_sales']:.2f}",
        f"- Median sales: {summary['median_sales']:.2f}",
        f"- Zero-sales rate: {summary['zero_sales_rate']:.2%}",
        f"- Duplicate date/store/item keys: {summary['duplicate_key_count']:,}",
        f"- Missing values: {summary['missing_value_count']:,}",
        "",
        "## Modeling Implications",
        "",
        "- Chronological validation is required because store/item demand is autocorrelated.",
        "- Weekly seasonality should be represented with day-of-week and lag features.",
        "- Store and item effects are material, so models need categorical and hierarchical features.",
        "- Lag and rolling features must be shifted to avoid target leakage.",
        "- The public dataset does not include price, promotions, or stockouts, so these remain documented optional inputs.",
        "",
        "## Generated Figures",
        "",
    ]
    lines.extend(f"- `{path.as_posix()}`" for path in figures if path is not None)
    EDA_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_eda(validation_days: int = VALIDATION_DAYS, test_days: int = TEST_DAYS) -> dict:
    ensure_directories()
    raw = load_raw_sales()
    df = normalize_schema(raw)
    write_dataset_metadata()
    figures = [
        plot_daily_sales(df),
        plot_day_of_week(df),
        plot_monthly_sales(df),
        plot_top_items(df),
        plot_store_comparison(df),
        plot_store_item_heatmap(df),
        plot_example_series(df),
        plot_lag_correlations(df),
        plot_split_timeline(df, validation_days=validation_days, test_days=test_days),
    ]
    model_plot = plot_model_comparison(MODEL_COMPARISON_PATH)
    if model_plot is not None:
        figures.append(model_plot)
    summary = summarize_sales(df)
    EDA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_eda_report(summary, figures)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-days", type=int, default=VALIDATION_DAYS)
    parser.add_argument("--test-days", type=int, default=TEST_DAYS)
    args = parser.parse_args()
    summary = generate_eda(validation_days=args.validation_days, test_days=args.test_days)
    print(json.dumps(summary, indent=2))
    print(f"Wrote EDA report to {EDA_REPORT_PATH}")
    print(f"Dataset metadata: {DATA_METADATA_PATH}")


if __name__ == "__main__":
    main()
