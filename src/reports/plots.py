from pathlib import Path
import os

from src.config import FIGURES_DIR, PROJECT_ROOT, ensure_directories

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "figure.dpi": 130,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )


def save_current_figure(name: str) -> Path:
    ensure_directories()
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_daily_sales(df: pd.DataFrame) -> Path:
    set_plot_style()
    daily = df.groupby("date")["sales"].sum()
    daily.plot()
    plt.title("Total Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    return save_current_figure("eda_total_daily_sales.png")


def plot_day_of_week(df: pd.DataFrame) -> Path:
    set_plot_style()
    tmp = df.assign(day_of_week=df["date"].dt.day_name())
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    tmp.groupby("day_of_week")["sales"].mean().reindex(order).plot(kind="bar")
    plt.title("Average Sales by Day of Week")
    plt.xlabel("")
    plt.ylabel("Average units sold")
    return save_current_figure("eda_sales_by_day_of_week.png")


def plot_monthly_sales(df: pd.DataFrame) -> Path:
    set_plot_style()
    tmp = df.assign(month=df["date"].dt.month)
    tmp.groupby("month")["sales"].mean().plot(kind="bar")
    plt.title("Average Sales by Month")
    plt.xlabel("Month")
    plt.ylabel("Average units sold")
    return save_current_figure("eda_sales_by_month.png")


def plot_top_items(df: pd.DataFrame, n: int = 15) -> Path:
    set_plot_style()
    df.groupby("item")["sales"].mean().sort_values(ascending=False).head(n).plot(kind="bar")
    plt.title(f"Top {n} Items by Average Demand")
    plt.xlabel("Item")
    plt.ylabel("Average units sold")
    return save_current_figure("eda_top_items.png")


def plot_store_comparison(df: pd.DataFrame) -> Path:
    set_plot_style()
    df.groupby("store")["sales"].mean().sort_values().plot(kind="barh")
    plt.title("Average Demand by Store")
    plt.xlabel("Average units sold")
    plt.ylabel("Store")
    return save_current_figure("eda_store_comparison.png")


def plot_store_item_heatmap(df: pd.DataFrame) -> Path:
    set_plot_style()
    pivot = df.pivot_table(index="store", columns="item", values="sales", aggfunc="mean")
    plt.imshow(pivot, aspect="auto", cmap="viridis")
    plt.colorbar(label="Average units sold")
    plt.title("Store/Item Average Demand Heatmap")
    plt.xlabel("Item index")
    plt.ylabel("Store")
    return save_current_figure("eda_store_item_heatmap.png")


def plot_example_series(df: pd.DataFrame) -> Path:
    set_plot_style()
    key = df.groupby(["store", "item"])["sales"].mean().idxmax()
    series = df[(df["store"] == key[0]) & (df["item"] == key[1])].sort_values("date")
    plt.plot(series["date"], series["sales"], alpha=0.45, label="daily")
    plt.plot(series["date"], series["sales"].rolling(28).mean(), label="28-day average")
    plt.title(f"Demand Series Example: Store {key[0]} / Item {key[1]}")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.legend()
    return save_current_figure("eda_example_series.png")


def plot_lag_correlations(df: pd.DataFrame, max_lag: int = 56) -> Path:
    set_plot_style()
    rows = []
    for lag in range(1, max_lag + 1):
        corr = df.groupby(["store", "item"])["sales"].apply(lambda s: s.corr(s.shift(lag))).mean()
        rows.append({"lag": lag, "correlation": corr})
    pd.DataFrame(rows).plot(x="lag", y="correlation", legend=False)
    plt.title("Average Sales Autocorrelation by Lag")
    plt.xlabel("Lag days")
    plt.ylabel("Correlation")
    return save_current_figure("eda_lag_correlations.png")


def plot_split_timeline(df: pd.DataFrame, validation_days: int, test_days: int) -> Path:
    set_plot_style()
    min_date = df["date"].min()
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    validation_start = test_start - pd.Timedelta(days=validation_days)
    plt.hlines(1, min_date, validation_start, linewidth=10, label="train")
    plt.hlines(1, validation_start, test_start, linewidth=10, label="validation")
    plt.hlines(1, test_start, max_date, linewidth=10, label="test")
    plt.yticks([])
    plt.title("Chronological Split Timeline")
    plt.xlabel("Date")
    plt.legend()
    return save_current_figure("eda_split_timeline.png")


def plot_model_comparison(path: Path) -> Path | None:
    if not path.exists():
        return None
    set_plot_style()
    comparison = pd.read_csv(path).sort_values("rmse", ascending=False)
    comparison.plot(x="model", y="rmse", kind="barh", legend=False)
    plt.title("Model Comparison by Validation RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("")
    return save_current_figure("model_comparison_rmse.png")
