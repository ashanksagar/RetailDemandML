import html
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config import (
    BACKTEST_PATH,
    DRIFT_REPORT_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
)
from src.features.feature_store import build_online_feature_row
from src.models.registry import active_production_metadata

router = APIRouter(tags=["dashboard"])


class DashboardForecastRequest(BaseModel):
    store: str = Field(default="12", description="Store identifier.")
    item: str = Field(default="48", description="Item identifier.")
    days: int = Field(default=28, ge=1, le=90, description="Forecast horizon in days.")
    start_date: str | None = Field(default=None, description="Optional first forecast date.")


class DashboardForecastRow(BaseModel):
    forecast_date: str
    horizon: int
    prediction: float


class DashboardForecastResponse(BaseModel):
    store: str
    item: str
    model_version: str
    rows: list[DashboardForecastRow]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if limit is not None:
        frame = frame.head(limit)
    return frame.where(pd.notna(frame), None).to_dict("records")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _bar_table(rows: list[dict[str, Any]], label_key: str, value_key: str, limit: int = 10) -> str:
    if not rows:
        return "<p class='muted'>No artifact available yet.</p>"
    top = rows[:limit]
    max_value = max(float(row.get(value_key) or 0) for row in top) or 1.0
    body = []
    for row in top:
        label = html.escape(str(row.get(label_key, "")))
        value = float(row.get(value_key) or 0)
        width = max(2, int((value / max_value) * 100))
        body.append(
            f"<div class='bar-row'><span title='{label}'>{label}</span>"
            f"<div class='bar'><i style='width:{width}%'></i></div><strong>{value:.3f}</strong></div>"
        )
    return "".join(body)


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int = 8) -> str:
    if not rows:
        return "<p class='muted'>No rows available.</p>"
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows[:limit]:
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(_fmt(row.get(column)))}</td>" for column in columns)
            + "</tr>"
        )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _line_svg(rows: list[dict[str, Any]], value_key: str = "rmse") -> str:
    values = [float(row[value_key]) for row in rows if row.get(value_key) is not None]
    if len(values) < 2:
        return "<p class='muted'>Need at least two points.</p>"
    width, height, pad = 520, 160, 20
    min_value, max_value = min(values), max(values)
    spread = max(max_value - min_value, 1e-6)
    points = []
    for index, value in enumerate(values):
        x = pad + index * ((width - 2 * pad) / max(len(values) - 1, 1))
        y = height - pad - ((value - min_value) / spread) * (height - 2 * pad)
        points.append(f"{x:.1f},{y:.1f}")
    labels = "".join(
        f"<text x='{pad + i * ((width - 2 * pad) / max(len(values) - 1, 1)):.1f}' y='{height - 3}'>{i + 1}</text>"
        for i in range(len(values))
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Backtest RMSE trend'>"
        f"<polyline points='{' '.join(points)}' fill='none' stroke='#2563eb' stroke-width='3'/>"
        f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#94a3b8'/>"
        f"{labels}</svg>"
    )


def dashboard_payload() -> dict[str, Any]:
    metrics = _read_json(METRICS_PATH, {})
    drift = _read_json(DRIFT_REPORT_PATH, {"status": "unknown", "finding_count": 0, "findings": []})
    comparison = _read_csv_records(MODEL_COMPARISON_PATH)
    backtest = _read_csv_records(BACKTEST_PATH)
    importance = _read_csv_records(FEATURE_IMPORTANCE_PATH, limit=20)
    active_model = active_production_metadata()
    return {
        "active_model": active_model,
        "metrics": metrics,
        "model_comparison": comparison,
        "backtest": backtest,
        "drift": drift,
        "feature_importance": importance,
        "artifact_status": {
            "metrics": METRICS_PATH.exists(),
            "model_comparison": MODEL_COMPARISON_PATH.exists(),
            "backtest": BACKTEST_PATH.exists(),
            "feature_importance": FEATURE_IMPORTANCE_PATH.exists(),
            "drift": DRIFT_REPORT_PATH.exists(),
        },
    }


@router.get("/dashboard/data", summary="Dashboard artifact data")
def dashboard_data() -> dict[str, Any]:
    return dashboard_payload()


@router.post("/dashboard/forecast", response_model=DashboardForecastResponse, summary="Dashboard 28-day forecast")
def dashboard_forecast(request: DashboardForecastRequest) -> DashboardForecastResponse:
    from src.api.main import load_model, load_serving_feature_store

    try:
        artifact = load_model()
        feature_store = load_serving_feature_store()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    last_date = pd.Timestamp(feature_store["last_date"])
    start = pd.Timestamp(request.start_date) if request.start_date else last_date + pd.Timedelta(days=1)
    rows: list[DashboardForecastRow] = []
    for offset in range(request.days):
        forecast_date = start + pd.Timedelta(days=offset)
        features = build_online_feature_row(request.store, request.item, forecast_date, feature_store)
        prediction = float(model.predict(pd.DataFrame([features])[feature_columns])[0])
        rows.append(
            DashboardForecastRow(
                forecast_date=str(forecast_date.date()),
                horizon=offset + 1,
                prediction=max(0.0, prediction),
            )
        )
    return DashboardForecastResponse(
        store=request.store,
        item=request.item,
        model_version=str(artifact.get("model_name", "local-model")),
        rows=rows,
    )


@router.get("/dashboard", response_class=HTMLResponse, summary="Internal ML operations dashboard")
def dashboard() -> HTMLResponse:
    payload = dashboard_payload()
    active = payload["active_model"] or {}
    metrics = payload["metrics"].get("production_test", {})
    drift = payload["drift"]
    comparison = payload["model_comparison"]
    backtest = payload["backtest"]
    importance = payload["feature_importance"]
    last_refreshed = date.today().isoformat()
    html_body = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetailDemandML Operations</title>
  <style>
    :root {{ color-scheme: light; --ink:#111827; --muted:#64748b; --line:#d8dee8; --bg:#f8fafc; --panel:#ffffff; --accent:#2563eb; --warn:#b45309; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, Segoe UI, Arial, sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:20px 28px; border-bottom:1px solid var(--line); background:var(--panel); display:flex; justify-content:space-between; align-items:flex-end; gap:20px; }}
    h1 {{ margin:0; font-size:22px; letter-spacing:0; }}
    h2 {{ margin:0 0 12px; font-size:15px; }}
    main {{ padding:22px 28px 36px; display:grid; gap:18px; }}
    .muted {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:12px; }}
    .two {{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
    section, .metric {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:16px; }}
    .metric span {{ display:block; font-size:12px; color:var(--muted); margin-bottom:6px; }}
    .metric strong {{ font-size:22px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ padding:8px 6px; border-bottom:1px solid #e5e7eb; text-align:left; white-space:nowrap; }}
    th {{ color:var(--muted); font-weight:600; }}
    .bar-row {{ display:grid; grid-template-columns: minmax(150px, 240px) 1fr 68px; gap:10px; align-items:center; font-size:13px; margin:8px 0; }}
    .bar-row span {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .bar {{ height:8px; background:#e2e8f0; border-radius:999px; overflow:hidden; }}
    .bar i {{ display:block; height:100%; background:var(--accent); }}
    form {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)) auto; gap:10px; align-items:end; }}
    label {{ display:grid; gap:4px; font-size:12px; color:var(--muted); }}
    input {{ height:34px; border:1px solid var(--line); border-radius:5px; padding:6px 8px; font:inherit; }}
    button {{ height:34px; border:0; border-radius:5px; background:var(--accent); color:white; padding:0 14px; font-weight:600; cursor:pointer; }}
    .forecast-output {{ margin-top:14px; max-height:360px; overflow:auto; }}
    .badge {{ display:inline-block; padding:3px 8px; border-radius:999px; font-size:12px; background:#e0f2fe; color:#075985; }}
    .warn {{ background:#fef3c7; color:var(--warn); }}
    @media (max-width: 980px) {{ .grid, .two, form {{ grid-template-columns:1fr; }} header {{ align-items:flex-start; flex-direction:column; }} }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>RetailDemandML Operations</h1>
      <div class="muted">Internal model monitoring and forecast diagnostics. Last refreshed {last_refreshed}.</div>
    </div>
    <div><span class="badge">active: {html.escape(str(active.get("name", "unregistered")))}</span></div>
  </header>
  <main>
    <div class="grid">
      <div class="metric"><span>Deployed model</span><strong>{html.escape(str(active.get("name", "n/a")))}</strong></div>
      <div class="metric"><span>Production RMSE</span><strong>{_fmt(metrics.get("rmse"))}</strong></div>
      <div class="metric"><span>Production SMAPE</span><strong>{_fmt(metrics.get("smape"))}</strong></div>
      <div class="metric"><span>Drift status</span><strong>{html.escape(str(drift.get("status", "unknown")))}</strong><div class="muted">{drift.get("finding_count", 0)} findings</div></div>
    </div>

    <section>
      <h2>Forecast Store / Item</h2>
      <form id="forecast-form">
        <label>Store<input name="store" value="12"></label>
        <label>Item<input name="item" value="48"></label>
        <label>Days<input name="days" type="number" value="28" min="1" max="90"></label>
        <label>Start date<input name="start_date" placeholder="optional"></label>
        <button type="submit">Forecast</button>
      </form>
      <div id="forecast-output" class="forecast-output muted">Submit a store/item pair to generate a forecast.</div>
    </section>

    <div class="two">
      <section>
        <h2>Feature Importance</h2>
        {_bar_table(importance, "feature", "importance", limit=12)}
      </section>
      <section>
        <h2>Model Comparison</h2>
        {_table(comparison, ["model", "mae", "rmse", "smape"], limit=8)}
      </section>
    </div>

    <div class="two">
      <section>
        <h2>Performance Over Time</h2>
        {_line_svg(backtest, "rmse")}
        {_table(backtest, ["fold", "train_end", "test_start", "test_end", "rmse", "smape"], limit=6)}
      </section>
      <section>
        <h2>Top Drift Findings</h2>
        {_table(drift.get("findings", []), ["feature", "kind", "value", "threshold"], limit=10)}
      </section>
    </div>
  </main>
  <script>
    const form = document.getElementById('forecast-form');
    const output = document.getElementById('forecast-output');
    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      const data = Object.fromEntries(new FormData(form).entries());
      data.days = Number(data.days || 28);
      if (!data.start_date) delete data.start_date;
      output.textContent = 'Forecasting...';
      const response = await fetch('/dashboard/forecast', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(data)
      }});
      if (!response.ok) {{
        output.textContent = 'Forecast failed: ' + await response.text();
        return;
      }}
      const payload = await response.json();
      const rows = payload.rows.map(row => `<tr><td>${{row.forecast_date}}</td><td>${{row.horizon}}</td><td>${{row.prediction.toFixed(2)}}</td></tr>`).join('');
      output.classList.remove('muted');
      output.innerHTML = `<table><thead><tr><th>Date</th><th>Horizon</th><th>Prediction</th></tr></thead><tbody>${{rows}}</tbody></table>`;
    }});
  </script>
</body>
</html>
"""
    return HTMLResponse(html_body)
