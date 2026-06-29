import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLUMNS,
    DRIFT_MARKDOWN_PATH,
    DRIFT_REPORT_PATH,
    DRIFT_SUMMARY_PATH,
    FEATURES_PATH,
    ProjectConfig,
    ensure_directories,
    load_project_config,
)


@dataclass(frozen=True)
class DriftFinding:
    feature: str
    kind: str
    value: float
    threshold: float
    severity: str
    description: str


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(ref.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        edges = np.linspace(ref.min(), ref.max() + epsilon, bins + 1)
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), epsilon, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def numeric_drift_summary(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    categorical_overrides = set(CATEGORICAL_COLUMNS)
    numeric_columns = sorted(
        set(reference.select_dtypes("number").columns)
        .intersection(current.columns)
        .difference(categorical_overrides)
    )
    rows = []
    for column in numeric_columns:
        ref_mean = reference[column].mean()
        cur_mean = current[column].mean()
        ref_std = reference[column].std() or 1.0
        rows.append(
            {
                "feature": column,
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "standardized_mean_shift": (cur_mean - ref_mean) / ref_std,
                "psi": population_stability_index(reference[column], current[column]),
                "reference_missing_rate": float(reference[column].isna().mean()),
                "current_missing_rate": float(current[column].isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def categorical_drift_summary(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    inferred = set(reference.select_dtypes(include=["object", "category", "bool"]).columns)
    categorical_columns = sorted(
        (inferred.union(CATEGORICAL_COLUMNS))
        .intersection(reference.columns)
        .intersection(current.columns)
    )
    rows = []
    for column in categorical_columns:
        ref_levels = set(reference[column].dropna().astype(str).unique())
        cur = current[column].dropna().astype(str)
        new_category_rate = float((~cur.isin(ref_levels)).mean()) if len(cur) else 0.0
        rows.append(
            {
                "feature": column,
                "reference_unique": len(ref_levels),
                "current_unique": int(cur.nunique()),
                "new_category_rate": new_category_rate,
                "reference_missing_rate": float(reference[column].isna().mean()),
                "current_missing_rate": float(current[column].isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> tuple[list[DriftFinding], pd.DataFrame]:
    config = config or load_project_config()
    numeric = numeric_drift_summary(reference, current)
    categorical = categorical_drift_summary(reference, current)
    findings: list[DriftFinding] = []

    for row in numeric.to_dict("records"):
        psi = abs(float(row["psi"]))
        if psi >= config.monitoring.psi_threshold:
            findings.append(
                DriftFinding(
                    feature=row["feature"],
                    kind="psi",
                    value=psi,
                    threshold=config.monitoring.psi_threshold,
                    severity="warning",
                    description="Numeric distribution shifted by PSI threshold.",
                )
            )
        shift = abs(float(row["standardized_mean_shift"]))
        if shift >= config.monitoring.mean_shift_threshold:
            findings.append(
                DriftFinding(
                    feature=row["feature"],
                    kind="mean_shift",
                    value=shift,
                    threshold=config.monitoring.mean_shift_threshold,
                    severity="warning",
                    description="Numeric mean shifted relative to reference standard deviation.",
                )
            )
        missing_shift = abs(float(row["current_missing_rate"] - row["reference_missing_rate"]))
        if missing_shift >= config.monitoring.missing_rate_shift_threshold:
            findings.append(
                DriftFinding(
                    feature=row["feature"],
                    kind="missing_rate_shift",
                    value=missing_shift,
                    threshold=config.monitoring.missing_rate_shift_threshold,
                    severity="warning",
                    description="Missing rate changed materially.",
                )
            )

    for row in categorical.to_dict("records"):
        new_rate = float(row["new_category_rate"])
        if new_rate >= config.monitoring.new_category_rate_threshold:
            findings.append(
                DriftFinding(
                    feature=row["feature"],
                    kind="new_category_rate",
                    value=new_rate,
                    threshold=config.monitoring.new_category_rate_threshold,
                    severity="warning",
                    description="Current data contains unseen categorical levels.",
                )
            )

    numeric["feature_type"] = "numeric"
    categorical["feature_type"] = "categorical"
    summary = pd.concat([numeric, categorical], ignore_index=True, sort=False)
    return findings, summary


def write_drift_report(
    findings: list[DriftFinding],
    summary: pd.DataFrame,
    report_path: Path = DRIFT_REPORT_PATH,
    summary_path: Path = DRIFT_SUMMARY_PATH,
    markdown_path: Path = DRIFT_MARKDOWN_PATH,
) -> None:
    ensure_directories()
    payload = {
        "status": "warning" if findings else "ok",
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary.to_csv(summary_path, index=False)
    lines = [
        "# RetailDemandML Drift Report",
        "",
        f"Status: `{payload['status']}`",
        f"Findings: `{len(findings)}`",
        "",
    ]
    if findings:
        lines.append("## Findings")
        lines.append("")
        lines.extend(
            f"- `{finding.feature}` {finding.kind}: {finding.value:.4f} "
            f"(threshold {finding.threshold:.4f})"
            for finding in findings
        )
    else:
        lines.append("No drift findings exceeded configured thresholds.")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_reference_current(df: pd.DataFrame, current_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < current_fraction < 1:
        raise ValueError("current_fraction must be between 0 and 1.")
    split_index = max(1, int(len(df) * (1 - current_fraction)))
    reference = df.iloc[:split_index].copy()
    current = df.iloc[split_index:].copy()
    if reference.empty or current.empty:
        raise ValueError("Reference/current split produced an empty partition.")
    return reference, current


def run_drift_monitor(
    reference_path: Path | None = None,
    current_path: Path | None = None,
    features_path: Path = FEATURES_PATH,
    config_path: Path | None = None,
) -> dict:
    config = load_project_config(config_path)
    if reference_path and current_path:
        reference = pd.read_csv(reference_path)
        current = pd.read_csv(current_path)
    else:
        df = pd.read_csv(features_path)
        reference, current = split_reference_current(df)
    findings, summary = detect_drift(reference, current, config=config)
    write_drift_report(findings, summary)
    return {"status": "warning" if findings else "ok", "finding_count": len(findings)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--current", type=Path, default=None)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    result = run_drift_monitor(
        reference_path=args.reference,
        current_path=args.current,
        features_path=args.features,
        config_path=args.config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
