"""
Pre-flight validation for InsightOptima uploads.

Kept in a standalone module to avoid import issues with data_loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PreflightReport:
    """Pre-analysis validation report shown to the user before encoding."""

    total_rows: int = 0
    valid_rows: int = 0
    skipped_short: int = 0
    skipped_empty: int = 0
    missing_rating: int = 0
    missing_date: int = 0
    detected_platform: str = "Unknown"
    auto_mapped: bool = False
    mapping_confidence: str = "low"  # high | medium | low
    warnings: list[str] = field(default_factory=list)
    ready: bool = False
    column_mapping: dict[str, str | None] = field(default_factory=dict)


def run_preflight(raw_df: pd.DataFrame, column_mapping: dict[str, str | None]) -> PreflightReport:
    """
    Validate uploaded data and column mapping before running analysis.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw uploaded dataframe.
    column_mapping : dict
        User-confirmed or auto-detected column mapping.

    Returns
    -------
    PreflightReport
        Validation report with warnings and ready flag.
    """
    import sys

    dl = sys.modules.get("insightoptima.data_loader")
    if dl is None:
        from services.data_loader import auto_resolve_columns  # noqa: fallback
    else:
        auto_resolve_columns = dl.auto_resolve_columns

    report = PreflightReport(
        total_rows=len(raw_df),
        column_mapping=column_mapping.copy(),
    )

    _, platform, confidence = auto_resolve_columns(raw_df)
    report.detected_platform = platform
    report.mapping_confidence = confidence
    report.auto_mapped = confidence in ("high", "medium")

    text_col = column_mapping.get("text")
    if not text_col or text_col not in raw_df.columns:
        report.warnings.append("No review text column selected. Please map a text column below.")
        report.ready = False
        return report

    texts = raw_df[text_col].astype(str).str.strip()
    texts = texts.replace({"nan": "", "None": ""})

    if column_mapping.get("summary") and column_mapping["summary"] in raw_df.columns:
        summaries = raw_df[column_mapping["summary"]].astype(str).str.strip()
        combined = summaries + ". " + texts
        combined = combined.str.replace(r"^\.?\s*", "", regex=True)
    else:
        combined = texts

    empty_mask = combined.str.len() == 0
    short_mask = (combined.str.len() > 0) & (combined.str.len() < 5)
    valid_mask = combined.str.len() >= 5

    report.skipped_empty = int(empty_mask.sum())
    report.skipped_short = int(short_mask.sum())
    report.valid_rows = int(valid_mask.sum())

    if column_mapping.get("rating") and column_mapping["rating"] in raw_df.columns:
        report.missing_rating = int(raw_df[column_mapping["rating"]].isna().sum())
    else:
        report.missing_rating = report.total_rows
        report.warnings.append("No rating column mapped — sentiment will be derived from text only.")

    if column_mapping.get("created_at") and column_mapping["created_at"] in raw_df.columns:
        report.missing_date = int(
            pd.to_datetime(raw_df[column_mapping["created_at"]], errors="coerce").isna().sum()
        )
    else:
        report.missing_date = report.total_rows
        report.warnings.append("No date column mapped — trend analysis unavailable.")

    if report.valid_rows == 0:
        report.warnings.append(
            "No rows with at least 5 characters of text. Check your text column mapping."
        )
        report.ready = False
    elif report.valid_rows < 3:
        report.warnings.append(f"Only {report.valid_rows} valid rows — results may be unreliable.")
        report.ready = True
    else:
        report.ready = True

    if report.skipped_short > 0:
        report.warnings.append(f"{report.skipped_short:,} rows skipped (text too short, < 5 chars).")

    if report.total_rows > 10000:
        report.warnings.append(
            f"Large dataset ({report.total_rows:,} rows). Analysis may take 1–3 minutes."
        )

    return report
