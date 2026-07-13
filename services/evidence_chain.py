"""
Evidence chain module for InsightOptima.

Links roadmap recommendations back to original source reviews —
core UXR requirement: every insight must be traceable to verbatim user feedback.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def count_topic_reviews(
    df: pd.DataFrame,
    topic: str,
    *,
    negative_only: bool = True,
) -> int:
    """Count reviews matching a pain point topic."""
    mask = df["topic"] == topic
    if negative_only:
        mask &= df["is_negative"]
    return int(mask.sum())


def fetch_topic_evidence(
    df: pd.DataFrame,
    topic: str,
    *,
    max_rows: int = 15,
    negative_only: bool = True,
) -> pd.DataFrame:
    """
    Retrieve source reviews supporting a roadmap pain point.

    Sorted by rage_index (desc) then sentiment (asc) so the most frustrated
    users appear first — strongest qualitative evidence for UXR reporting.

    Parameters
    ----------
    df : pd.DataFrame
        Fully encoded review dataframe.
    topic : str
        Pain point topic label (matches ``Core Pain Point`` in roadmap).
    max_rows : int
        Maximum evidence rows to return for display.
    negative_only : bool
        If True, only return negative reviews (default for pain point evidence).

    Returns
    -------
    pd.DataFrame
        Evidence subset with display-ready columns.
    """
    mask = df["topic"] == topic
    if negative_only:
        mask &= df["is_negative"]

    subset = df.loc[mask].copy()
    if subset.empty:
        # Fallback: show all reviews for topic if no negative flagged
        subset = df.loc[df["topic"] == topic].copy()

    if subset.empty:
        return pd.DataFrame()

    subset = subset.sort_values(
        ["rage_index", "sentiment"],
        ascending=[False, True],
    )

    return subset.head(max_rows)


def summarize_topic_evidence(df: pd.DataFrame, topic: str) -> dict[str, Any]:
    """
    Compute summary statistics for a pain point's evidence base.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded review dataframe.
    topic : str
        Pain point topic label.

    Returns
    -------
    dict
        Summary metrics for evidence panel header.
    """
    mask = (df["topic"] == topic) & df["is_negative"]
    subset = df.loc[mask]

    if subset.empty:
        subset = df.loc[df["topic"] == topic]

    if subset.empty:
        return {
            "total_count": 0,
            "displayed_count": 0,
            "avg_rage": 0.0,
            "avg_sentiment": 0.0,
            "lifecycle_breakdown": {},
            "has_rating": False,
            "avg_rating": None,
        }

    lifecycle_breakdown = (
        subset["lifecycle_stage"].value_counts().to_dict()
        if "lifecycle_stage" in subset.columns
        else {}
    )

    avg_rating = None
    has_rating = "rating" in subset.columns and subset["rating"].notna().any()
    if has_rating:
        avg_rating = round(float(subset["rating"].dropna().mean()), 2)

    return {
        "total_count": len(subset),
        "displayed_count": min(len(subset), 15),
        "avg_rage": round(float(subset["rage_index"].mean()), 1),
        "avg_sentiment": round(float(subset["sentiment"].mean()), 3),
        "lifecycle_breakdown": lifecycle_breakdown,
        "has_rating": has_rating,
        "avg_rating": avg_rating,
    }


def format_evidence_for_display(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format evidence rows for st.dataframe display.

    Parameters
    ----------
    evidence_df : pd.DataFrame
        Raw evidence subset from fetch_topic_evidence().

    Returns
    -------
    pd.DataFrame
        Display-ready table with readable column names.
    """
    if evidence_df.empty:
        return evidence_df

    display = pd.DataFrame()
    display["Review ID"] = evidence_df["review_id"].values
    display["User Comment"] = evidence_df["text"].values
    display["Rage Index"] = evidence_df["rage_index"].values
    display["Sentiment"] = evidence_df["sentiment"].values
    display["Lifecycle Stage"] = evidence_df["lifecycle_stage"].values

    if "rating" in evidence_df.columns and evidence_df["rating"].notna().any():
        display["Rating"] = evidence_df["rating"].values

    if "created_at" in evidence_df.columns:
        display["Date"] = pd.to_datetime(evidence_df["created_at"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        ).values

    return display
