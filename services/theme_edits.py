"""
Manual theme edits for InsightOptima research workspace.

Renames / merges write back to the row-level ``topic`` column so roadmap
and brief recompute on the next render.
"""

from __future__ import annotations

import pandas as pd


def rename_topic(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    """Rename a single topic label across all matching rows."""
    if "topic" not in df.columns:
        raise ValueError("DataFrame has no topic column")
    old_s = str(old).strip()
    new_s = str(new).strip()
    if not old_s or not new_s or old_s == new_s:
        return df.copy()
    out = df.copy()
    mask = out["topic"].astype(str) == old_s
    out.loc[mask, "topic"] = new_s
    return out


def merge_topics(df: pd.DataFrame, sources: list[str], into: str) -> pd.DataFrame:
    """
    Merge multiple topic labels into one canonical label.

    ``into`` may be a new name or one of the sources.
    """
    if "topic" not in df.columns:
        raise ValueError("DataFrame has no topic column")
    target = str(into).strip()
    src = [str(s).strip() for s in sources if str(s).strip()]
    if not target or len(src) < 1:
        return df.copy()
    # Always include target as a source if it already exists as a theme name
    # being kept; sources list is what gets remapped.
    out = df.copy()
    mask = out["topic"].astype(str).isin(src)
    out.loc[mask, "topic"] = target
    return out


def list_theme_stats(df: pd.DataFrame, *, negative_only: bool = True, n: int = 20) -> pd.DataFrame:
    """Top themes with counts and average rage for the editor UI."""
    if "topic" not in df.columns or df.empty:
        return pd.DataFrame(columns=["topic", "count", "avg_rage"])
    work = df
    if negative_only and "is_negative" in df.columns:
        work = df[df["is_negative"]]
    if work.empty:
        return pd.DataFrame(columns=["topic", "count", "avg_rage"])
    count_col = "review_id" if "review_id" in work.columns else "text"
    agg: dict[str, tuple[str, str]] = {"count": (count_col, "count")}
    if "rage_index" in work.columns:
        agg["avg_rage"] = ("rage_index", "mean")
    g = (
        work.groupby("topic", dropna=False)
        .agg(**{k: v for k, v in agg.items()})
        .reset_index()
        .sort_values("count", ascending=False)
        .head(n)
    )
    if "avg_rage" in g.columns:
        g["avg_rage"] = g["avg_rage"].round(1)
    else:
        g["avg_rage"] = 0.0
    return g
