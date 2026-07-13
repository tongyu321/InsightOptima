"""
Lightweight quant / qual strand snapshots for mixed-methods compare.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from services.evidence_chain import fetch_topic_evidence

StrandKey = Literal["quant", "qual"]


def detect_strand_key(source_label: str) -> StrandKey | None:
    """Map a case source label to quant or qual."""
    s = source_label.lower()
    if "qual" in s or "pubpeer" in s or ("zenodo" in s and "open" in s):
        return "qual"
    if "quant" in s or "drugs.com" in s or "uci" in s or "drug review" in s:
        return "quant"
    return None


def build_strand_snapshot(
    df: pd.DataFrame,
    *,
    source_label: str,
    strand: StrandKey,
    max_themes: int = 8,
) -> dict[str, Any]:
    """Compact snapshot for side-by-side compare (no full corpus)."""
    total = int(len(df))
    neg_n = int(df["is_negative"].sum()) if "is_negative" in df.columns else 0
    has_rating = "rating" in df.columns and df["rating"].notna().any()

    themes: list[dict[str, Any]] = []
    top_quotes: list[dict[str, Any]] = []
    if "topic" in df.columns and not df.empty:
        work = df[df["is_negative"]] if "is_negative" in df.columns else df
        if not work.empty:
            count_col = "review_id" if "review_id" in work.columns else "text"
            if "rage_index" in work.columns:
                g = (
                    work.groupby("topic", dropna=False)
                    .agg(count=(count_col, "count"), avg_rage=("rage_index", "mean"))
                    .reset_index()
                    .sort_values("count", ascending=False)
                    .head(max_themes)
                )
                g["avg_rage"] = pd.to_numeric(g["avg_rage"], errors="coerce").round(1)
            else:
                g = (
                    work.groupby("topic", dropna=False)
                    .agg(count=(count_col, "count"))
                    .reset_index()
                    .sort_values("count", ascending=False)
                    .head(max_themes)
                )
                g["avg_rage"] = 0.0
            for _, row in g.iterrows():
                topic = str(row["topic"])
                themes.append(
                    {
                        "theme": topic,
                        "n": int(row["count"]),
                        "avg_rage": float(row["avg_rage"]) if pd.notna(row.get("avg_rage")) else 0.0,
                    }
                )
                ev = fetch_topic_evidence(df, topic, max_rows=1)
                if not ev.empty:
                    top_quotes.append(
                        {
                            "theme": topic,
                            "text": str(ev.iloc[0].get("text", ""))[:220],
                        }
                    )

    evidence_type = (
        "Rated reviews — volume + severity signals"
        if strand == "quant"
        else "Open-ended text — thematic / text-only signals"
    )

    return {
        "strand": strand,
        "source_label": source_label,
        "n": total,
        "neg_n": neg_n,
        "neg_pct": round(neg_n / total * 100, 1) if total else 0.0,
        "n_themes": int(df["topic"].nunique()) if "topic" in df.columns else 0,
        "has_rating": bool(has_rating),
        "evidence_type": evidence_type,
        "themes": themes,
        "top_quotes": top_quotes,
    }


def both_strands_ready(snapshots: dict[str, Any] | None) -> bool:
    if not isinstance(snapshots, dict):
        return False
    return "quant" in snapshots and "qual" in snapshots
