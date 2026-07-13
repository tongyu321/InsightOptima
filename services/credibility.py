"""
Analysis credibility / quality metrics for InsightOptima.

Surfaces how trustworthy the current run is — coverage, concentration,
uncategorized share — so UXR users don't over-read noisy clusters.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

UNCATEGORIZED_LABELS = {
    "mixed / uncategorized complaints",
    "general negative feedback",
    "unclassified feedback",
    "",
    "nan",
    "none",
}


def compute_credibility_report(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build a credibility report for an encoded review dataframe.

    Returns
    -------
    dict
        Metrics + human-readable warnings + overall confidence (0–100).
    """
    total = len(df)
    if total == 0:
        return {
            "total_reviews": 0,
            "confidence": 0,
            "grade": "F",
            "warnings": ["No reviews to analyze."],
            "metrics": {},
        }

    negative = df[df["is_negative"]] if "is_negative" in df.columns else df.iloc[0:0]
    neg_n = len(negative)

    # Uncategorized among negatives
    unc_n = 0
    if neg_n and "topic" in negative.columns:
        unc_n = int(
            negative["topic"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(UNCATEGORIZED_LABELS)
            .sum()
        )
    unc_pct = round(unc_n / neg_n * 100, 1) if neg_n else 0.0

    # Short texts
    short_n = 0
    if "text" in df.columns:
        short_n = int(df["text"].astype(str).str.len().lt(30).sum())
    short_pct = round(short_n / total * 100, 1)

    # Top-topic concentration (among negatives)
    top_share = 0.0
    top_topic = "N/A"
    unique_neg_topics = 0
    if neg_n and "topic" in negative.columns:
        vc = negative["topic"].value_counts()
        unique_neg_topics = int(vc.nunique())
        if len(vc):
            top_topic = str(vc.index[0])
            top_share = round(float(vc.iloc[0] / neg_n * 100), 1)

    # Language unknown share
    unknown_lang_pct = 0.0
    if "language" in df.columns:
        unknown_lang_pct = round(
            float((df["language"].astype(str) == "unknown").mean() * 100), 1
        )

    # Lifecycle fallback: if one stage dominates >85%, keyword coding may be weak
    stage_dominance = 0.0
    if "lifecycle_stage" in df.columns:
        stage_dominance = round(float(df["lifecycle_stage"].value_counts(normalize=True).iloc[0] * 100), 1)

    topic_method = df.attrs.get("topic_method", "unknown")
    merge_stats = df.attrs.get("topic_merge", {}) or {}

    warnings: list[str] = []
    score = 100.0

    if total < 50:
        warnings.append(f"Small sample ({total} reviews) — themes may be unstable.")
        score -= 15
    if neg_n < 20:
        warnings.append(f"Only {neg_n} negative reviews — pain-point clusters may be noisy.")
        score -= 10
    if unc_pct >= 25:
        warnings.append(f"{unc_pct}% of negative reviews are uncategorized.")
        score -= min(20, unc_pct / 2)
    if top_share >= 60 and unique_neg_topics <= 3:
        warnings.append(
            f"Themes are concentrated: “{top_topic}” covers {top_share}% of negatives."
        )
        score -= 15
    if short_pct >= 20:
        warnings.append(f"{short_pct}% of texts are very short (<30 chars).")
        score -= 8
    if unknown_lang_pct >= 40:
        warnings.append(f"{unknown_lang_pct}% language detection unknown — sentiment may be weaker.")
        score -= 8
    if stage_dominance >= 85:
        top_stage = str(df["lifecycle_stage"].value_counts().index[0]) if "lifecycle_stage" in df.columns else ""
        if top_stage == "General feedback" or df.attrs.get("lifecycle_mode") == "disabled":
            # Expected when product-lifecycle keywords are rare — not a quality failure
            pass
        else:
            warnings.append(
                f"Lifecycle coding is skewed ({stage_dominance}% in one stage) — keyword heuristics may be weak for this corpus."
            )
            score -= 10
    if topic_method == "tfidf":
        warnings.append("Using TF-IDF fallback (BERTopic unavailable) — labels are statistical, not semantic.")
        score -= 5

    if not warnings:
        warnings.append(
            "No major quality flags — still treat themes and priorities as discussion drafts, not final conclusions."
        )

    confidence = int(max(0, min(100, round(score))))
    if confidence >= 80:
        grade = "A"
    elif confidence >= 65:
        grade = "B"
    elif confidence >= 50:
        grade = "C"
    elif confidence >= 35:
        grade = "D"
    else:
        grade = "F"

    return {
        "total_reviews": total,
        "negative_reviews": neg_n,
        "confidence": confidence,
        "grade": grade,
        "warnings": warnings,
        "metrics": {
            "uncategorized_negative_pct": unc_pct,
            "uncategorized_negative_count": unc_n,
            "short_text_pct": short_pct,
            "top_topic": top_topic,
            "top_topic_share_pct": top_share,
            "unique_negative_topics": unique_neg_topics,
            "unknown_language_pct": unknown_lang_pct,
            "lifecycle_stage_dominance_pct": stage_dominance,
            "topic_method": topic_method,
            "topics_merged": merge_stats.get("merges", 0),
            "topics_before_merge": merge_stats.get("topics_before"),
            "topics_after_merge": merge_stats.get("topics_after"),
            "analysis_purpose": "discover + prioritize + evidence",
        },
    }
