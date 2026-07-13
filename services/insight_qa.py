"""
Natural-language insight Q&A (Viable-inspired).

Answers analyst questions over encoded reviews using retrieval + aggregation.
No external LLM required — deterministic, auditable answers with evidence.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _filter_by_time(df: pd.DataFrame, question: str) -> pd.DataFrame:
    """Very light time window hints: last week / month."""
    if "created_at" not in df.columns or df["created_at"].isna().all():
        return df
    q = question.lower()
    ts = pd.to_datetime(df["created_at"], errors="coerce")
    if ts.isna().all():
        return df
    latest = ts.max()
    if "last week" in q or "past week" in q or "上周" in question:
        return df.loc[ts >= (latest - pd.Timedelta(days=7))]
    if "last month" in q or "past month" in q or "上个月" in question or "近一个月" in question:
        return df.loc[ts >= (latest - pd.Timedelta(days=30))]
    return df


def ask_insight(
    df: pd.DataFrame,
    question: str,
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Answer a natural-language question about the review corpus.

    Supported intents (auto-detected):
    - top complaints / pain points
    - topic deep-dive (keywords in question match topics)
    - sentiment / rage overview
    - lifecycle / stage risk
    - evidence quotes for a theme
    """
    q = (question or "").strip()
    if not q:
        return {
            "answer": "Ask a question about complaints, topics, sentiment, or a specific feature.",
            "intent": "empty",
            "metrics": {},
            "topics": [],
            "quotes": [],
        }

    working = _filter_by_time(df, q)
    neg = working[working["is_negative"]] if "is_negative" in working.columns else working
    q_lower = q.lower()
    q_tokens = _tokens(q)

    # Intent: overview / top complaints
    if any(k in q_lower for k in ("top", "main", "biggest", "worst", "主要", "最", "抱怨", "complaint", "pain")):
        if neg.empty or "topic" not in neg.columns:
            return {
                "answer": "No negative reviews available to rank complaints.",
                "intent": "top_complaints",
                "metrics": {"reviews_considered": len(working)},
                "topics": [],
                "quotes": [],
            }
        vc = neg["topic"].value_counts().head(top_n)
        lines = [f"{i}. **{topic}** — {count} mentions" for i, (topic, count) in enumerate(vc.items(), 1)]
        answer = (
            f"Across {len(neg):,} negative reviews"
            + (f" (filtered from {len(df):,})" if len(working) != len(df) else "")
            + ", the top complaint themes are:\n"
            + "\n".join(lines)
        )
        quotes = _quotes_for_topics(neg, list(vc.index[:3]), per_topic=1)
        return {
            "answer": answer,
            "intent": "top_complaints",
            "metrics": {
                "negative_reviews": len(neg),
                "unique_topics": int(neg["topic"].nunique()),
            },
            "topics": [{"topic": t, "count": int(c)} for t, c in vc.items()],
            "quotes": quotes,
        }

    # Intent: sentiment / rage
    if any(k in q_lower for k in ("sentiment", "rage", "frustrat", "angry", "情绪", "愤怒")):
        neg_pct = float(working["is_negative"].mean() * 100) if len(working) else 0
        avg_rage = float(working.loc[working["is_negative"], "rage_index"].mean()) if working["is_negative"].any() else 0
        answer = (
            f"In this slice ({len(working):,} reviews), negative rate is **{neg_pct:.1f}%** "
            f"and average rage among negatives is **{avg_rage:.1f}/100**."
        )
        return {
            "answer": answer,
            "intent": "sentiment",
            "metrics": {"negative_pct": round(neg_pct, 1), "avg_rage": round(avg_rage, 1)},
            "topics": [],
            "quotes": _quotes_for_topics(neg, list(neg["topic"].value_counts().head(2).index), per_topic=1)
            if not neg.empty and "topic" in neg.columns
            else [],
        }

    # Intent: lifecycle
    if any(k in q_lower for k in ("lifecycle", "onboarding", "retention", "stage", "漏斗", "留存", "激活")):
        if "lifecycle_stage" not in working.columns:
            return {"answer": "No lifecycle stage field available.", "intent": "lifecycle", "metrics": {}, "topics": [], "quotes": []}
        rows = []
        for stage, g in working.groupby("lifecycle_stage"):
            rate = float(g["is_negative"].mean() * 100) if len(g) else 0
            rows.append((stage, len(g), rate))
        rows.sort(key=lambda x: x[2], reverse=True)
        lines = [f"- **{s}**: {n} reviews, {r:.1f}% negative" for s, n, r in rows]
        worst = rows[0][0] if rows else "N/A"
        answer = f"Highest drop-out risk stage: **{worst}**.\n" + "\n".join(lines)
        return {"answer": answer, "intent": "lifecycle", "metrics": {"highest_risk_stage": worst}, "topics": [], "quotes": []}

    # Intent: topic deep-dive via keyword overlap with topic labels + text
    topic_hits: list[tuple[str, float]] = []
    if "topic" in working.columns:
        for topic, count in working["topic"].value_counts().items():
            score = len(q_tokens & _tokens(str(topic)))
            if score:
                topic_hits.append((str(topic), score + count / 1000.0))
        topic_hits.sort(key=lambda x: x[1], reverse=True)

    if topic_hits:
        topic = topic_hits[0][0]
        subset = working[working["topic"] == topic]
        neg_n = int(subset["is_negative"].sum()) if "is_negative" in subset.columns else len(subset)
        avg_rage = float(subset["rage_index"].mean()) if "rage_index" in subset.columns else 0
        quotes = []
        for _, r in subset.sort_values("rage_index", ascending=False).head(3).iterrows():
            quotes.append({"topic": topic, "text": str(r.get("text", "")), "rage": r.get("rage_index", "")})
        answer = (
            f"For **{topic}**: {len(subset):,} related reviews "
            f"({neg_n} negative), avg rage **{avg_rage:.1f}/100**.\n"
            f"Most frustrated users say things like the quotes below."
        )
        return {
            "answer": answer,
            "intent": "topic_deep_dive",
            "metrics": {"topic": topic, "reviews": len(subset), "negative": neg_n, "avg_rage": round(avg_rage, 1)},
            "topics": [{"topic": topic, "count": len(subset)}],
            "quotes": quotes,
        }

    # Fallback: keyword search in text
    if q_tokens and "text" in working.columns:
        mask = working["text"].astype(str).str.lower().apply(lambda s: any(tok in s for tok in q_tokens))
        hits = working.loc[mask]
        if len(hits):
            answer = (
                f"Found **{len(hits):,}** reviews matching keywords from your question. "
                f"Negative share: **{hits['is_negative'].mean()*100:.1f}%**."
                if "is_negative" in hits.columns
                else f"Found **{len(hits):,}** matching reviews."
            )
            quotes = []
            sample = hits.sort_values("rage_index", ascending=False).head(3) if "rage_index" in hits.columns else hits.head(3)
            for _, r in sample.iterrows():
                quotes.append({"topic": r.get("topic", ""), "text": str(r.get("text", "")), "rage": r.get("rage_index", "")})
            return {
                "answer": answer,
                "intent": "keyword_search",
                "metrics": {"matches": len(hits)},
                "topics": [],
                "quotes": quotes,
            }

    return {
        "answer": (
            "I can answer questions like:\n"
            "- What are the top complaints?\n"
            "- How bad is sentiment / rage?\n"
            "- Which lifecycle stage is riskiest?\n"
            "- What do users say about <feature/topic>?"
        ),
        "intent": "help",
        "metrics": {},
        "topics": [],
        "quotes": [],
    }


def _quotes_for_topics(df: pd.DataFrame, topics: list[str], per_topic: int = 1) -> list[dict[str, Any]]:
    quotes: list[dict[str, Any]] = []
    for topic in topics:
        subset = df[df["topic"] == topic]
        if subset.empty:
            continue
        ordered = subset.sort_values("rage_index", ascending=False) if "rage_index" in subset.columns else subset
        for _, r in ordered.head(per_topic).iterrows():
            quotes.append({"topic": topic, "text": str(r.get("text", "")), "rage": r.get("rage_index", "")})
    return quotes
