"""
Optional behavior context for priority drafts.

Attaches supporting product metrics next to review themes so analysts can
cross-check — not to claim causal retention outcomes.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

BEHAVIOR_ALIASES: dict[str, list[str]] = {
    "topic_or_feature": [
        "topic", "feature", "feature_name", "pain_point", "theme", "category",
        "event", "funnel_step", "stage",
    ],
    "churn_rate": ["churn_rate", "churn", "churn_pct", "cancel_rate"],
    "retention_d7": ["retention_d7", "d7_retention", "retention_7d", "ret_d7"],
    "retention_d30": ["retention_d30", "d30_retention", "retention_30d", "ret_d30"],
    "users_affected": ["users_affected", "affected_users", "users", "volume", "count"],
    "conversion_drop": ["conversion_drop", "drop_pct", "funnel_drop", "dropoff"],
}


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lstrip("\ufeff").lower()).strip("_")


def auto_map_behavior_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Map raw behavior file columns to canonical fields."""
    cols = {_normalize_col(c): c for c in df.columns}
    mapping: dict[str, str | None] = {}
    for field, aliases in BEHAVIOR_ALIASES.items():
        mapping[field] = None
        candidates = [field, *aliases]
        for alias in candidates:
            key = _normalize_col(alias)
            if key in cols:
                mapping[field] = cols[key]
                break
    return mapping


def load_behavior_dataframe(file_or_path: Any) -> pd.DataFrame:
    """Read CSV/Excel behavior export into a dataframe."""
    name = getattr(file_or_path, "name", str(file_or_path)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_or_path)
    return pd.read_csv(file_or_path)


def normalize_behavior_df(
    raw: pd.DataFrame,
    column_mapping: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """Normalize behavior data to canonical columns."""
    mapping = column_mapping or auto_map_behavior_columns(raw)
    if not mapping.get("topic_or_feature"):
        raise ValueError("Behavior file needs a topic/feature column.")

    out = pd.DataFrame()
    out["topic_or_feature"] = raw[mapping["topic_or_feature"]].astype(str).str.strip()

    for field in ("churn_rate", "retention_d7", "retention_d30", "users_affected", "conversion_drop"):
        src = mapping.get(field)
        if src and src in raw.columns:
            out[field] = pd.to_numeric(raw[src], errors="coerce")
        else:
            out[field] = pd.NA

    out = out[out["topic_or_feature"].str.len() > 0]
    return out.reset_index(drop=True)


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def match_behavior_to_topic(topic: str, behavior_df: pd.DataFrame) -> dict[str, Any] | None:
    """Best-effort keyword match from a roadmap topic to a behavior row."""
    if behavior_df is None or behavior_df.empty:
        return None

    topic_tokens = _token_set(topic)
    best: dict[str, Any] | None = None
    best_score = 0.0

    for _, row in behavior_df.iterrows():
        feat = str(row["topic_or_feature"])
        feat_tokens = _token_set(feat)
        if not feat_tokens:
            continue
        score = len(topic_tokens & feat_tokens) / max(len(topic_tokens | feat_tokens), 1)
        if feat.lower() in topic.lower() or topic.lower() in feat.lower():
            score = max(score, 0.7)
        if score > best_score and score >= 0.25:
            best_score = score
            best = row.to_dict()
            best["match_score"] = round(score, 2)

    return best


def enrich_roadmap_with_behavior(
    roadmap_df: pd.DataFrame,
    behavior_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Attach optional behavior context beside review themes.

    Columns are supporting signals for human priority review — not causal proof
    and not a retention forecast.
    """
    result = roadmap_df.copy()
    if result.empty:
        return result

    matches: list[str] = []
    churns: list[Any] = []
    users: list[Any] = []
    signals: list[str] = []

    for _, row in result.iterrows():
        topic = str(row["Core Pain Point"])
        match = match_behavior_to_topic(topic, behavior_df) if behavior_df is not None else None

        if not match:
            matches.append("")
            churns.append(pd.NA)
            users.append(pd.NA)
            signals.append("review-only")
            continue

        matches.append(str(match.get("topic_or_feature", "")))
        churn = match.get("churn_rate")
        user_n = match.get("users_affected")
        drop = match.get("conversion_drop")
        ret7 = match.get("retention_d7")

        churns.append(churn if pd.notna(churn) else pd.NA)
        users.append(user_n if pd.notna(user_n) else pd.NA)

        signal_bits = []
        if pd.notna(churn):
            signal_bits.append(f"churn {float(churn):.1f}%")
        if pd.notna(ret7):
            signal_bits.append(f"D7 ret {float(ret7):.1f}%")
        if pd.notna(drop):
            signal_bits.append(f"drop {float(drop):.1f}%")
        signals.append(" · ".join(signal_bits) if signal_bits else "matched")

    result["Behavior Match"] = matches
    result["Churn Rate (%)"] = churns
    result["Users Affected"] = users
    result["Supporting Signal"] = signals
    return result
