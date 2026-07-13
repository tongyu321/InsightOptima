"""
Implementation roadmap generator for InsightOptima.

Builds prioritized P0/P1/P2 action items from encoded review data,
using rage-volume matrix statistics and topic-level evidence samples.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Research-note templates keyed by common keyword patterns in discovered topics.
# diagnosis = pattern note (what shows up in the corpus); action = next research note.
ACTION_TEMPLATES: list[dict[str, str]] = [
    {
        "keywords": ["shipping", "delivery", "arrived", "package", "packaging", "plastic"],
        "diagnosis": "Feedback repeatedly describes fulfillment or packaging problems after purchase.",
        "action": "Interview recent buyers who mentioned delivery/packaging; map steps from order to open-box.",
    },
    {
        "keywords": ["price", "expensive", "cost", "worth"],
        "diagnosis": "Commenters frame price as misaligned with perceived quality or value.",
        "action": "Probe value trade-offs in follow-up interviews; compare wording with pricing/messaging copy.",
    },
    {
        "keywords": ["taste", "flavor", "quality", "fresh", "stale", "bitter", "sour", "chips", "coffee", "tea"],
        "diagnosis": "Negative reviews cluster on sensory quality or freshness inconsistency.",
        "action": "Collect batch/context details from high-rage quotes; validate with a second source if available.",
    },
    {
        "keywords": ["broken", "bug", "crash", "error", "work", "lid"],
        "diagnosis": "Users describe reliability failures that block intended use.",
        "action": "Reproduce with affected users or support tickets; turn top failure modes into research tasks.",
    },
    {
        "keywords": ["slow", "loading", "performance", "lag"],
        "diagnosis": "Friction language centers on slowness during key tasks.",
        "action": "Observe task flows with people who reported lag; note where waits interrupt goals.",
    },
    {
        "keywords": ["login", "password", "verify", "account", "sign"],
        "diagnosis": "Onboarding and account-access friction appears early in negative feedback.",
        "action": "Usability sessions on sign-in/verify paths with first-time or locked-out users.",
    },
    {
        "keywords": ["refund", "return", "cancel", "subscription"],
        "diagnosis": "Returns, refunds, or cancellation pain is a recurring complaint theme.",
        "action": "Journey-map cancel/refund with recent requesters; capture policy vs expectation gaps.",
    },
    {
        "keywords": ["notification", "spam", "email"],
        "diagnosis": "People report unwanted or excessive communications.",
        "action": "Diary or intercept study on notification volume; inventory channels vs preference controls.",
    },
    {
        "keywords": ["frosting", "cake", "cookie", "candy", "chocolate"],
        "diagnosis": "Expectation mismatch on dessert/snack experience shows up in disappointment language.",
        "action": "Compare product imagery/claims with verbatim complaints; interview recent buyers of that SKU.",
    },
    {
        "keywords": ["side effect", "nausea", "sleep", "weight", "withdrawal", "dose", "medication", "pill"],
        "diagnosis": "Patient narratives concentrate on side effects, dosing, or adherence friction.",
        "action": "Treat as discussion themes only — follow up with clinician-informed protocol; do not claim outcomes.",
    },
]


def _match_template(topic: str) -> dict[str, str] | None:
    """Find the best action template for a topic label based on keyword overlap."""
    topic_lower = topic.lower()
    for template in ACTION_TEMPLATES:
        if any(kw in topic_lower for kw in template["keywords"]):
            return template
    return None


def _assign_priority(avg_rage: float, volume: int, volume_p75: float, rage_p75: float) -> str:
    """
    Assign P0/P1/P2 priority based on rage-volume quadrant position.

    Purple Hidden Sting Zone (low volume, high rage) gets P0 — hidden retention killers.
    Red Core Blast Zone (high volume, high rage) also gets P0.
    """
    high_rage = avg_rage >= rage_p75
    high_volume = volume >= volume_p75

    if high_rage and (high_volume or avg_rage >= 75):
        return "P0"
    if high_rage or high_volume:
        return "P1"
    return "P2"


def _priority_score(priority: str, avg_rage: float, volume: int, total_negative: int) -> float:
    """
    Draft urgency score (0–100) from rage × mention share.

    Used only to sort a priority draft — not a forecast of retention lift.
    """
    volume_share = volume / max(total_negative, 1)
    base = {"P0": 78.0, "P1": 55.0, "P2": 32.0}.get(priority, 40.0)
    rage_multiplier = avg_rage / 100.0
    share_boost = min(volume_share * 40, 18.0)
    return round(min(100.0, base * rage_multiplier + share_boost), 1)


def _get_sample_evidence(df: pd.DataFrame, topic: str, n: int = 2) -> str:
    """Pull representative review snippets as evidence for a pain point topic."""
    samples = df[(df["topic"] == topic) & (df["is_negative"])].nsmallest(n, "sentiment")
    snippets = [f'"{str(row["text"])[:120]}..."' for _, row in samples.iterrows()]
    return " | ".join(snippets) if snippets else "No sample available."


def compute_rage_volume_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build topic-level rage-volume matrix for quadrant scatter plot.

    Duplicated here to avoid circular imports with app.py metrics layer.
    """
    negative_df = df[df["is_negative"]].copy()
    if negative_df.empty:
        return pd.DataFrame(columns=["topic", "volume", "avg_rage", "quadrant"])

    grouped = (
        negative_df.groupby("topic")
        .agg(volume=("review_id", "count"), avg_rage=("rage_index", "mean"))
        .reset_index()
    )
    grouped["avg_rage"] = grouped["avg_rage"].round(1)

    volume_median = grouped["volume"].median()
    rage_median = grouped["avg_rage"].median()

    def assign_quadrant(row: pd.Series) -> str:
        high_vol = row["volume"] >= volume_median
        high_rage = row["avg_rage"] >= rage_median
        if high_vol and high_rage:
            return "Red Core Blast Zone"
        if not high_vol and high_rage:
            return "Purple Hidden Sting Zone"
        if high_vol and not high_rage:
            return "Yellow Monitor Zone"
        return "Green Low Priority Zone"

    grouped["quadrant"] = grouped.apply(assign_quadrant, axis=1)
    return grouped.sort_values("avg_rage", ascending=False)


def build_implementation_roadmap(
    df: pd.DataFrame,
    behavior_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a draft priority roadmap from encoded feedback.

    Focus: discover themes → draft P0/P1/P2 → keep evidence-ready fields.
    Does not forecast retention or revenue outcomes.
    """
    from services.behavior_data import enrich_roadmap_with_behavior

    matrix = compute_rage_volume_matrix(df)
    empty_cols = [
        "Priority",
        "Core Pain Point",
        "AI Diagnosis",
        "Recommended Action",
        "Priority Score",
        "Mentions",
        "Avg Rage",
    ]
    if matrix.empty:
        return pd.DataFrame(columns=empty_cols)

    total_negative = int(df["is_negative"].sum())
    volume_p75 = matrix["volume"].quantile(0.75)
    rage_p75 = matrix["avg_rage"].quantile(0.75)

    roadmap_rows: list[dict[str, Any]] = []

    for _, row in matrix.iterrows():
        topic = str(row["topic"])
        avg_rage = float(row["avg_rage"])
        volume = int(row["volume"])
        quadrant = str(row["quadrant"])

        priority = _assign_priority(avg_rage, volume, volume_p75, rage_p75)
        template = _match_template(topic)

        if template:
            diagnosis = template["diagnosis"]
            action = template["action"]
        else:
            evidence = _get_sample_evidence(df, topic)
            diagnosis = (
                f"Recurring negative sentiment cluster ({volume} mentions, "
                f"avg rage {avg_rage:.0f}/100). Zone: {quadrant}. "
                f"Sample: {evidence}"
            )
            action = (
                f"Interview people whose feedback matches “{topic}”; "
                "confirm mechanisms before recommending a product change."
            )

        score = _priority_score(priority, avg_rage, volume, total_negative)

        roadmap_rows.append(
            {
                "Priority": priority,
                "Core Pain Point": topic,
                "AI Diagnosis": diagnosis,
                "Recommended Action": action,
                "Priority Score": score,
                "Mentions": volume,
                "Avg Rage": avg_rage,
                "_sort_key": {"P0": 0, "P1": 1, "P2": 2}.get(priority, 3),
            }
        )

    roadmap_df = pd.DataFrame(roadmap_rows)
    roadmap_df = roadmap_df.sort_values(
        ["_sort_key", "Priority Score"],
        ascending=[True, False],
    ).drop(columns=["_sort_key"])

    roadmap_df = enrich_roadmap_with_behavior(roadmap_df.reset_index(drop=True), behavior_df)
    roadmap_df["_sort_key"] = roadmap_df["Priority"].map({"P0": 0, "P1": 1, "P2": 2}).fillna(3)
    roadmap_df = roadmap_df.sort_values(
        ["_sort_key", "Priority Score"],
        ascending=[True, False],
    ).drop(columns=["_sort_key"])
    return roadmap_df.reset_index(drop=True)


def apply_roadmap_overrides(
    roadmap_df: pd.DataFrame,
    overrides: dict[str, Any] | None,
) -> pd.DataFrame:
    """
    Overlay analyst priority / pattern / next-research notes onto a generated roadmap.

    Expected overrides shape::

        {
          "Theme A": {
            "Priority": "P0" | "P1" | "P2",
            "diagnosis": "...",
            "action": "...",
          }
        }
    """
    if roadmap_df is None or roadmap_df.empty or not overrides:
        return roadmap_df

    out = roadmap_df.copy()
    for idx, row in out.iterrows():
        theme = str(row.get("Core Pain Point", ""))
        ov = overrides.get(theme) or {}
        if not ov:
            continue
        pri = str(ov.get("Priority") or "").strip().upper()
        if pri in ("P0", "P1", "P2"):
            out.at[idx, "Priority"] = pri
        if "diagnosis" in ov and str(ov["diagnosis"]).strip():
            out.at[idx, "AI Diagnosis"] = str(ov["diagnosis"]).strip()
        if "action" in ov and str(ov["action"]).strip():
            out.at[idx, "Recommended Action"] = str(ov["action"]).strip()

    out["_sort_key"] = out["Priority"].map({"P0": 0, "P1": 1, "P2": 2}).fillna(3)
    if "Priority Score" in out.columns:
        out = out.sort_values(["_sort_key", "Priority Score"], ascending=[True, False])
    else:
        out = out.sort_values(["_sort_key"], ascending=[True])
    return out.drop(columns=["_sort_key"]).reset_index(drop=True)
