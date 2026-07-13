"""
Revenue-at-Risk engine (Enterpret-inspired).

Translates qualitative pain clusters into estimated revenue exposure
using review volume, rage intensity, optional ARPU, and behavior churn.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from services.behavior_data import match_behavior_to_topic
from services.roadmap_generator import build_implementation_roadmap


def compute_revenue_at_risk(
    df: pd.DataFrame,
    roadmap_df: pd.DataFrame | None = None,
    *,
    monthly_arpu: float = 12.0,
    active_users: int = 10_000,
    behavior_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Estimate monthly Revenue at Risk (RaR) for each roadmap pain point.

    Model (transparent, not magical):
    - Affected users ≈ active_users × (topic_volume / total_negatives) × rage_weight
    - Or use behavior users_affected / churn_rate when available
    - RaR ≈ affected_users × monthly_arpu × churn_probability

    Returns
    -------
    dict
        totals + per-topic table + assumptions used.
    """
    roadmap = roadmap_df if roadmap_df is not None else build_implementation_roadmap(df, behavior_df)
    total_neg = max(int(df["is_negative"].sum()) if "is_negative" in df.columns else 1, 1)

    rows: list[dict[str, Any]] = []
    for _, item in roadmap.iterrows():
        topic = str(item["Core Pain Point"])
        priority = str(item["Priority"])
        lift = float(item.get("Est. Retention Recovery (%)", 0) or 0)

        topic_mask = (df["topic"] == topic) & df["is_negative"] if "topic" in df.columns else df.index == -1
        subset = df.loc[topic_mask]
        volume = len(subset)
        avg_rage = float(subset["rage_index"].mean()) if volume and "rage_index" in subset.columns else 50.0

        behavior = match_behavior_to_topic(topic, behavior_df) if behavior_df is not None else None
        method = "review-proxy"

        if behavior and pd.notna(behavior.get("users_affected")):
            affected = float(behavior["users_affected"])
            method = "behavior-users"
        else:
            share = volume / total_neg
            rage_weight = min(max(avg_rage / 100.0, 0.25), 1.0)
            affected = active_users * share * rage_weight

        if behavior and pd.notna(behavior.get("churn_rate")):
            churn_p = min(float(behavior["churn_rate"]) / 100.0, 0.5)
            method = "behavior-churn"
        else:
            # Map rage + priority to a conservative churn probability
            base = {"P0": 0.08, "P1": 0.045, "P2": 0.02}.get(priority, 0.03)
            churn_p = min(base * (avg_rage / 70.0), 0.25)

        rar = round(affected * monthly_arpu * churn_p, 2)
        rows.append(
            {
                "Priority": priority,
                "Pain Point": topic,
                "Mentions": volume,
                "Avg Rage": round(avg_rage, 1),
                "Est. Affected Users": int(round(affected)),
                "Churn Probability": round(churn_p * 100, 1),
                "Revenue at Risk ($/mo)": rar,
                "Method": method,
                "Linked Retention Lift (%)": lift,
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("Revenue at Risk ($/mo)", ascending=False).reset_index(drop=True)

    total_rar = float(table["Revenue at Risk ($/mo)"].sum()) if not table.empty else 0.0
    p0_rar = float(table.loc[table["Priority"] == "P0", "Revenue at Risk ($/mo)"].sum()) if not table.empty else 0.0

    return {
        "total_monthly_rar": round(total_rar, 2),
        "p0_monthly_rar": round(p0_rar, 2),
        "annual_rar": round(total_rar * 12, 2),
        "assumptions": {
            "monthly_arpu": monthly_arpu,
            "active_users": active_users,
            "note": "Estimates are directional. Calibrate ARPU/users with your finance team.",
        },
        "table": table,
    }
