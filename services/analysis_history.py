"""
Analysis run history and version comparison.

Stores lightweight snapshots (KPIs + roadmap) so analysts can see what changed
between two InsightOptima runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = PROJECT_ROOT / "data" / "history"


def _ensure_history_dir() -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR


def save_analysis_snapshot(
    df: pd.DataFrame,
    roadmap_df: pd.DataFrame,
    *,
    source_label: str,
    credibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Persist a compact snapshot of the current analysis run.

    Returns the snapshot metadata (including id).
    """
    _ensure_history_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_source = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_label)[:40]
    run_id = f"{stamp}_{safe_source or 'run'}"

    neg = int(df["is_negative"].sum()) if "is_negative" in df.columns else 0
    topics = (
        df.loc[df["is_negative"], "topic"].value_counts().head(30).to_dict()
        if "is_negative" in df.columns and "topic" in df.columns
        else {}
    )

    roadmap_rows = []
    if roadmap_df is not None and not roadmap_df.empty:
        for _, row in roadmap_df.iterrows():
            roadmap_rows.append(
                {
                    "priority": str(row.get("Priority", "")),
                    "topic": str(row.get("Core Pain Point", "")),
                    "priority_score": float(row.get("Priority Score", 0) or 0),
                    "mentions": int(row.get("Mentions", 0) or 0),
                }
            )

    snapshot = {
        "id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_label": source_label,
        "kpis": {
            "total_reviews": len(df),
            "negative_reviews": neg,
            "negative_pct": round(neg / len(df) * 100, 1) if len(df) else 0.0,
            "avg_rage": round(float(df.loc[df["is_negative"], "rage_index"].mean()), 1)
            if neg and "rage_index" in df.columns
            else 0.0,
            "unique_topics": int(df.loc[df["is_negative"], "topic"].nunique())
            if neg and "topic" in df.columns
            else 0,
        },
        "credibility": credibility or {},
        "topic_counts": {str(k): int(v) for k, v in topics.items()},
        "roadmap": roadmap_rows,
        "topic_method": df.attrs.get("topic_method", "unknown"),
    }

    path = HISTORY_DIR / f"{run_id}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot["path"] = str(path)
    return snapshot


def list_snapshots(limit: int = 20) -> list[dict[str, Any]]:
    """List recent snapshots, newest first (metadata only)."""
    if not HISTORY_DIR.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(HISTORY_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append(
                {
                    "id": data.get("id", path.stem),
                    "created_at": data.get("created_at", ""),
                    "source_label": data.get("source_label", ""),
                    "kpis": data.get("kpis", {}),
                    "path": str(path),
                }
            )
        except Exception:
            continue
    return items


def load_snapshot(run_id: str) -> dict[str, Any] | None:
    """Load a full snapshot by id."""
    path = HISTORY_DIR / f"{run_id}.json"
    if not path.exists():
        # allow partial match
        matches = list(HISTORY_DIR.glob(f"{run_id}*.json"))
        if not matches:
            return None
        path = matches[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compare_snapshots(older: dict[str, Any], newer: dict[str, Any]) -> dict[str, Any]:
    """
    Diff two analysis snapshots for UXR change tracking.
    """
    old_topics = set((older.get("topic_counts") or {}).keys())
    new_topics = set((newer.get("topic_counts") or {}).keys())

    old_roadmap = {r["topic"]: r for r in older.get("roadmap") or []}
    new_roadmap = {r["topic"]: r for r in newer.get("roadmap") or []}

    shared = old_topics & new_topics
    rage_deltas = []
    for topic in shared:
        old_n = int((older.get("topic_counts") or {}).get(topic, 0))
        new_n = int((newer.get("topic_counts") or {}).get(topic, 0))
        old_score = float((old_roadmap.get(topic) or {}).get("priority_score", 0) or 0)
        new_score = float((new_roadmap.get(topic) or {}).get("priority_score", 0) or 0)
        # backward compat with older snapshots
        if not old_score:
            old_score = float((old_roadmap.get(topic) or {}).get("retention_lift", 0) or 0)
        if not new_score:
            new_score = float((new_roadmap.get(topic) or {}).get("retention_lift", 0) or 0)
        rage_deltas.append(
            {
                "topic": topic,
                "volume_delta": new_n - old_n,
                "priority_score_delta": round(new_score - old_score, 1),
                "priority_before": (old_roadmap.get(topic) or {}).get("priority", ""),
                "priority_after": (new_roadmap.get(topic) or {}).get("priority", ""),
            }
        )
    rage_deltas.sort(key=lambda x: abs(x["volume_delta"]), reverse=True)

    old_k = older.get("kpis") or {}
    new_k = newer.get("kpis") or {}

    return {
        "older_id": older.get("id"),
        "newer_id": newer.get("id"),
        "kpi_delta": {
            "negative_pct": round(float(new_k.get("negative_pct", 0)) - float(old_k.get("negative_pct", 0)), 1),
            "avg_rage": round(float(new_k.get("avg_rage", 0)) - float(old_k.get("avg_rage", 0)), 1),
            "unique_topics": int(new_k.get("unique_topics", 0)) - int(old_k.get("unique_topics", 0)),
            "total_reviews": int(new_k.get("total_reviews", 0)) - int(old_k.get("total_reviews", 0)),
        },
        "new_topics": sorted(new_topics - old_topics),
        "resolved_topics": sorted(old_topics - new_topics),
        "shared_topic_changes": rage_deltas[:15],
        "priority_changes": [
            d
            for d in rage_deltas
            if d["priority_before"] and d["priority_after"] and d["priority_before"] != d["priority_after"]
        ],
    }


def snapshots_to_dataframe(snapshots: list[dict[str, Any]]) -> pd.DataFrame:
    """Flat table for UI listing."""
    rows = []
    for s in snapshots:
        k = s.get("kpis") or {}
        rows.append(
            {
                "Run ID": s.get("id", ""),
                "Created": str(s.get("created_at", ""))[:19].replace("T", " "),
                "Source": s.get("source_label", ""),
                "Reviews": k.get("total_reviews", 0),
                "Negative %": k.get("negative_pct", 0),
                "Avg Rage": k.get("avg_rage", 0),
            }
        )
    return pd.DataFrame(rows)
