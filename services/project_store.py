"""
Local research project persistence for InsightOptima (single-machine).

Each project lives under data/projects/<project_id>/:
  project.json   — metadata
  corpus.parquet — encoded review dataframe
  behavior.parquet — optional behavior join
  edits.json     — brief_overrides, roadmap_overrides, study_title
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = PROJECT_ROOT / "data" / "projects"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_projects_dir() -> Path:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECTS_DIR


def _slugify(name: str) -> str:
    raw = re.sub(r"[^\w\s-]", "", name.strip(), flags=re.UNICODE)
    slug = re.sub(r"[-\s]+", "-", raw).strip("-").lower()
    return (slug or "project")[:48]


def project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def list_projects() -> list[dict[str, Any]]:
    """Return project metadata sorted by updated_at descending."""
    _ensure_projects_dir()
    items: list[dict[str, Any]] = []
    for path in PROJECTS_DIR.iterdir():
        if not path.is_dir():
            continue
        meta_path = path / "project.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta.setdefault("id", path.name)
        items.append(meta)
    items.sort(key=lambda m: str(m.get("updated_at") or ""), reverse=True)
    return items


def create_project_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_slugify(name)}_{uuid.uuid4().hex[:6]}"


def save_project(
    *,
    name: str,
    review_df: pd.DataFrame,
    source_label: str = "",
    behavior_df: pd.DataFrame | None = None,
    brief_overrides: dict[str, Any] | None = None,
    roadmap_overrides: dict[str, Any] | None = None,
    study_title: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """
    Persist a project to disk. Creates a new id when project_id is None.

    Returns the project metadata dict (includes id).
    """
    _ensure_projects_dir()
    pid = project_id or create_project_id(name)
    root = project_dir(pid)
    root.mkdir(parents=True, exist_ok=True)

    meta_path = root / "project.json"
    created_at = _utc_now()
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            created_at = str(old.get("created_at") or created_at)
        except (json.JSONDecodeError, OSError):
            pass

    meta = {
        "id": pid,
        "name": name.strip() or pid,
        "source_label": source_label or "",
        "created_at": created_at,
        "updated_at": _utc_now(),
        "n_rows": int(len(review_df)),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    review_df.to_parquet(root / "corpus.parquet", index=False)

    behavior_path = root / "behavior.parquet"
    if behavior_df is not None and not behavior_df.empty:
        behavior_df.to_parquet(behavior_path, index=False)
    elif behavior_path.exists():
        behavior_path.unlink()

    edits: dict[str, Any] = {
        "brief_overrides": brief_overrides or {},
        "roadmap_overrides": roadmap_overrides or {},
    }
    if study_title:
        edits["study_title"] = study_title
    (root / "edits.json").write_text(
        json.dumps(edits, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta


def load_project(project_id: str) -> dict[str, Any]:
    """
    Load a project from disk.

    Returns dict with keys: meta, review_df, behavior_df, brief_overrides,
    roadmap_overrides, study_title.
    """
    root = project_dir(project_id)
    meta_path = root / "project.json"
    corpus_path = root / "corpus.parquet"
    if not meta_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(f"Project not found or incomplete: {project_id}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("id", project_id)
    review_df = pd.read_parquet(corpus_path)

    behavior_df = None
    behavior_path = root / "behavior.parquet"
    if behavior_path.exists():
        behavior_df = pd.read_parquet(behavior_path)

    edits: dict[str, Any] = {}
    edits_path = root / "edits.json"
    if edits_path.exists():
        try:
            edits = json.loads(edits_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            edits = {}

    return {
        "meta": meta,
        "review_df": review_df,
        "behavior_df": behavior_df,
        "brief_overrides": edits.get("brief_overrides") or {},
        "roadmap_overrides": edits.get("roadmap_overrides") or {},
        "study_title": edits.get("study_title"),
    }


def delete_project(project_id: str) -> None:
    """Remove a project directory (best-effort)."""
    import shutil

    root = project_dir(project_id)
    if root.exists() and root.is_dir():
        shutil.rmtree(root)
