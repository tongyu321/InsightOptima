"""
Disk cache for encoded analysis results.

Avoids re-running BERTopic / sentiment on identical review corpora.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "analysis"


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


CACHE_SCHEMA = "v3-robust-ingest-friction"

def corpus_fingerprint(df: pd.DataFrame) -> str:
    """
    Stable hash of review texts (+ ratings when present).

    Order-independent so reshuffled uploads still cache-hit.
    Includes CACHE_SCHEMA so encoder upgrades invalidate stale parquet.
    """
    texts = df["text"].astype(str).fillna("").tolist() if "text" in df.columns else []
    ratings = (
        df["rating"].astype(str).fillna("").tolist()
        if "rating" in df.columns
        else [""] * len(texts)
    )
    pairs = sorted(zip(texts, ratings))
    payload = CACHE_SCHEMA + "\n" + "\n".join(f"{t}\t{r}" for t, r in pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def cache_path(fingerprint: str) -> Path:
    return _ensure_cache_dir() / f"{fingerprint}.parquet"


def meta_path(fingerprint: str) -> Path:
    return _ensure_cache_dir() / f"{fingerprint}.meta.json"


def load_cached_encoding(fingerprint: str) -> pd.DataFrame | None:
    """Return cached encoded dataframe if present and readable."""
    path = cache_path(fingerprint)
    meta = meta_path(fingerprint)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if meta.exists():
            info = json.loads(meta.read_text(encoding="utf-8"))
            df.attrs.update(info.get("attrs", {}))
        return df
    except Exception:
        # Fallback: pickle sibling if parquet engine missing
        pkl = path.with_suffix(".pkl")
        if pkl.exists():
            try:
                return pd.read_pickle(pkl)
            except Exception:
                return None
        return None


def save_cached_encoding(fingerprint: str, df: pd.DataFrame) -> str:
    """Persist encoded dataframe; returns cache key."""
    _ensure_cache_dir()
    attrs = {k: v for k, v in dict(df.attrs).items() if _jsonable(v)}
    meta = {"attrs": attrs, "rows": len(df)}
    meta_path(fingerprint).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        df.to_parquet(cache_path(fingerprint), index=False)
        return fingerprint
    except Exception:
        pkl = cache_path(fingerprint).with_suffix(".pkl")
        df.to_pickle(pkl)
        return fingerprint


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def clear_analysis_cache() -> int:
    """Delete cached analysis files. Returns number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for p in CACHE_DIR.glob("*"):
        if p.is_file():
            p.unlink(missing_ok=True)
            n += 1
    return n
