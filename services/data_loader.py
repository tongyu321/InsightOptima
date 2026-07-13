"""
Data ingestion layer for InsightOptima.

Supports CSV, TSV, Excel, and JSON uploads with three-layer column detection:
  1. Exact alias matching (App Store, Amazon, Zendesk, etc.)
  2. Fuzzy substring matching
  3. Manual column mapping (UI fallback — guarantees any text column works)

All uploads are normalized to a fixed internal schema before semantic encoding.
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal schema field names
# ---------------------------------------------------------------------------

MAPPABLE_FIELDS: list[str] = [
    "text",
    "summary",
    "rating",
    "created_at",
    "review_id",
    "lifecycle_stage",
    "sentiment",
    "rage_index",
    "topic",
]

OPTIONAL_FIELDS = {"summary", "rating", "created_at", "review_id", "lifecycle_stage", "sentiment", "rage_index", "topic"}

# ---------------------------------------------------------------------------
# Layer 1 — Exact alias registry (platform export column names)
# ---------------------------------------------------------------------------

COLUMN_ALIASES: dict[str, list[str]] = {
    "text": [
        "text", "review", "comment", "content", "review_text", "body",
        "review body", "review_content", "review text", "reviewtext",
        "message", "feedback", "description", "review_body", "user_comment",
        "customer_feedback", "verbatim", "response", "notes", "detail",
        "reviewcontent", "comment_body", "ticket_description",
        "comment_text", "open_ended", "open ended", "answer", "answers",
        "response_text", "survey_response", "free_text", "freetext",
        "user_feedback", "complaint", "complaints", "opinion", "opinions",
        "post", "post_text", "tweet", "caption", "story", "narrative",
        "qual_text", "qualitative", "transcript", "utterance",
    ],
    "summary": [
        "summary", "title", "review_title", "headline", "subject",
        "ticket_subject", "issue_title", "topic_title", "recipe_name",
        "recipe", "product_name", "product name", "name",
    ],
    "rating": [
        "rating", "score", "stars", "star_rating", "overall", "overall_rating",
        "star rating", "review_score", "review_rating", "app_rating",
        "customer_rating", "nps", "sentiment_rating", "thumbs", "likes",
        "star", "ratings", "grade", "mark", "points",
    ],
    "lifecycle_stage": [
        "lifecycle_stage", "stage", "user_stage", "lifecycle", "phase",
        "user_phase", "journey_stage",
    ],
    "sentiment": ["sentiment", "sentiment_score", "polarity", "compound", "sentiment_label"],
    "rage_index": ["rage_index", "rage", "frustration", "frustration_index"],
    "topic": ["topic", "theme", "category", "pain_point", "tag", "issue_type", "label"],
    "created_at": [
        "created_at", "date", "timestamp", "review_date", "time", "reviewed_at",
        "submitted_at", "created", "submission_date", "review_time", "posted_at",
        "datetime", "created_date", "updated_at", "comment_date", "posted",
    ],
    "review_id": [
        "review_id", "id", "reviewid", "ticket_id", "comment_id", "feedback_id",
        "uuid", "record_id", "response_id", "answer_id",
    ],
}

# Layer 2 — Fuzzy substring patterns (column name contains any of these)
FUZZY_PATTERNS: dict[str, list[str]] = {
    "text": [
        "review", "comment", "feedback", "message", "content", "body",
        "verbatim", "text", "description", "note", "answer", "response",
        "opinion", "complaint", "utterance", "transcript",
    ],
    "summary": ["summary", "title", "subject", "headline", "recipe", "product name"],
    "rating": ["rating", "score", "star", "nps", "grade", "thumb"],
    "created_at": ["date", "time", "timestamp", "created", "posted", "submitted"],
    "review_id": ["id", "uuid", "ticket", "record", "comment_id"],
    "topic": ["topic", "category", "theme", "tag", "type", "label"],
}

# Known platform fingerprints for auto-detection messaging
PLATFORM_FINGERPRINTS: dict[str, list[str]] = {
    "Amazon Fine Food Reviews": ["productid", "helpfulnessnumerator", "profileName"],
    "App Store Connect": ["app name", "review title", "review text", "developer response"],
    "Google Play Console": ["package name", "review link", "device", "app version code"],
    "Zendesk": ["ticket id", "requester", "assignee", "priority", "status"],
    "Intercom": ["conversation id", "user id", "admin id", "part_type"],
    "Trustpilot": ["reviewcontent", "consumer", "business unit"],
    "Generic Review Export": ["review", "rating", "comment"],
}


def _normalize_col_name(name: str) -> str:
    """Normalize column name for comparison: lowercase, collapse separators."""
    return re.sub(r"[\s_\-]+", " ", str(name).lower().strip())


def _read_csv_with_encoding(source: Any) -> pd.DataFrame:
    """Try multiple encodings and separators for robust CSV/TSV parsing."""
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
    separators = [",", "\t", ";", "|"]

    if isinstance(source, (str, Path)):
        raw_bytes = Path(source).read_bytes()
    else:
        source.seek(0)
        raw_bytes = source.read()
        source.seek(0)

    last_error: Exception | None = None

    for encoding in encodings:
        for sep in separators:
            try:
                text = raw_bytes.decode(encoding)
                df = pd.read_csv(
                    StringIO(text),
                    sep=sep,
                    on_bad_lines="skip",
                    engine="python",
                )
                if len(df.columns) >= 1 and len(df) >= 0:
                    # Reject if only one column and sep wrong (whole line in one col)
                    if len(df.columns) == 1 and sep == "," and "\t" in text[:5000]:
                        continue
                    return df
            except Exception as exc:
                last_error = exc
                continue

    raise ValueError(
        f"Could not parse CSV/TSV file. Tried encodings: {encodings}. "
        f"Last error: {last_error}"
    )


def _read_zip_archive(source: Any) -> pd.DataFrame:
    """
    Open a .zip and read the first supported tabular file inside.

    Prefers the largest .csv/.tsv/.xlsx/.xls/.json member (skips __MACOSX / hidden).
    """
    import zipfile
    from io import BytesIO

    if isinstance(source, (str, Path)):
        raw_bytes = Path(source).read_bytes()
        label = Path(source).name
    else:
        source.seek(0)
        raw_bytes = source.read()
        source.seek(0)
        label = getattr(source, "name", "upload.zip")

    try:
        zf = zipfile.ZipFile(BytesIO(raw_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Could not open ZIP file “{label}”: {exc}") from exc

    supported = {".csv", ".tsv", ".xlsx", ".xls", ".json"}
    candidates: list[tuple[int, str]] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/")
        base = name.split("/")[-1]
        if not base or base.startswith("."):
            continue
        if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
            continue
        suffix = Path(base).suffix.lower()
        if suffix not in supported:
            continue
        candidates.append((info.file_size, name))

    if not candidates:
        raise ValueError(
            f"ZIP “{label}” has no CSV, TSV, Excel, or JSON file inside. "
            f"Members: {zf.namelist()[:12]}"
        )

    # Largest file first — usually the main table in public dataset zips
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[0][1]
    suffix = Path(chosen).suffix.lower()
    payload = zf.read(chosen)
    buf = BytesIO(payload)
    buf.name = Path(chosen).name  # type: ignore[attr-defined]

    if suffix in {".csv", ".tsv"}:
        return _read_csv_with_encoding(buf)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buf)
    if suffix == ".json":
        return _read_json_file(buf)
    raise ValueError(f"Unsupported member in ZIP: {chosen}")


def read_raw_file(source: Any) -> pd.DataFrame:
    """
    Read uploaded file into a raw pandas DataFrame without normalization.

    Supports: .csv, .tsv, .xlsx, .xls, .json, .zip (containing one of the above)

    Parameters
    ----------
    source : UploadedFile | str | Path
        File source.

    Returns
    -------
    pd.DataFrame
        Raw dataframe exactly as stored in the file.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".zip":
            return _read_zip_archive(path)
        if suffix in {".csv", ".tsv"}:
            return _read_csv_with_encoding(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".json":
            return _read_json_file(path)
        raise ValueError(f"Unsupported file format: {suffix}")

    filename = getattr(source, "name", "upload.csv").lower()

    if filename.endswith(".zip"):
        return _read_zip_archive(source)
    if filename.endswith(".json"):
        return _read_json_file(source)
    if filename.endswith(".tsv"):
        return _read_csv_with_encoding(source)
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(source)
    if filename.endswith(".csv"):
        return _read_csv_with_encoding(source)

    # Unknown extension — try CSV then Excel then JSON
    for reader in (_read_csv_with_encoding, pd.read_excel, _read_json_file):
        try:
            source.seek(0)
            return reader(source)
        except Exception:
            source.seek(0)
            continue

    raise ValueError(
        "Unsupported file format. Please upload CSV, TSV, Excel (.xlsx), JSON, or ZIP."
    )


def _read_json_file(source: Any) -> pd.DataFrame:
    """Parse JSON file — supports array of objects or {records: [...]} shape."""
    if isinstance(source, (str, Path)):
        content = Path(source).read_text(encoding="utf-8")
    else:
        source.seek(0)
        content = source.read().decode("utf-8")
        source.seek(0)

    data = json.loads(content)
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        for key in ("records", "data", "reviews", "items", "results"):
            if key in data and isinstance(data[key], list):
                return pd.DataFrame(data[key])
        return pd.DataFrame([data])
    raise ValueError("JSON must be an array of objects or an object with a records/data key.")


def _score_text_column(series: pd.Series) -> float:
    """Score how likely a column contains review text (higher = more likely)."""
    sample = series.dropna().astype(str).head(200)
    if len(sample) == 0:
        return 0.0
    avg_len = sample.str.len().mean()
    # Penalize columns that look like IDs (mostly numeric/short)
    numeric_ratio = sample.str.match(r"^[\d\-\.]+$").mean()
    return avg_len * (1.0 - numeric_ratio)


def _fuzzy_match_column(normalized_cols: dict[str, str], target: str) -> str | None:
    """Layer 2: match column by substring patterns."""
    patterns = FUZZY_PATTERNS.get(target, [])
    best_col: str | None = None
    best_score = 0.0

    for norm_name, original in normalized_cols.items():
        for pattern in patterns:
            if pattern in norm_name:
                score = len(pattern) / max(len(norm_name), 1)
                if score > best_score:
                    best_score = score
                    best_col = original

    return best_col


def _detect_platform(normalized_col_names: set[str]) -> str:
    """Detect known export format from column name fingerprint."""
    for platform, fingerprint in PLATFORM_FINGERPRINTS.items():
        matches = sum(1 for fp in fingerprint if _normalize_col_name(fp) in normalized_col_names)
        if matches >= 2 or (matches == 1 and len(fingerprint) == 1):
            return platform
    return "Custom / Unknown Format"


def auto_resolve_columns(raw_df: pd.DataFrame) -> tuple[dict[str, str | None], str, str]:
    """
    Three-layer column auto-detection.

    Returns
    -------
    tuple[dict, str, str]
        (column_mapping, detected_platform, confidence_level)
    """
    lower_cols = {_normalize_col_name(c): c for c in raw_df.columns}
    resolved: dict[str, str | None] = {f: None for f in MAPPABLE_FIELDS}

    # Layer 1: exact alias match
    for target, candidates in COLUMN_ALIASES.items():
        for candidate in candidates:
            norm = _normalize_col_name(candidate)
            if norm in lower_cols:
                resolved[target] = lower_cols[norm]
                break

    # Layer 2: fuzzy substring match (only for still-unresolved fields)
    for target in MAPPABLE_FIELDS:
        if resolved[target] is None:
            resolved[target] = _fuzzy_match_column(lower_cols, target)

    # Layer 3: heuristic text column — longest average string length
    if resolved["text"] is None:
        text_scores = {
            col: _score_text_column(raw_df[col])
            for col in raw_df.columns
            if raw_df[col].dtype == object or pd.api.types.is_string_dtype(raw_df[col])
        }
        if text_scores:
            best_col = max(text_scores, key=text_scores.get)  # type: ignore[arg-type]
            if text_scores[best_col] >= 8:
                resolved["text"] = best_col

    platform = _detect_platform(set(lower_cols.keys()))

    # Confidence scoring
    if resolved["text"] is not None:
        text_via = "exact" if any(
            _normalize_col_name(c) in lower_cols
            for c in COLUMN_ALIASES["text"]
            if lower_cols.get(_normalize_col_name(c)) == resolved["text"]
        ) else "fuzzy/heuristic"
        if text_via == "exact" and platform != "Custom / Unknown Format":
            confidence = "high"
        elif resolved["text"] is not None:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        confidence = "low"

    return resolved, platform, confidence


def suggest_text_columns(raw_df: pd.DataFrame, top_n: int = 5) -> list[str]:
    """Return columns ranked by likelihood of containing review text."""
    scores = {
        col: _score_text_column(raw_df[col])
        for col in raw_df.columns
    }
    ranked = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [c for c in ranked[:top_n] if scores[c] > 0]


# Minimum usable text length (chars). Short survey answers still count.
MIN_TEXT_CHARS = 5


def normalize_rating_to_5(series: pd.Series) -> pd.Series:
    """
    Map heterogeneous rating scales onto an approximate 1–5 star range.

    Handles common 1–5, 0–5, 1–10, 0–10, and 0–100 exports without requiring
    the uploader to pre-normalize.
    """
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return s

    mx = float(valid.max())
    mn = float(valid.min())

    # Already looks like stars
    if mx <= 5.5 and mn >= 0:
        out = s.clip(lower=1.0, upper=5.0)
        out = out.where(out.isna() | (out >= 1.0), 1.0)
        return out

    # 0–10 / 1–10
    if mx <= 10.5:
        span = max(mx - mn, 1e-6)
        return (1.0 + (s - mn) / span * 4.0).clip(1.0, 5.0)

    # 0–100 style
    if mx <= 100.5:
        return (1.0 + s.clip(0, 100) / 100.0 * 4.0).clip(1.0, 5.0)

    # Unknown high-cardinality — percentile rank into 1–5
    ranks = s.rank(pct=True, method="average")
    return (1.0 + ranks * 4.0).clip(1.0, 5.0)


def normalize_raw_reviews(
    raw_df: pd.DataFrame,
    column_mapping: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """
    Normalize raw dataframe to InsightOptima internal schema using column mapping.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw uploaded dataframe.
    column_mapping : dict | None
        Explicit column mapping. If None, runs auto_resolve_columns().

    Returns
    -------
    pd.DataFrame
        Normalized dataframe ready for semantic encoding.

    Raises
    ------
    ValueError
        If text column is not mapped or no valid rows remain.
    """
    if column_mapping is None:
        column_mapping, _, _ = auto_resolve_columns(raw_df)

    text_col = column_mapping.get("text")
    if not text_col or text_col not in raw_df.columns:
        raise ValueError(
            "Review text column is required. "
            "Use the column mapping panel to select which column contains user comments."
        )

    text_parts: list[pd.Series] = []
    if column_mapping.get("summary") and column_mapping["summary"] in raw_df.columns:
        text_parts.append(raw_df[column_mapping["summary"]].astype(str).str.strip())
    text_parts.append(raw_df[text_col].astype(str).str.strip())

    combined_text = text_parts[0]
    for part in text_parts[1:]:
        combined_text = combined_text + ". " + part

    combined_text = combined_text.str.replace(r"\s+", " ", regex=True).str.strip()
    combined_text = combined_text.replace({"nan": "", "None": "", ".": ""}).str.strip()

    valid_mask = combined_text.str.len() >= MIN_TEXT_CHARS
    if not valid_mask.any():
        raise ValueError(
            f"No valid review texts found after mapping. "
            f"Ensure the selected text column contains user comments "
            f"(min {MIN_TEXT_CHARS} characters)."
        )

    raw_df = raw_df.loc[valid_mask].copy()
    combined_text = combined_text.loc[valid_mask]

    df = pd.DataFrame()
    df["text"] = combined_text.values

    id_col = column_mapping.get("review_id")
    if id_col and id_col in raw_df.columns:
        df["review_id"] = raw_df[id_col].astype(str).values
    else:
        df["review_id"] = [f"REV-{i + 1:05d}" for i in range(len(df))]

    rating_col = column_mapping.get("rating")
    if rating_col and rating_col in raw_df.columns:
        df["rating"] = normalize_rating_to_5(raw_df[rating_col]).values
        df.attrs["rating_normalized"] = True
    else:
        df["rating"] = np.nan

    date_col = column_mapping.get("created_at")
    if date_col and date_col in raw_df.columns:
        # Handle Unix timestamps (common in Amazon dataset)
        raw_dates = raw_df[date_col]
        if pd.api.types.is_numeric_dtype(raw_dates):
            df["created_at"] = pd.to_datetime(raw_dates, unit="s", errors="coerce").values
        else:
            df["created_at"] = pd.to_datetime(raw_dates, errors="coerce").values
    else:
        df["created_at"] = pd.Timestamp.now()

    for optional_field in ("lifecycle_stage", "sentiment", "rage_index", "topic"):
        col = column_mapping.get(optional_field)
        if col and col in raw_df.columns:
            if optional_field in ("sentiment", "rage_index"):
                df[optional_field] = pd.to_numeric(raw_df[col], errors="coerce").values
            else:
                df[optional_field] = raw_df[col].astype(str).values

    return df.reset_index(drop=True)


def load_reviews_file(
    source: Any,
    column_mapping: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """
    Load and normalize a review file (backward-compatible entry point).

    Parameters
    ----------
    source : UploadedFile | str | Path
        File to load.
    column_mapping : dict | None
        Optional explicit column mapping.

    Returns
    -------
    pd.DataFrame
        Normalized raw review dataframe (not yet semantically encoded).
    """
    raw_df = read_raw_file(source)
    if column_mapping is None:
        column_mapping, _, _ = auto_resolve_columns(raw_df)
    return normalize_raw_reviews(raw_df, column_mapping)
