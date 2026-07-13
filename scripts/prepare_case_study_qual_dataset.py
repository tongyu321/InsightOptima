"""
Prepare the QUALITATIVE portfolio case-study dataset from a PUBLIC source.

Source: Qualitative-coded open-ended survey answers — Zenodo
Dataset: Data for "\"PubPeer is okay, but …\": researchers' perceptions of
         post-publication reviews"
DOI: https://doi.org/10.5281/zenodo.20413424
License: CC BY 4.0

This is the qualitative strand for a mixed-methods portfolio pair:
  - Quant case: UCI Drugs.com reviews (ratings + large-n themes)
  - Qual case:  open-ended survey verbatims with published coding scheme

Usage:
    python scripts/prepare_case_study_qual_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_CSV = DATA_DIR / "case_study_pubpeer_qual.csv"
OUT_META = DATA_DIR / "case_study_pubpeer_qual.SOURCE.json"
OUT_SCHEME = DATA_DIR / "case_study_pubpeer_coding_scheme.csv"

ZENODO_RECORD = "20413424"
QUAL_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/"
    "Qualitative-coded-answers.csv/content"
)
SCHEME_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/"
    "Coding%20Scheme.csv/content"
)

# Human-readable labels for survey open-ended prompts
QUESTION_LABELS = {
    "Q3explain": "Explain PubPeer concern / citation context",
    "End": "Final open comment",
    "Q4noWontOther": "Why not use PubPeer (other)",
    "Q4noButWillother": "Not yet, but may use (other)",
    "Q4yesOther": "Why use PubPeer (other)",
}

SOURCE_META = {
    "title": (
        'Qualitative-coded answers — "PubPeer is okay, but …": '
        "researchers' perceptions of post-publication reviews"
    ),
    "strand": "qualitative",
    "mixed_methods_pair": "case_study_drug_reviews.csv (quantitative strand)",
    "domain": "research_integrity / researcher_perceptions",
    "original_publisher": "Zenodo",
    "doi": "https://doi.org/10.5281/zenodo.20413424",
    "zenodo_url": "https://zenodo.org/records/20413424",
    "download_url": QUAL_URL,
    "coding_scheme_url": SCHEME_URL,
    "license": "CC BY 4.0 — cite Hepkema & Bordignon",
    "citation": (
        "Hepkema, Wytske, & Bordignon, Frederique. (2026). "
        'Data for "\\"PubPeer is okay, but ...\\": researchers\' perceptions '
        "of post-publication reviews\" [Data set]. Zenodo. "
        "https://doi.org/10.5281/zenodo.20413424"
    ),
    "notes": (
        "Open-ended survey verbatims with published thematic codes "
        "(Coding Scheme.csv). InsightOptima re-clusters the text for demo; "
        "author codes are retained in author_codes for comparison."
    ),
}


def download_qual_csv(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading public Zenodo file:\n  {QUAL_URL}")
    urlretrieve(QUAL_URL, dest)
    print(f"Saved {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def download_coding_scheme(dest: Path) -> Path:
    print(f"Downloading coding scheme:\n  {SCHEME_URL}")
    urlretrieve(SCHEME_URL, dest)
    print(f"Saved {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def read_qual_csv(path: Path) -> pd.DataFrame:
    """PubPeer export uses semicolon separators and UTF-8 BOM."""
    return pd.read_csv(path, encoding="utf-8-sig", sep=";", engine="python")


def normalize_for_app(df: pd.DataFrame) -> pd.DataFrame:
    if "Open answers" not in df.columns:
        raise ValueError(f"Expected 'Open answers' column. Got: {list(df.columns)}")

    text = df["Open answers"].astype(str).str.strip()
    mask = text.notna() & (text != "") & (text.str.lower() != "nan")
    df = df.loc[mask].copy().reset_index(drop=True)
    text = text.loc[mask].reset_index(drop=True)

    out = pd.DataFrame()
    if "AnswerID" in df.columns:
        out["feedback_id"] = df["AnswerID"].astype(str).str.strip()
    else:
        out["feedback_id"] = [f"QUAL-{i + 1:05d}" for i in range(len(df))]

    out["feedback"] = text

    q = (
        df["Question"].astype(str)
        if "Question" in df.columns
        else pd.Series(["open"] * len(df))
    )
    labels = q.map(lambda x: QUESTION_LABELS.get(x, x))
    short = (
        df["short responseID"].astype(str)
        if "short responseID" in df.columns
        else out["feedback_id"]
    )
    out["title"] = labels.astype(str) + " — " + short.astype(str)

    if "Coding new" in df.columns:
        out["author_codes"] = df["Coding new"].astype(str).str.strip()

    out["channel"] = "open-ended survey (qualitative)"
    out["product_area"] = labels.values
    out["source_dataset"] = (
        "Zenodo 10.5281/zenodo.20413424 — PubPeer perceptions (qualitative)"
    )
    # No star ratings — text-only sentiment (explicit qualitative strand)
    return out.reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DATA_DIR / "tmp_qual" / "Qualitative-coded-answers.csv"
    download_qual_csv(tmp)
    download_coding_scheme(OUT_SCHEME)

    raw = read_qual_csv(tmp)
    print(f"Loaded {len(raw):,} rows, columns={list(raw.columns)}")
    cleaned = normalize_for_app(raw)

    SOURCE_META["subset_rows"] = int(len(cleaned))
    cleaned.to_csv(OUT_CSV, index=False, encoding="utf-8")
    OUT_META.write_text(json.dumps(SOURCE_META, indent=2), encoding="utf-8")
    print(f"Wrote {len(cleaned):,} rows -> {OUT_CSV}")
    print(f"Source metadata -> {OUT_META}")
    print(f"Coding scheme -> {OUT_SCHEME}")
    print(f"Cite: {SOURCE_META['citation']}")


if __name__ == "__main__":
    main()
