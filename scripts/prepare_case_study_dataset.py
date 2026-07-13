"""
Prepare the portfolio case-study dataset from a PUBLIC source.

Source: Drug Review Dataset (Drugs.com) — UCI Machine Learning Repository
DOI: https://doi.org/10.24432/C5SK5S
UCI page: https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com
Direct files: https://archive.ics.uci.edu/ml/machine-learning-databases/00462/
License: CC BY 4.0 (cite the authors when using)

Usage:
    python scripts/prepare_case_study_dataset.py
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_CSV = DATA_DIR / "case_study_drug_reviews.csv"
OUT_META = DATA_DIR / "case_study_drug_reviews.SOURCE.json"
SAMPLE_SIZE = 1500
RANDOM_SEED = 42

# Stable public UCI mirror (verified HTTP 200)
UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"

SOURCE_META = {
    "title": "Drug Review Dataset (Drugs.com)",
    "domain": "public_health",
    "original_publisher": "UCI Machine Learning Repository",
    "doi": "https://doi.org/10.24432/C5SK5S",
    "uci_url": "https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com",
    "download_url": UCI_ZIP_URL,
    "content_origin": "https://www.drugs.com/ (patient reviews collected for research)",
    "license": "CC BY 4.0 — cite Kallumadi & Gräßer (2018)",
    "citation": (
        "Kallumadi, Surya and Gräßer, Felix. (2018). "
        "Drug Review Dataset (Drugs.com). UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5SK5S"
    ),
    "subset_rows": SAMPLE_SIZE,
    "subset_seed": RANDOM_SEED,
    "notes": (
        "This file is a reproducible random subset for InsightOptima demos. "
        "Full corpus remains available from UCI."
    ),
}


def download_uci_train() -> pd.DataFrame:
    """Download drugsCom_raw.zip from UCI and read the train TSV."""
    print(f"Downloading public UCI zip:\n  {UCI_ZIP_URL}")
    with urlopen(UCI_ZIP_URL, timeout=120) as resp:
        payload = resp.read()
    print(f"Downloaded {len(payload):,} bytes")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
        print(f"Archive contains: {names}")
        train_name = next((n for n in names if "train" in n.lower() and n.endswith(".tsv")), None)
        if train_name is None:
            train_name = next((n for n in names if n.endswith(".tsv")), None)
        if train_name is None:
            raise FileNotFoundError(f"No TSV found in zip. Files: {names}")
        with zf.open(train_name) as fh:
            df = pd.read_csv(fh, sep="\t")
    print(f"Loaded {train_name}: {len(df):,} rows, columns={list(df.columns)}")
    return df


def normalize_for_app(df: pd.DataFrame) -> pd.DataFrame:
    """Map Drugs.com columns to InsightOptima-friendly names."""
    colmap = {str(c).lower().strip(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for name in names:
            if name.lower() in colmap:
                return colmap[name.lower()]
        return None

    text_col = pick("review", "text", "comment", "review_text")
    if text_col is None:
        raise ValueError(f"No review text column found. Columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["feedback_id"] = [f"DRUG-{i + 1:05d}" for i in range(len(df))]
    # Drugs.com reviews often wrap text in quotes and HTML entities
    text = (
        df[text_col]
        .astype(str)
        .str.replace(r'\\"', '"', regex=True)
        .str.replace(r"&#039;", "'", regex=False)
        .str.replace(r"&amp;", "&", regex=False)
        .str.strip()
    )
    out["feedback"] = text

    drug_col = pick("drugName", "drug_name", "drug")
    condition_col = pick("condition")
    if drug_col and condition_col:
        out["title"] = df[drug_col].astype(str) + " — " + df[condition_col].astype(str)
    elif drug_col:
        out["title"] = df[drug_col].astype(str)
    else:
        out["title"] = "Drug review"

    rating_col = pick("rating", "score", "stars")
    if rating_col:
        raw = pd.to_numeric(df[rating_col], errors="coerce")
        # Drugs.com ratings are 1–10; map to ~1–5 for familiar UX metrics
        if raw.dropna().max() and float(raw.dropna().max()) > 5:
            out["rating"] = (raw / 2.0).clip(1, 5).round(1)
        else:
            out["rating"] = raw

    date_col = pick("date", "created_at", "timestamp")
    if date_col:
        out["created_at"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    out["channel"] = "drugs.com patient review"
    out["product_area"] = "Public health — medication experience"
    out["source_dataset"] = "UCI Drug Review Dataset (Drugs.com)"
    return out.dropna(subset=["feedback"]).reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = download_uci_train()
    cleaned = normalize_for_app(raw)
    # Prefer a mix of low/high ratings so negative themes appear in the demo
    if "rating" in cleaned.columns:
        low = cleaned[cleaned["rating"] <= 2.5]
        high = cleaned[cleaned["rating"] > 2.5]
        n_low = min(len(low), int(SAMPLE_SIZE * 0.55))
        n_high = min(len(high), SAMPLE_SIZE - n_low)
        part_low = low.sample(n=n_low, random_state=RANDOM_SEED) if n_low else low
        part_high = high.sample(n=n_high, random_state=RANDOM_SEED) if n_high else high
        subset = pd.concat([part_low, part_high], ignore_index=True)
        subset = subset.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        subset = cleaned.sample(n=min(SAMPLE_SIZE, len(cleaned)), random_state=RANDOM_SEED)

    SOURCE_META["subset_rows"] = int(len(subset))
    subset.to_csv(OUT_CSV, index=False, encoding="utf-8")
    OUT_META.write_text(json.dumps(SOURCE_META, indent=2), encoding="utf-8")
    print(f"Wrote {len(subset):,} rows -> {OUT_CSV}")
    print(f"Source metadata -> {OUT_META}")
    print(f"Cite: {SOURCE_META['citation']}")


if __name__ == "__main__":
    main()
