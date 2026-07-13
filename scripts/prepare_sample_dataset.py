"""
Prepare a real sample review dataset for InsightOptima.

Downloads the Amazon Fine Food Reviews dataset from Hugging Face (568K real reviews,
CC BY-SA 4.0) and saves a 2,000-row subset as CSV + Excel in data/.

Usage:
    python scripts/prepare_sample_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_SIZE = 2000
RANDOM_SEED = 42

# Hugging Face dataset: Amazon Fine Food Reviews (Stanford SNAP, CC BY-SA 4.0)
HF_DATASET = "PJ2005/amazon-fine-food-reviews"


def download_sample(n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Stream and sample real Amazon food reviews from Hugging Face.

    Parameters
    ----------
    n : int
        Number of reviews to sample.

    Returns
    -------
    pd.DataFrame
        Sampled review dataframe.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing huggingface datasets library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        from datasets import load_dataset

    print(f"Streaming {n:,} reviews from Hugging Face ({HF_DATASET})...")
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if i + 1 >= n:
            break

    return pd.DataFrame(rows)


def save_sample(df: pd.DataFrame) -> tuple[Path, Path]:
    """
    Save sample to data/sample_reviews.csv and data/sample_reviews.xlsx.

    Returns
    -------
    tuple[Path, Path]
        Paths to CSV and Excel files.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / "sample_reviews.csv"
    xlsx_path = DATA_DIR / "sample_reviews.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    return csv_path, xlsx_path


def main() -> None:
    """Download, save, and print summary of sample dataset."""
    df = download_sample(SAMPLE_SIZE)
    csv_path, xlsx_path = save_sample(df)

    print(f"\nSaved {len(df):,} real reviews:")
    print(f"  CSV:   {csv_path}")
    print(f"  Excel: {xlsx_path}")
    print(f"\nColumns: {list(df.columns)}")
    if "Score" in df.columns:
        print(f"Rating distribution:\n{df['Score'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
