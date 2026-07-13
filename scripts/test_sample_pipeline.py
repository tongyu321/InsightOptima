"""Headless smoke test: load sample data and print roadmap topics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loader import build_implementation_roadmap, encode_reviews, load_reviews_file

SAMPLE = ROOT / "data" / "sample_reviews.csv"
if not SAMPLE.exists():
    SAMPLE = ROOT / "data" / "sample_reviews.xlsx"

print(f"Loading {SAMPLE}...")
raw = load_reviews_file(SAMPLE)
print(f"Rows: {len(raw)}")

def progress(pct, msg):
    if pct in (0.0, 0.25, 0.5, 0.75, 1.0):
        print(f"  [{pct:.0%}] {msg}")

encoded = encode_reviews(raw, progress_callback=progress)
print(f"Encoded: {len(encoded)} reviews, topics: {encoded['topic'].nunique()}")
print(f"Topic engine: {encoded.attrs.get('topic_method', '?')}")
print(f"Languages: {encoded['language'].value_counts().head(5).to_dict() if 'language' in encoded.columns else 'N/A'}")

print("\nTop 10 roadmap pain points:")
roadmap = build_implementation_roadmap(encoded)
for _, row in roadmap.head(10).iterrows():
    print(f"  [{row['Priority']}] {row['Core Pain Point']}")

print("\nSample topic labels:")
for topic in encoded["topic"].value_counts().head(8).index:
    print(f"  - {topic}")
