"""
Multilingual NLP utilities for InsightOptima review analysis.

Detects review language and computes sentiment using language-appropriate models:
- English  → VADER
- Chinese  → SnowNLP
- Other    → XLM-RoBERTa multilingual sentiment (lazy-loaded)
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

# Lazy-loaded models
_vader_analyzer: Any = None
_sentiment_pipeline: Any = None


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def _get_multilingual_pipeline():
    """Lazy-load CardiffNLP XLM-RoBERTa sentiment model (supports 100+ languages)."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline

        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            top_k=None,
            truncation=True,
            max_length=512,
        )
    return _sentiment_pipeline


def detect_language(text: str) -> str:
    """
    Detect ISO 639-1 language code for a review text.

    Returns 'unknown' if detection fails.
    """
    if not text or len(text.strip()) < 10:
        return "unknown"

    try:
        from langdetect import detect

        return detect(text)
    except Exception:
        return "unknown"


def detect_corpus_languages(texts: list[str], top_n: int = 5) -> list[tuple[str, int]]:
    """
    Detect language distribution across a corpus.

    Returns list of (language_code, count) sorted by frequency.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        lang = detect_language(str(text))
        counts[lang] += 1
    return counts.most_common(top_n)


def is_primarily_non_english(texts: list[str], threshold: float = 0.35) -> bool:
    """Return True if more than threshold fraction of texts are non-English."""
    if not texts:
        return False
    langs = [detect_language(str(t)) for t in texts[:200]]
    non_en = sum(1 for lang in langs if lang not in ("en", "unknown"))
    return (non_en / len(langs)) >= threshold


def compute_sentiment_multilingual(
    text: str,
    rating: float | None = None,
    lang: str | None = None,
) -> float:
    """
    Compute sentiment score (-1.0 to 1.0) using language-appropriate model.

    Blends with star rating when available (60% rating + 40% text).
    """
    lang = lang or detect_language(text)

    if lang.startswith("zh"):
        try:
            from snownlp import SnowNLP

            score_01 = SnowNLP(text).sentiments
            text_sentiment = (score_01 - 0.5) * 2.0
        except Exception:
            text_sentiment = _vader_fallback(text)
    elif lang == "en":
        text_sentiment = _vader_fallback(text)
    else:
        text_sentiment = _multilingual_transformer_sentiment(text)

    if rating is not None and not np.isnan(rating):
        clamped = float(np.clip(rating, 1.0, 5.0))
        rating_sentiment = (clamped - 3.0) / 2.0
        return round(0.6 * rating_sentiment + 0.4 * text_sentiment, 3)

    return round(float(text_sentiment), 3)


def _vader_fallback(text: str) -> float:
    """VADER compound score for English or fallback."""
    return _get_vader().polarity_scores(text)["compound"]


def _multilingual_transformer_sentiment(text: str) -> float:
    """
    Map XLM-RoBERTa 3-class output to -1..1 scale.

    Labels: negative, neutral, positive
    """
    try:
        pipe = _get_multilingual_pipeline()
        results = pipe(text[:512])[0]
        label_scores = {r["label"].lower(): r["score"] for r in results}
        pos = label_scores.get("positive", 0.0)
        neg = label_scores.get("negative", 0.0)
        neu = label_scores.get("neutral", 0.0)
        return float(pos - neg + neu * 0.0)  # neutral contributes 0
    except Exception:
        return _vader_fallback(text)
