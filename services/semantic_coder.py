"""
Local semantic coding engine for InsightOptima.

Transforms raw review text into structured UXR metrics:
- lifecycle_stage (Onboarding / Core Feature Activation / Daily Retention)
- sentiment (-1.0 to 1.0)
- rage_index (0-100)
- topic (data-driven pain point label)
- is_negative (boolean)

Uses VADER sentiment + keyword heuristics + BERTopic semantic clustering
(with TF-IDF fallback for small datasets).

Future: swap encoder backend to OpenAI/Anthropic via semantic_coding_with_openai().
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from services.topic_modeling import extract_topic_labels
from services.topic_merge import apply_topic_merge
from services.credibility import compute_credibility_report
from services.analysis_cache import (
    corpus_fingerprint,
    load_cached_encoding,
    save_cached_encoding,
)

# ---------------------------------------------------------------------------
# Lifecycle stage keyword signals (UXR coding scheme)
# ---------------------------------------------------------------------------

LIFECYCLE_KEYWORDS: dict[str, list[str]] = {
    "Onboarding": [
        "sign up", "signup", "register", "registration", "tutorial", "onboard",
        "first time", "getting started", "setup", "set up", "login", "log in",
        "verify", "verification", "password", "account create", "welcome",
        "intro", "guide", "instructions", "confusing at first", "hard to start",
        "sso setup", "saml", "metadata", "first-time", "checklist", "wizard",
        "idp", "certificate rotation", "onboarding checklist",
    ],
    "Core Feature Activation": [
        "feature", "function", "export", "import", "search", "filter", "dashboard",
        "payment", "checkout", "purchase", "bug", "crash", "broken", "error",
        "doesn't work", "does not work", "not working", "slow", "loading",
        "performance", "billing", "subscription", "upgrade", "download", "upload",
        "sync", "integrate", "api", "quality", "taste", "flavor", "product",
        "package", "delivery", "shipping", "refund", "return", "price", "expensive",
        "scim", "provisioning", "deprovision", "group-to-role", "permission",
        "role mapping", "audit log", "compliance", "soc2", "access control",
        "nested groups", "attribute mapping",
    ],
    "Daily Retention": [
        "daily", "every day", "notification", "reminder", "habit", "routine",
        "keep using", "come back", "loyal", "long time", "years", "month",
        "update", "new version", "dark mode", "widget", "customize", "preference",
        "recommend", "favorite", "regular", "subscription renew", "cancel",
        "alert", "monitoring", "operations", "sla", "overnight", "status page",
    ],
}

# Intensifier lexicon for rage/frustration scoring (UX psychology: anger > dissatisfaction)
RAGE_INTENSIFIERS: list[str] = [
    "terrible", "awful", "horrible", "hate", "worst", "disgusting", "furious",
    "angry", "outraged", "unacceptable", "scam", "fraud", "garbage", "trash",
    "useless", "waste", "never again", "refund", "lawsuit", "ridiculous",
    "pathetic", "disappointed", "frustrated", "frustrating", "infuriating",
    "unusable", "broken", "disaster", "nightmare", "fed up", "sick of",
]

NEGATIVE_THRESHOLD = -0.05  # VADER compound below this → negative review
# Soft floor: when corpus has almost no VADER-negatives, still surface friction
MIN_NEGATIVE_SHARE = 0.03
FRICTION_SENTIMENT_FLOOR = 0.15  # mild-negative / low-neutral also eligible as friction

COMPLAINT_CUES: tuple[str, ...] = (
    "terrible", "awful", "horrible", "worst", "disgusting", "disappointed",
    "waste", "refund", "never again", "do not recommend", "don't recommend",
    "not recommend", "threw", "garbage", "trash", "useless", "ruined",
    "failed", "failure", "broken", "doesn't work", "does not work",
    "bland", "gross", "nasty", "inedible", "overpriced", "scam",
)


def _get_analyzer() -> SentimentIntensityAnalyzer:
    """Lazy-init VADER analyzer (cached at module level)."""
    if not hasattr(_get_analyzer, "_instance"):
        _get_analyzer._instance = SentimentIntensityAnalyzer()  # type: ignore[attr-defined]
    return _get_analyzer._instance  # type: ignore[attr-defined]


def _rating_to_sentiment(rating: float) -> float:
    """
    Map star rating (1-5) to sentiment scale (-1.0 to 1.0).

    Parameters
    ----------
    rating : float
        Star rating value.

    Returns
    -------
    float
        Normalized sentiment score.
    """
    clamped = float(np.clip(rating, 1.0, 5.0))
    return round((clamped - 3.0) / 2.0, 3)  # 1→-1.0, 3→0.0, 5→1.0


def _row_is_negative(text: str, sentiment: float, rating: float | None) -> bool:
    """
    Multi-signal negativity — not VADER-only.

    A row counts as friction if any of:
    - text sentiment below threshold
    - star rating ≤ 2.5 (on normalized 1–5)
    - explicit complaint cues in text
    """
    if sentiment < NEGATIVE_THRESHOLD:
        return True
    if rating is not None and not np.isnan(rating) and float(rating) <= 2.5:
        return True
    lower = text.lower()
    if any(cue in lower for cue in COMPLAINT_CUES):
        return True
    return False


def _ensure_friction_coverage(result: pd.DataFrame) -> pd.DataFrame:
    """
    If almost no negatives were flagged, promote lowest-sentiment rows so
    theme clustering still has a workable friction pool (domain-agnostic).
    """
    n = len(result)
    if n == 0 or "is_negative" not in result.columns:
        return result
    neg_n = int(result["is_negative"].sum())
    if neg_n / max(n, 1) >= MIN_NEGATIVE_SHARE and neg_n >= 5:
        return result

    target = max(5, int(np.ceil(n * MIN_NEGATIVE_SHARE)))
    target = min(target, max(5, n // 5), n)

    scored = result.copy()
    rating = pd.to_numeric(scored["rating"], errors="coerce") if "rating" in scored.columns else None
    sentiment = pd.to_numeric(scored["sentiment"], errors="coerce")
    score = sentiment.fillna(0.0)
    if rating is not None and rating.notna().any():
        score = score + (rating.fillna(3.0) - 3.0) * 0.35

    eligible = scored.index[sentiment.fillna(0.0) < FRICTION_SENTIMENT_FLOOR]
    if len(eligible) == 0:
        eligible = scored.index

    order = score.loc[eligible].sort_values().index.tolist()
    promote = order[: max(0, target - neg_n)]
    if promote:
        result.loc[promote, "is_negative"] = True
        result.attrs["friction_backfill"] = len(promote)
    return result


def _compute_sentiment(text: str, rating: float | None = None) -> float:
    """Compute sentiment using multilingual NLP (language auto-detected)."""
    from services.multilingual_nlp import compute_sentiment_multilingual

    return compute_sentiment_multilingual(text, rating=rating)


def _compute_rage_index(text: str, sentiment: float, rating: float | None = None) -> float:
    """
    Compute rage/frustration index (0-100) from sentiment, rating, and intensifier words.

    Higher scores indicate stronger user anger/frustration — not just mild dissatisfaction.
    """
    text_lower = text.lower()

    # Base rage inversely proportional to sentiment
    base = (1.0 - sentiment) / 2.0 * 100  # sentiment -1 → 100, sentiment 1 → 0

    if rating is not None and not np.isnan(rating):
        rating_rage = (1.0 - (float(rating) - 1.0) / 4.0) * 100
        base = 0.55 * rating_rage + 0.45 * base

    # Boost for explicit rage intensifiers
    intensifier_hits = sum(1 for word in RAGE_INTENSIFIERS if word in text_lower)
    boost = min(intensifier_hits * 6, 30)

    # Exclamation / caps as frustration signals
    if text.count("!") >= 2:
        boost += 5
    if len(text) > 20 and sum(1 for c in text if c.isupper()) / len(text) > 0.4:
        boost += 5

    return round(float(np.clip(base + boost, 0, 100)), 1)


def _classify_lifecycle_stage(text: str) -> tuple[str, bool]:
    """
    Classify review into user lifecycle stage using keyword signal matching.

    Returns (stage, had_keyword_hit). No hit → ("General feedback", False).
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for stage, keywords in LIFECYCLE_KEYWORDS.items():
        scores[stage] = sum(1 for kw in keywords if kw in text_lower)

    best_stage = max(scores, key=lambda s: scores[s])
    if scores[best_stage] == 0:
        return "General feedback", False
    return best_stage, True


def _assign_positive_topics(texts: list[str]) -> list[str]:
    """Assign retention-driver labels to positive reviews via keyword matching."""
    positive_signals = {
        "Product quality praise": ["delicious", "tasty", "quality", "fresh", "great taste", "love the"],
        "Value for money": ["worth", "price", "value", "affordable", "deal", "cheap"],
        "Fast delivery / shipping": ["fast shipping", "quick delivery", "arrived", "shipping"],
        "Customer support": ["support", "customer service", "helpful", "responsive"],
        "Ease of use": ["easy", "simple", " intuitive", "user friendly", "convenient"],
        "Reliability / consistency": ["reliable", "consistent", "always", "trust", "depend"],
    }

    results: list[str] = []
    for text in texts:
        text_lower = text.lower()
        best_topic = "Positive experience"
        best_score = 0
        for topic, keywords in positive_signals.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_topic = topic
        results.append(best_topic)
    return results


def encode_reviews(
    df: pd.DataFrame,
    progress_callback: Callable[[float, str], None] | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Run full semantic encoding pipeline on a normalized review dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized reviews with at minimum: review_id, text.
        Optional: rating, lifecycle_stage, sentiment, rage_index, topic.
    progress_callback : callable | None
        Optional callback(percent: float, message: str) for UI progress updates.
    use_cache : bool
        If True, reuse disk-cached encodings for identical corpora.

    Returns
    -------
    pd.DataFrame
        Fully encoded review dataframe with all analysis fields populated.
    """
    result = df.copy()
    n = len(result)

    def report(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    fingerprint = corpus_fingerprint(result) if use_cache and "text" in result.columns else ""
    if fingerprint:
        cached = load_cached_encoding(fingerprint)
        if cached is not None and len(cached) == n:
            report(1.0, f"Loaded cached analysis ({n:,} reviews).")
            cached.attrs["cache_hit"] = True
            cached.attrs["cache_key"] = fingerprint
            if "credibility" not in cached.attrs:
                cached.attrs["credibility"] = compute_credibility_report(cached)
            if "lifecycle_mode" not in cached.attrs and "lifecycle_stage" in cached.columns:
                general_share = float(
                    (cached["lifecycle_stage"].astype(str) == "General feedback").mean()
                )
                cached.attrs["lifecycle_mode"] = (
                    "disabled" if general_share >= 0.85 else "active"
                )
            return cached

    report(0.0, "Starting semantic encoding...")

    # Step 1: Per-row sentiment, rage, lifecycle
    sentiments: list[float] = []
    rage_scores: list[float] = []
    lifecycle_stages: list[str] = []
    is_negative_flags: list[bool] = []
    lifecycle_hits = 0

    for pos, (idx, row) in enumerate(result.iterrows()):
        text = str(row["text"])
        rating = row.get("rating", np.nan)
        rating_val = float(rating) if pd.notna(rating) else None

        pre_encoded = (
            "sentiment" in result.columns
            and "rage_index" in result.columns
            and "lifecycle_stage" in result.columns
            and pd.notna(row.get("sentiment"))
            and pd.notna(row.get("rage_index"))
            and pd.notna(row.get("lifecycle_stage"))
        )

        if pre_encoded:
            sentiments.append(float(row["sentiment"]))
            rage_scores.append(float(row["rage_index"]))
            lifecycle_stages.append(str(row["lifecycle_stage"]))
            _, hit = _classify_lifecycle_stage(text)
            if hit:
                lifecycle_hits += 1
            is_negative_flags.append(
                bool(
                    row.get(
                        "is_negative",
                        _row_is_negative(text, float(row["sentiment"]), rating_val),
                    )
                )
            )
        else:
            sent = _compute_sentiment(text, rating_val)
            rage = _compute_rage_index(text, sent, rating_val)
            stage, hit = _classify_lifecycle_stage(text)
            if hit:
                lifecycle_hits += 1
            sentiments.append(sent)
            rage_scores.append(rage)
            lifecycle_stages.append(stage)
            is_negative_flags.append(_row_is_negative(text, sent, rating_val))

        if (pos + 1) % max(1, n // 10) == 0:
            report((pos + 1) / n * 0.6, f"Encoding review {pos + 1:,} / {n:,}...")

    result["sentiment"] = sentiments
    result["rage_index"] = rage_scores
    result["lifecycle_stage"] = lifecycle_stages
    result["is_negative"] = is_negative_flags
    result = _ensure_friction_coverage(result)

    # Weak product-lifecycle signal → disable funnel framing for this corpus
    hit_rate = lifecycle_hits / max(n, 1)
    if hit_rate < 0.15:
        result["lifecycle_stage"] = "General feedback"
        result.attrs["lifecycle_mode"] = "disabled"
    else:
        result.attrs["lifecycle_mode"] = "active"

    # Language detection for corpus profile
    from services.multilingual_nlp import detect_language

    result["language"] = [detect_language(str(row["text"])) for _, row in result.iterrows()]

    report(0.65, "Extracting pain point topics from negative reviews...")

    topic_method_used = "none"

    # Step 2: Topic extraction (skip if topics already supplied)
    if "topic" not in result.columns or result["topic"].isna().all() or (result["topic"] == "Unclassified feedback").all():
        topics: list[str] = [""] * n
        negative_mask = result["is_negative"].values
        negative_texts = result.loc[negative_mask, "text"].tolist()
        negative_indices = result.index[negative_mask].tolist()

        if negative_texts:

            def topic_progress(msg: str) -> None:
                report(0.72, msg)

            negative_topics, topic_method_used = extract_topic_labels(
                negative_texts,
                progress_callback=topic_progress,
            )
            for idx, topic in zip(negative_indices, negative_topics):
                topics[idx] = topic

        positive_mask = ~negative_mask
        positive_texts = result.loc[positive_mask, "text"].tolist()
        positive_indices = result.index[positive_mask].tolist()

        if positive_texts:
            positive_topics = _assign_positive_topics(positive_texts)
            for idx, topic in zip(positive_indices, positive_topics):
                topics[idx] = topic

        result["topic"] = topics

    # Merge near-duplicate pain-point labels
    report(0.9, "Merging similar pain-point themes...")
    result, merge_stats = apply_topic_merge(result)
    if merge_stats.get("merges"):
        report(
            0.93,
            f"Merged {merge_stats['merges']} near-duplicate themes "
            f"({merge_stats.get('topics_before')} → {merge_stats.get('topics_after')}).",
        )

    method_label = {"bertopic": "BERTopic (semantic)", "tfidf": "TF-IDF (statistical)", "none": "pre-encoded"}.get(
        topic_method_used, topic_method_used
    )
    report(0.97, "Scoring analysis credibility...")
    credibility = compute_credibility_report(result)
    result.attrs["topic_method"] = topic_method_used
    result.attrs["credibility"] = credibility
    result.attrs["cache_hit"] = False
    result.attrs["cache_key"] = fingerprint

    if use_cache and fingerprint:
        try:
            save_cached_encoding(fingerprint, result)
        except Exception:
            pass

    report(
        1.0,
        f"Encoding complete — {n:,} reviews processed. "
        f"Topic engine: {method_label}. Credibility: {credibility['grade']} ({credibility['confidence']}/100).",
    )
    return result


def semantic_coding_with_openai(reviews: list[str], model: str = "gpt-4o") -> list[dict[str, Any]]:
    """
    Placeholder: LLM-based semantic coding via OpenAI API.

    Set environment variable OPENAI_API_KEY to enable in a future release.
    """
    raise NotImplementedError(
        "OpenAI semantic coding not yet wired. "
        "The local encoder (encode_reviews) is active and requires no API key."
    )


def semantic_coding_with_anthropic(reviews: list[str], model: str = "claude-sonnet-4-20250514") -> list[dict[str, Any]]:
    """Placeholder: LLM-based semantic coding via Anthropic API."""
    raise NotImplementedError(
        "Anthropic semantic coding not yet wired. "
        "The local encoder (encode_reviews) is active and requires no API key."
    )
