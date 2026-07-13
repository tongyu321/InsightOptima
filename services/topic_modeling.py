"""
Topic modeling for InsightOptima — BERTopic primary, TF-IDF fallback.

Produces human-readable pain point labels for UXR dashboards and roadmaps.
Uses sentence embeddings + KeyBERT-inspired representations so topic names
read as themes (e.g. "Broken Packaging") rather than raw keyword chains.
"""

from __future__ import annotations

import re
from typing import Callable

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer

# Minimum negative reviews required before attempting BERTopic
BERTOPIC_MIN_REVIEWS = 8

# Extra tokens that dominate c-TF-IDF but carry little UX meaning
_TOPIC_STOPWORDS = frozenset(ENGLISH_STOP_WORDS).union(
    {
        "the", "and", "to", "it", "in", "this", "that", "is", "was", "for", "on",
        "with", "as", "at", "by", "an", "be", "are", "were", "they", "them", "their",
        "you", "your", "my", "me", "we", "our", "he", "she", "his", "her", "its",
        "not", "but", "or", "if", "so", "just", "very", "really", "also", "one",
        "get", "got", "like", "would", "could", "can", "have", "has", "had", "do",
        "did", "does", "been", "being", "br", "msg", "href", "http", "https", "www",
        "amazon", "com", "product", "item", "buy", "bought", "purchase", "ordered",
        "order", "received", "said", "say", "know", "think", "even", "much", "many",
        "still", "way", "thing", "things", "something", "anything", "everything",
        "time", "times", "make", "made", "use", "used", "using", "go", "going",
        "come", "came", "want", "wanted", "need", "needed", "try", "tried",
        "good", "bad", "great", "ok", "okay", "well", "back", "first", "last",
        "new", "old", "little", "lot", "lots", "bit", "sure", "maybe", "perhaps",
        "review", "reviews", "reviewer", "reviewers", "customer", "customers",
        "people", "someone", "anyone", "please", "thank", "thanks",
        "purchased", "buying", "bought", "seller", "store", "online",
        "definitely", "absolutely", "totally", "literally", "basically",
        "probably", "actually", "however", "though", "although", "since",
        "because", "enough", "almost", "already", "another", "every",
        "difficult", "hard", "easy", "seems", "seemed", "looks", "looked",
        "enjoyed", "enjoy", "love", "loved", "hate", "hated", "pretty",
        "contains", "contain", "containing", "doesn", "isn", "aren", "won",
        "wasn", "wouldn", "couldn", "shouldn", "didn", "hasn", "haven",
        "mixture", "mix", "stuff", "things",
        # Recipe / food-site noise that steals topic labels
        "recipe", "recipes", "homemade", "cooking", "baked", "bake", "oven",
        "minutes", "cups", "cup", "tablespoon", "teaspoon", "ingredients",
    }
)

# Never use these as the *name* of a negative theme (praise / ironic positives)
_POSITIVE_LABEL_BLOCKLIST = frozenset(
    {
        "yummy", "delicious", "tasty", "amazing", "awesome", "wonderful",
        "excellent", "fantastic", "perfect", "favorite", "favourite", "best",
        "love", "loved", "loves", "sounds", "scrumptious", "mouthwatering",
        "delightful", "heavenly", "yumm", "nom", "addicting", "addictive",
        "superb", "outstanding", "fabulous", "gorgeous", "beautiful",
        "away",  # "right away" / "away delicious" fragments
    }
)

# Issue / quality words — preferred when composing noun + issue labels
_ISSUE_WORDS = frozenset(
    {
        "broken", "damaged", "stale", "bitter", "rancid", "moldy", "spoiled",
        "expired", "awful", "terrible", "horrible", "disgusting", "disappointing",
        "disappointed", "defective", "leaking", "melted", "crushed", "missing",
        "wrong", "late", "slow", "expensive", "overpriced", "bland", "salty",
        "sour", "smelly", "smell", "odor", "taste", "flavor", "quality", "waste",
        "wasted", "refund", "return", "cancel", "spam", "crash", "bug", "error",
        "fail", "failed", "failure", "delay", "delayed", "empty", "opened", "fake",
        "mushy", "gooey", "gummy", "dry", "soggy", "inedible", "gross", "nasty",
        "ruined", "unhealthy", "artificial",
    }
)


def _get_embedding_model(multilingual: bool = False):
    """Lazy-load SentenceTransformer — English or multilingual variant."""
    model_name = (
        "paraphrase-multilingual-MiniLM-L12-v2"
        if multilingual
        else "all-MiniLM-L6-v2"
    )
    if not hasattr(_get_embedding_model, "_cache"):
        _get_embedding_model._cache = {}  # type: ignore[attr-defined]
    cache: dict = _get_embedding_model._cache  # type: ignore[attr-defined]
    if model_name not in cache:
        from sentence_transformers import SentenceTransformer

        cache[model_name] = SentenceTransformer(model_name)
    return cache[model_name]


def _clean_text_for_tfidf(text: str) -> str:
    """Minimal text normalization before TF-IDF vectorization."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_token(token: str) -> str:
    """Normalize a keyword/phrase for stopword checks."""
    return re.sub(r"\s+", " ", token.replace("_", " ").strip().lower())


def _stem_light(word: str) -> str:
    """Very light stemmer to catch taste/tasting, chip/chips near-duplicates."""
    w = _normalize_token(word)
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _is_usable_term(token: str) -> bool:
    """Return True if the term is meaningful enough for a pain-point label."""
    cleaned = _normalize_token(token)
    if len(cleaned) < 3:
        return False
    parts = cleaned.split()
    if all(p in _TOPIC_STOPWORDS for p in parts):
        return False
    # Drop truncated / garbage fragments ("food ve", "br br")
    if any(len(p) <= 2 for p in parts):
        return False
    if re.fullmatch(r"[\d\s]+", cleaned):
        return False
    if cleaned in {"br br", "href", "nbsp"}:
        return False
    # Entire phrase is praise / ironic-positive — unusable as a negative theme name
    if all(p in _POSITIVE_LABEL_BLOCKLIST or p in _TOPIC_STOPWORDS for p in parts):
        return False
    return True


def _term_has_positive_token(token: str) -> bool:
    return any(p in _POSITIVE_LABEL_BLOCKLIST for p in _normalize_token(token).split())


def _quote_cue_from_text(text: str, max_words: int = 7) -> str:
    """Short verbatim cue when keyword labels collapse."""
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return "General negative feedback"
    words = cleaned.split()[:max_words]
    cue = " ".join(words)
    if len(cleaned.split()) > max_words:
        cue += "…"
    return f"Quote cue: {cue}"


def topic_label_needs_review(label: str) -> bool:
    """True when an auto theme name is still a draft (Issue:/Quote cue:/praise residue)."""
    text = str(label or "").strip()
    if not text:
        return True
    lower = text.lower()
    if lower.startswith("issue:") or lower.startswith("quote cue:"):
        return True
    if any(p in _POSITIVE_LABEL_BLOCKLIST for p in re.findall(r"[a-z]+", lower)):
        return True
    return False


def _title_phrase(phrase: str) -> str:
    """Title-case a phrase while keeping short connectors lowercase."""
    small = {"and", "or", "of", "the", "a", "an", "to", "for", "in", "on"}
    words = _normalize_token(phrase).split()
    titled = []
    for i, w in enumerate(words):
        if i > 0 and w in small:
            titled.append(w)
        else:
            titled.append(w.capitalize())
    return " ".join(titled)


def _phrase_overlap(a: str, b: str) -> bool:
    """True when two phrases share a meaningful stem (redundant pair)."""
    stems_a = {_stem_light(p) for p in _normalize_token(a).split()}
    stems_b = {_stem_light(p) for p in _normalize_token(b).split()}
    return bool(stems_a & stems_b)


# Labels that are too vague alone — keep digging for a better term
_GENERIC_TERMS = frozenset(
    {
        "food", "flavor", "flavour", "taste", "product", "item", "stuff",
        "recipe", "quality", "review", "order", "amazon",
    }
)


def _compose_label(terms: list[str], max_terms: int = 1) -> str:
    """
    Compose a UXR-friendly topic label from ranked terms.

    Prefers multi-word phrases; otherwise pairs issue adjectives with product nouns
    (e.g. Stale + Chips → "Stale Chips"). Noun-only clusters become "Issue: …".
    """
    if not terms:
        return ""

    # Drop praise tokens from candidates (negative-cluster naming)
    filtered = [t for t in terms if not _term_has_positive_token(t) and _is_usable_term(t)]
    if not filtered:
        # Keep non-positive usable terms even if specificity is low
        filtered = [t for t in terms if _is_usable_term(t) and not _term_has_positive_token(t)]
    if not filtered:
        return ""

    # Prefer specific terms over vague ones when ranking candidates
    def specificity(term: str) -> int:
        parts = _normalize_token(term).split()
        return sum(1 for p in parts if p not in _GENERIC_TERMS)

    ranked = sorted(filtered, key=specificity, reverse=True)
    phrases = [t for t in ranked if " " in _normalize_token(t)]
    unigrams = [t for t in ranked if " " not in _normalize_token(t)]

    if phrases:
        # Prefer the most specific phrase; optionally rewrite "noun issue" order
        best = max(phrases, key=specificity)
        parts = _normalize_token(best).split()
        if len(parts) == 2:
            a, b = parts
            if a not in _ISSUE_WORDS and b in _ISSUE_WORDS:
                attribute_issues = {"taste", "flavor", "smell", "odor", "quality", "price"}
                if b not in attribute_issues:
                    return f"{_title_phrase(b)} {_title_phrase(a)}"
            if a in _ISSUE_WORDS and b not in _ISSUE_WORDS:
                attribute_issues = {"taste", "flavor", "smell", "odor", "quality", "price"}
                if a in attribute_issues:
                    return f"{_title_phrase(b)} {_title_phrase(a)}"
                return f"{_title_phrase(a)} {_title_phrase(b)}"
        label = _title_phrase(best)
        if specificity(best) == 0 and unigrams:
            specific = next((u for u in unigrams if specificity(u) > 0), None)
            if specific:
                return f"{_title_phrase(specific)} {_title_phrase(best)}"
        # Phrase with no issue word → mark as issue-with-X
        if not any(p in _ISSUE_WORDS for p in parts):
            return f"Issue: {label}"
        return label

    # Legacy multi-phrase path kept unused when max_terms==1; retained for callers
    if phrases and max_terms > 1:
        chosen = [phrases[0]]
        for p in phrases[1:]:
            if _phrase_overlap(chosen[0], p):
                continue
            if specificity(p) == 0:
                continue
            chosen.append(p)
            if len(chosen) >= max_terms:
                break
        return " / ".join(_title_phrase(p) for p in chosen)

    issue = next((u for u in unigrams if _normalize_token(u) in _ISSUE_WORDS), None)
    nouns = [u for u in unigrams if _normalize_token(u) not in _ISSUE_WORDS]

    if issue and nouns:
        issue_norm = _normalize_token(issue)
        attribute_issues = {"taste", "flavor", "smell", "odor", "quality", "price"}
        nouns_sorted = sorted(nouns, key=specificity, reverse=True)
        noun = nouns_sorted[0]
        if issue_norm in attribute_issues:
            return f"{_title_phrase(noun)} {_title_phrase(issue)}"
        return f"{_title_phrase(issue)} {_title_phrase(noun)}"

    if issue and not nouns:
        return _title_phrase(issue)

    # Noun-only / ingredient-only → Issue: prefix (not a praise-looking bare noun)
    specific_uni = [u for u in unigrams if specificity(u) > 0] or unigrams
    chosen = specific_uni[:max_terms]
    if not chosen:
        return ""
    if len(chosen) == 1:
        return f"Issue: {_title_phrase(chosen[0])}"
    if len(chosen) >= 2 and _stem_light(chosen[0]) == _stem_light(chosen[1]):
        return f"Issue: {_title_phrase(chosen[0])}"
    return f"Issue: {' / '.join(_title_phrase(u) for u in chosen)}"


def _format_topic_words(
    words: list[tuple[str, float]],
    max_terms: int = 1,
    *,
    cue_text: str | None = None,
) -> str:
    """
    Turn BERTopic/KeyBERT keyword tuples into a readable pain-point label.

    Example: [('broken packaging', 0.4), ('damaged box', 0.3)] → "Broken Packaging"
    Falls back to Quote cue: … when keywords are unusable for negative naming.
    """
    if not words:
        return _quote_cue_from_text(cue_text) if cue_text else "General negative feedback"

    selected: list[str] = []
    seen: set[str] = set()
    seen_stems: set[str] = set()
    for word, _ in words:
        if not _is_usable_term(word):
            continue
        if _term_has_positive_token(word):
            continue
        key = _normalize_token(word)
        stem = _stem_light(key.split()[0]) if key else ""
        if any(key in s or s in key for s in seen):
            continue
        if stem and stem in seen_stems:
            continue
        selected.append(word)
        seen.add(key)
        if stem:
            seen_stems.add(stem)
        if len(selected) >= 6:
            break

    label = _compose_label(selected, max_terms=max_terms)
    if not label:
        return _quote_cue_from_text(cue_text) if cue_text else "General negative feedback"
    return label


def _dedupe_labels(label_map: dict[int, str]) -> dict[int, str]:
    """
    Ensure topic labels stay unique across clusters.

    If two topics collapse to the same string, append a short secondary cue.
    """
    used: dict[str, int] = {}
    result: dict[int, str] = {}
    for topic_id, label in sorted(label_map.items()):
        base = label
        if base not in used:
            used[base] = 1
            result[topic_id] = base
            continue
        used[base] += 1
        result[topic_id] = f"{base} ({used[base]})"
    return result


def _build_vectorizer(multilingual: bool = False) -> CountVectorizer:
    """CountVectorizer tuned for readable topic keywords (uni + bi-grams)."""
    # min_df=1 / max_df=1.0: BERTopic re-fits on one-doc-per-topic corpora,
    # where stricter df thresholds raise ValueError.
    stop_list = sorted(_TOPIC_STOPWORDS)
    return CountVectorizer(
        stop_words=stop_list if not multilingual else None,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        max_features=8_000,
    )


def _build_representation_model():
    """
    KeyBERT-inspired keywords for clearer topic names.

    MMR is skipped — it can crash on sparse vocabularies after stopword filtering.
    """
    try:
        from bertopic.representation import KeyBERTInspired

        return KeyBERTInspired(top_n_words=10)
    except Exception:
        return None


def _extract_topics_tfidf(texts: list[str], n_topics: int = 10) -> list[str]:
    """
    Fallback topic discovery via TF-IDF + MiniBatchKMeans clustering.

    Used when review count is too small for BERTopic or BERTopic is unavailable.
    """
    if len(texts) == 0:
        return []

    cleaned = [_clean_text_for_tfidf(t) for t in texts]
    n_clusters = min(n_topics, max(2, len(texts) // 15))

    if len(texts) < 5:
        return ["General negative feedback"] * len(texts)

    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words=list(_TOPIC_STOPWORDS),
        min_df=2,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        return ["General negative feedback"] * len(texts)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3, batch_size=256)
    labels = kmeans.fit_predict(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    cluster_centers = kmeans.cluster_centers_

    cluster_topic_names: dict[int, str] = {}
    for cluster_id in range(n_clusters):
        center = cluster_centers[cluster_id]
        top_indices = center.argsort()[-8:][::-1]
        top_terms = [
            feature_names[i]
            for i in top_indices
            if center[i] > 0 and _is_usable_term(feature_names[i]) and not _term_has_positive_token(feature_names[i])
        ]
        # cue: first text in cluster
        cue_idx = next((i for i, lab in enumerate(labels) if int(lab) == cluster_id), None)
        cue = texts[cue_idx] if cue_idx is not None else None
        composed = _compose_label(top_terms, max_terms=1)
        if not composed:
            cluster_topic_names[cluster_id] = _quote_cue_from_text(cue or "")
        else:
            cluster_topic_names[cluster_id] = composed

    return [cluster_topic_names.get(label, "General negative feedback") for label in labels]


def _extract_topics_bertopic(
    texts: list[str],
    n_topics: int = 12,
    multilingual: bool = False,
) -> list[str]:
    """
    Semantic topic discovery via BERTopic + KeyBERT-style representations.

    Uses KMeans on embeddings for a stable, UXR-friendly topic count
    (HDBSCAN often collapses review corpora into 1–2 giant clusters).
    """
    from bertopic import BERTopic
    from sklearn.cluster import KMeans

    embedding_model = _get_embedding_model(multilingual=multilingual)
    vectorizer_model = _build_vectorizer(multilingual=multilingual)
    representation_model = _build_representation_model()

    # Target 6–14 themes depending on corpus size
    n_clusters = min(max(n_topics, 6), max(6, len(texts) // 20))
    n_clusters = min(n_clusters, max(3, len(texts) // 8))
    cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    topic_kwargs: dict = {
        "embedding_model": embedding_model,
        "vectorizer_model": vectorizer_model,
        "hdbscan_model": cluster_model,
        # KMeans assigns every doc — disable HDBSCAN-style outlier topic
        "nr_topics": None,
        "verbose": False,
        "calculate_probabilities": False,
        "low_memory": True,
        "top_n_words": 12,
    }
    if representation_model is not None:
        topic_kwargs["representation_model"] = representation_model

    topic_model = BERTopic(**topic_kwargs)
    try:
        assigned_topics, _ = topic_model.fit_transform(texts)
    except Exception:
        topic_kwargs.pop("representation_model", None)
        topic_model = BERTopic(**topic_kwargs)
        assigned_topics, _ = topic_model.fit_transform(texts)

    label_map: dict[int, str] = {}
    for topic_id in set(assigned_topics):
        if topic_id == -1:
            label_map[-1] = "Mixed / uncategorized complaints"
            continue
        keywords = topic_model.get_topic(topic_id)
        members = [texts[i] for i, t in enumerate(assigned_topics) if t == topic_id]
        # Prefer a longer member as quote cue (often richer complaint)
        cue = max(members, key=len) if members else None
        label_map[topic_id] = _format_topic_words(keywords or [], cue_text=cue)

    label_map = _dedupe_labels(label_map)
    return [label_map.get(t, "General negative feedback") for t in assigned_topics]


def extract_topic_labels(
    texts: list[str],
    n_topics: int = 12,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[str], str]:
    """
    Extract pain-point topic labels using the best available method.

    Priority: BERTopic (semantic + KeyBERT labels) → TF-IDF (statistical fallback).

    Parameters
    ----------
    texts : list[str]
        Review texts to cluster.
    n_topics : int
        Target topic count for BERTopic reduction.
    progress_callback : callable | None
        Optional callback(message: str) for UI status updates.

    Returns
    -------
    tuple[list[str], str]
        (topic_labels, method_used) where method_used is 'bertopic' or 'tfidf'.
    """
    if len(texts) == 0:
        return [], "none"

    def report(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    if len(texts) >= BERTOPIC_MIN_REVIEWS:
        try:
            from services.multilingual_nlp import is_primarily_non_english

            multilingual = is_primarily_non_english(texts)
            engine_note = "multilingual" if multilingual else "English"
            report(
                f"Running BERTopic ({engine_note}) on {len(texts):,} reviews "
                "(first run may take 30–90s while the model loads)..."
            )
            labels = _extract_topics_bertopic(texts, n_topics=n_topics, multilingual=multilingual)
            report(f"BERTopic found {len(set(labels))} distinct pain point themes.")
            return labels, "bertopic"
        except Exception as exc:
            report(f"BERTopic unavailable ({exc}). Falling back to TF-IDF clustering...")

    report(f"Running TF-IDF topic clustering on {len(texts):,} reviews...")
    labels = _extract_topics_tfidf(texts, n_topics=n_topics)
    return labels, "tfidf"
