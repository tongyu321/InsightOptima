"""
Merge near-duplicate pain-point topic labels.

Keeps dashboard/roadmap themes from fragmenting into near-identical clusters
(e.g. "Stale Coffee" vs "Stale Items" when token overlap is high).
"""

from __future__ import annotations

import re
from collections import defaultdict


_STOP = frozenset(
    {
        "the", "and", "or", "of", "a", "an", "to", "for", "in", "on", "with",
        "general", "negative", "feedback", "mixed", "uncategorized", "complaints",
    }
)


def _tokens(label: str) -> set[str]:
    parts = re.findall(r"[a-z0-9]+", label.lower())
    cleaned: set[str] = set()
    for p in parts:
        if len(p) < 3 or p in _STOP:
            continue
        # light stem
        for suffix in ("ing", "ed", "es", "s"):
            if len(p) > len(suffix) + 3 and p.endswith(suffix):
                p = p[: -len(suffix)]
                break
        cleaned.add(p)
    return cleaned


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _should_merge(a: str, b: str, threshold: float = 0.45) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if ta <= tb or tb <= ta:
        # one label's tokens fully contained in the other
        return True
    return _jaccard(ta, tb) >= threshold


def merge_topic_labels(
    topics: list[str],
    *,
    threshold: float = 0.45,
    protected: set[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """
    Merge similar topic strings into canonical labels.

    Parameters
    ----------
    topics : list[str]
        Per-review topic labels.
    threshold : float
        Jaccard similarity cutoff for merging.
    protected : set[str] | None
        Labels that should never be merged away (e.g. positive retention drivers).

    Returns
    -------
    tuple[list[str], dict[str, str]]
        (merged_labels_per_row, mapping old_label → canonical_label)
    """
    protected = protected or {
        "Positive experience",
        "Product quality praise",
        "Value for money",
        "Ease of use",
        "Fast delivery / shipping",
        "Reliability / consistency",
        "Customer support",
        "Mixed / uncategorized complaints",
        "General negative feedback",
    }

    # Count volumes — prefer high-volume labels as canonical
    counts: dict[str, int] = defaultdict(int)
    for t in topics:
        if t:
            counts[t] += 1

    unique = sorted(counts.keys(), key=lambda x: (-counts[x], x))
    parent: dict[str, str] = {t: t for t in unique}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(unique):
        if a in protected:
            continue
        for b in unique[:i]:
            if b in protected:
                continue
            if _should_merge(a, b, threshold=threshold):
                ra, rb = find(a), find(b)
                if ra == rb:
                    continue
                # Keep higher-volume (or shorter clearer) label as root
                if counts[ra] > counts[rb] or (counts[ra] == counts[rb] and len(ra) <= len(rb)):
                    parent[rb] = ra
                else:
                    parent[ra] = rb

    mapping = {t: find(t) for t in unique}
    merged = [mapping.get(t, t) for t in topics]
    return merged, mapping


def apply_topic_merge(df, topic_col: str = "topic"):
    """
    In-place-safe merge of the topic column on an encoded review dataframe.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (dataframe with merged topics, merge stats)
    """
    import pandas as pd

    if topic_col not in df.columns or df.empty:
        return df, {"merges": 0, "mapping": {}}

    result = df.copy()
    merged, mapping = merge_topic_labels(result[topic_col].astype(str).tolist())
    result[topic_col] = merged
    changed = sum(1 for k, v in mapping.items() if k != v)
    stats = {
        "merges": changed,
        "mapping": {k: v for k, v in mapping.items() if k != v},
        "topics_before": len(mapping),
        "topics_after": len(set(mapping.values())),
    }
    result.attrs["topic_merge"] = stats
    return result, stats
