"""
Research Brief / Executive Summary generator.

Stakeholder-ready order (UXR report style):
1. Research background
2. Findings
3. Sources for findings (verbatim evidence)
4. Research method
5. Recommended next steps
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from services.credibility import compute_credibility_report
from services.evidence_chain import fetch_topic_evidence


DEFAULT_RESEARCH_QUESTIONS = [
    "Where do negative experiences concentrate in this feedback corpus?",
    "Which themes are highest urgency (intensity × volume) for follow-up research?",
    "What verbatim evidence supports each claim — and where is confidence weak?",
]

PUBLIC_HEALTH_QUESTIONS = [
    "Where do negative medication / treatment experiences concentrate?",
    "Which side-effect or adherence themes are highest urgency for follow-up research?",
    "What verbatim patient evidence supports each claim — and what must we not over-claim?",
]


def _top_negative_themes(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    neg = df[df["is_negative"]] if "is_negative" in df.columns else df
    if neg.empty or "topic" not in neg.columns:
        return pd.DataFrame(columns=["topic", "count", "avg_rage", "avg_sentiment"])
    count_col = "review_id" if "review_id" in neg.columns else "text"
    g = (
        neg.groupby("topic", dropna=False)
        .agg(
            count=(count_col, "count"),
            avg_rage=("rage_index", "mean"),
            avg_sentiment=("sentiment", "mean"),
        )
        .reset_index()
        .sort_values(["count", "avg_rage"], ascending=[False, False])
        .head(n)
    )
    g["avg_rage"] = g["avg_rage"].round(1)
    g["avg_sentiment"] = g["avg_sentiment"].round(2)
    return g


def _infer_questions(product_context: str) -> list[str]:
    ctx = product_context.lower()
    if "drug" in ctx or "uci" in ctx or "health" in ctx or "patient" in ctx:
        return list(PUBLIC_HEALTH_QUESTIONS)
    return list(DEFAULT_RESEARCH_QUESTIONS)


def _background_blurb(product_context: str, total: int) -> str:
    ctx = product_context.lower()
    if "drug" in ctx or "uci" in ctx or "health" in ctx or "patient" in ctx:
        return (
            "This brief synthesizes a public patient-review corpus "
            "(UCI Drug Review Dataset / Drugs.com, CC BY 4.0) to surface "
            f"high-friction medication experience themes from n = {total:,} items. "
            "The goal is a discussion-ready priority draft with auditable quotes — "
            "not a clinical or causal outcome claim."
        )
    return (
        f"This brief synthesizes feedback from “{product_context}” (n = {total:,}) "
        "to surface concentrated friction themes, attach verbatim evidence, "
        "and draft a P0/P1/P2 discussion order for research and product partners."
    )


def build_research_brief(
    df: pd.DataFrame,
    roadmap_df: pd.DataFrame,
    *,
    study_title: str = "Feedback synthesis — priority draft with evidence",
    product_context: str = "Product feedback corpus",
    research_questions: list[str] | None = None,
    max_findings: int = 5,
) -> dict[str, Any]:
    """Build a structured research brief dict (UI + markdown export)."""
    cred = df.attrs.get("credibility")
    if not isinstance(cred, dict) or not cred:
        cred = compute_credibility_report(df)

    total = int(len(df))
    neg_n = int(df["is_negative"].sum()) if "is_negative" in df.columns else 0
    neg_pct = round(neg_n / total * 100, 1) if total else 0.0
    themes = _top_negative_themes(df, n=max_findings)
    questions = research_questions or _infer_questions(product_context)

    findings: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for _, row in themes.iterrows():
        topic = str(row["topic"])
        evidence = fetch_topic_evidence(df, topic, max_rows=3)
        quotes: list[dict[str, Any]] = []
        if not evidence.empty:
            for _, erow in evidence.iterrows():
                quotes.append(
                    {
                        "text": str(erow.get("text", ""))[:320],
                        "rage": erow.get("rage_index", ""),
                        "stage": str(erow.get("lifecycle_stage", "")),
                    }
                )

        priority = "—"
        action = ""
        if not roadmap_df.empty and "Core Pain Point" in roadmap_df.columns:
            match = roadmap_df[roadmap_df["Core Pain Point"].astype(str) == topic]
            if not match.empty:
                priority = str(match.iloc[0].get("Priority", "—"))
                action = str(match.iloc[0].get("Recommended Action", ""))

        finding = {
            "theme": topic,
            "n": int(row["count"]),
            "avg_rage": float(row["avg_rage"]),
            "avg_sentiment": float(row["avg_sentiment"]),
            "priority": priority,
            "action": action,
            "claim": (
                f"“{topic}” appears in {int(row['count'])} negative items "
                f"(avg rage {float(row['avg_rage'])}; priority draft {priority})."
            ),
        }
        findings.append(finding)
        sources.append({"theme": topic, "n": int(row["count"]), "quotes": quotes})

    p0 = int((roadmap_df["Priority"] == "P0").sum()) if not roadmap_df.empty and "Priority" in roadmap_df.columns else 0
    p1 = int((roadmap_df["Priority"] == "P1").sum()) if not roadmap_df.empty and "Priority" in roadmap_df.columns else 0

    limitations = [
        "Findings describe patterns in stated feedback; they do not prove causal clinical, retention, or revenue impact.",
        "Theme labels and priority scores are analytical drafts for discussion — not final research conclusions.",
        f"Analysis confidence grade: {cred.get('grade', 'N/A')} ({cred.get('confidence', '—')}/100).",
    ]
    for w in (cred.get("warnings") or [])[:3]:
        limitations.append(str(w))

    next_research = [
        "Interview 5–8 people who match the top 2 finding themes to validate mechanisms behind the quotes.",
        "Triangulate with a second public or internal source (tickets, survey open-ends, or support tags) for the same themes.",
        "Turn the leading P0 theme into a focused research protocol (tasks / diary / usability) before recommending a build.",
    ]

    return {
        "study_title": study_title,
        "product_context": product_context,
        "background": {
            "summary": _background_blurb(product_context, total),
            "objectives": [
                "Discover concentrated friction themes in the corpus",
                "Attach verbatim evidence to each priority claim",
                "Draft next-research recommendations without over-claiming outcomes",
            ],
            "research_questions": questions,
        },
        "research_questions": questions,  # compat
        "findings": findings,
        "sources": sources,
        "method": {
            "n_total": total,
            "n_negative": neg_n,
            "negative_pct": neg_pct,
            "n_themes": int(df["topic"].nunique()) if "topic" in df.columns else 0,
            "credibility_grade": cred.get("grade", "N/A"),
            "credibility_score": cred.get("confidence", 0),
            "approach": (
                "Mixed-signal synthesis: sentiment scoring + friction intensity (rage) + "
                "theme clustering, then evidence retrieval of verbatim quotes per theme."
            ),
            "priority_rule": (
                f"Priority draft currently marks {p0} P0 and {p1} P1 themes "
                f"across {int(len(roadmap_df))} roadmap rows (rage × volume; discussion order only)."
            ),
        },
        "priority_summary": {"p0": p0, "p1": p1, "themes_in_roadmap": int(len(roadmap_df))},
        "limitations": limitations,
        "next_research": next_research,
        "credibility": cred,
    }


def research_brief_to_markdown(brief: dict[str, Any]) -> str:
    """Render brief in UXR report order: background → findings → sources → method → next."""
    bg = brief.get("background") or {}
    m = brief["method"]
    lines: list[str] = [
        f"# Research Brief — {brief['study_title']}",
        "",
        "## 1. Research background",
        bg.get("summary") or f"Context: {brief.get('product_context', '')}",
        "",
        "### Objectives",
    ]
    for obj in bg.get("objectives") or []:
        lines.append(f"- {obj}")
    lines += ["", "### Research questions"]
    for i, q in enumerate(bg.get("research_questions") or brief.get("research_questions") or [], 1):
        lines.append(f"{i}. {q}")

    findings = [
        f for f in (brief.get("findings") or [])
        if str(f.get("status", "keep")).lower() != "drop"
    ]
    sources_by_theme = {
        str(s.get("theme")): s for s in (brief.get("sources") or [])
    }

    lines += ["", "## 2. Findings"]
    if not findings:
        lines.append("_No negative themes available in this run._")
    for i, f in enumerate(findings, 1):
        status = str(f.get("status", "keep"))
        status_note = f" [{status}]" if status == "needs_followup" else ""
        lines.append(f"### Finding {i}: {f['theme']}{status_note}")
        lines.append(f"- {f.get('claim', '')}")
        if f.get("action"):
            lines.append(f"- Next research note: {f['action']}")
        lines.append("")

    lines += ["", "## 3. Sources for findings"]
    kept_themes = {str(f["theme"]) for f in findings}
    if not kept_themes:
        lines.append("_No verbatim sources attached._")
    for theme in [str(f["theme"]) for f in findings]:
        src = sources_by_theme.get(theme) or {"theme": theme, "n": 0, "quotes": []}
        lines.append(f"### {src['theme']} (n = {src['n']})")
        if not src.get("quotes"):
            lines.append("- _No matching quotes retrieved._")
        for q in src.get("quotes") or []:
            meta = []
            if q.get("rage") != "" and q.get("rage") is not None:
                meta.append(f"rage {q['rage']}")
            if q.get("stage"):
                meta.append(str(q["stage"]))
            suffix = f" ({' · '.join(meta)})" if meta else ""
            lines.append(f"- “{q['text']}”{suffix}")
        lines.append("")

    lines += [
        "",
        "## 4. Research method",
        f"- Sample: **n = {m['n_total']:,}** ({m['n_negative']:,} negative · {m['negative_pct']}%)",
        f"- Themes detected: **{m['n_themes']}**",
        f"- Credibility: **grade {m['credibility_grade']}** ({m['credibility_score']}/100)",
        f"- Approach: {m['approach']}",
        f"- Priority draft rule: {m.get('priority_rule', '')}",
        "",
        "### Limitations",
    ]
    for lim in brief.get("limitations") or []:
        lines.append(f"- {lim}")

    lines += ["", "## 5. Recommended next steps"]
    for step in brief.get("next_research") or []:
        lines.append(f"- {step}")

    lines += [
        "",
        "---",
        "_Generated by InsightOptima — background → findings → sources → method → next. "
        "Not a retention, clinical, or revenue forecast._",
        "",
    ]
    return "\n".join(lines)


def apply_brief_overrides(
    brief: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Overlay analyst judgment onto a generated brief.

    Expected overrides shape::

        {
          "study_title": "...",
          "background_summary": "...",
          "limitations": ["...", "..."],
          "next_research": ["...", "..."],
          "by_theme": {
            "Theme A": {
              "claim": "...",
              "status": "keep" | "drop" | "needs_followup",
              "kept_quote_idxs": [0, 2],
            }
          }
        }
    """
    out = copy.deepcopy(brief)
    ov_root = overrides or {}

    if ov_root.get("study_title") and str(ov_root["study_title"]).strip():
        out["study_title"] = str(ov_root["study_title"]).strip()

    if ov_root.get("background_summary") and str(ov_root["background_summary"]).strip():
        bg = dict(out.get("background") or {})
        bg["summary"] = str(ov_root["background_summary"]).strip()
        out["background"] = bg

    if isinstance(ov_root.get("limitations"), list):
        cleaned = [str(x).strip() for x in ov_root["limitations"] if str(x).strip()]
        if cleaned:
            out["limitations"] = cleaned

    if isinstance(ov_root.get("next_research"), list):
        cleaned = [str(x).strip() for x in ov_root["next_research"] if str(x).strip()]
        if cleaned:
            out["next_research"] = cleaned

    by_theme = ov_root.get("by_theme") or {}
    if not by_theme:
        for f in out.get("findings") or []:
            f.setdefault("status", "keep")
        return out

    for finding in out.get("findings") or []:
        theme = str(finding.get("theme", ""))
        ov = by_theme.get(theme) or {}
        if "claim" in ov and str(ov["claim"]).strip():
            finding["claim"] = str(ov["claim"]).strip()
        status = str(ov.get("status", finding.get("status", "keep"))).lower()
        if status not in ("keep", "drop", "needs_followup"):
            status = "keep"
        finding["status"] = status

    for src in out.get("sources") or []:
        theme = str(src.get("theme", ""))
        ov = by_theme.get(theme) or {}
        quotes = list(src.get("quotes") or [])
        if "kept_quote_idxs" in ov and quotes:
            idxs = [int(i) for i in ov["kept_quote_idxs"] if 0 <= int(i) < len(quotes)]
            seen: set[int] = set()
            kept: list[dict[str, Any]] = []
            for i in idxs:
                if i not in seen:
                    kept.append(quotes[i])
                    seen.add(i)
            src["quotes"] = kept
        else:
            src["quotes"] = quotes

    return out
