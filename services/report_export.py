"""
Report export for InsightOptima.

Generates analyst-ready downloads:
- CSV  — implementation roadmap
- XLSX — summary + roadmap + evidence quotes (multi-sheet)
- HTML — printable executive report (browser → Print → PDF)
"""

from __future__ import annotations

import html
import io
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from services.evidence_chain import fetch_topic_evidence, summarize_topic_evidence
from services.roadmap_generator import build_implementation_roadmap, compute_rage_volume_matrix


def _escape(value: Any) -> str:
    return html.escape(str(value) if value is not None else "")


def compute_export_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Top-level metrics for the export cover section."""
    total = len(df)
    negative = int(df["is_negative"].sum()) if "is_negative" in df.columns else 0
    negative_pct = round(negative / total * 100, 1) if total else 0.0
    avg_rage = 0.0
    if "is_negative" in df.columns and df["is_negative"].any():
        avg_rage = round(float(df.loc[df["is_negative"], "rage_index"].mean()), 1)

    unique_topics = 0
    if "is_negative" in df.columns and "topic" in df.columns:
        unique_topics = int(df.loc[df["is_negative"], "topic"].nunique())

    lifecycle_mode = str(df.attrs.get("lifecycle_mode") or "active")
    if "lifecycle_stage" in df.columns and total:
        general_share = float((df["lifecycle_stage"].astype(str) == "General feedback").mean())
        if general_share >= 0.85:
            lifecycle_mode = "disabled"

    highest_risk = "N/A"
    if (
        lifecycle_mode != "disabled"
        and "lifecycle_stage" in df.columns
        and "is_negative" in df.columns
        and total
    ):
        rates: list[tuple[str, float]] = []
        for stage, stage_df in df.groupby("lifecycle_stage"):
            if len(stage_df) == 0 or str(stage) == "General feedback":
                continue
            rates.append((str(stage), float(stage_df["is_negative"].mean() * 100)))
        if rates:
            highest_risk = max(rates, key=lambda x: x[1])[0]

    languages: dict[str, int] = {}
    if "language" in df.columns:
        languages = df["language"].value_counts().head(8).to_dict()

    return {
        "total_reviews": total,
        "negative_count": negative,
        "negative_pct": negative_pct,
        "avg_rage": avg_rage,
        "unique_pain_topics": unique_topics,
        "highest_risk_stage": highest_risk,
        "lifecycle_mode": lifecycle_mode,
        "languages": languages,
        "topic_method": df.attrs.get("topic_method", "unknown"),
    }


def build_funnel_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Lifecycle negative-rate table for export sheets (empty when disabled)."""
    mode = str(df.attrs.get("lifecycle_mode") or "")
    if not mode and "lifecycle_stage" in df.columns and len(df):
        if float((df["lifecycle_stage"].astype(str) == "General feedback").mean()) >= 0.85:
            mode = "disabled"
    if mode == "disabled":
        return pd.DataFrame(
            columns=["Lifecycle Stage", "Total Reviews", "Negative Reviews", "Negative Rate (%)"]
        )

    stages = ["Onboarding", "Core Feature Activation", "Daily Retention"]
    rows: list[dict[str, Any]] = []
    for stage in stages:
        stage_df = df[df["lifecycle_stage"] == stage] if "lifecycle_stage" in df.columns else df.iloc[0:0]
        total = len(stage_df)
        neg = int(stage_df["is_negative"].sum()) if total and "is_negative" in stage_df.columns else 0
        rows.append(
            {
                "Lifecycle Stage": stage,
                "Total Reviews": total,
                "Negative Reviews": neg,
                "Negative Rate (%)": round(neg / total * 100, 1) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _roadmap_for_html(roadmap: pd.DataFrame) -> pd.DataFrame:
    """Public-facing roadmap columns for HTML (drop empty SaaS/behavior fields)."""
    if roadmap is None or roadmap.empty:
        return roadmap
    out = roadmap.copy()
    rename = {
        "AI Diagnosis": "Pattern note",
        "Recommended Action": "Next research note",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    drop_cols: list[str] = []
    for col in ("Behavior Match", "Churn Rate (%)", "Users Affected", "Supporting Signal"):
        if col not in out.columns:
            continue
        series = out[col].astype(str)
        if series.isin(["", "nan", "<NA>", "None", "review-only"]).all() or out[col].isna().all():
            drop_cols.append(col)
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def build_evidence_quotes(
    df: pd.DataFrame,
    roadmap_df: pd.DataFrame,
    *,
    quotes_per_topic: int = 3,
    max_topics: int = 10,
) -> pd.DataFrame:
    """
    Flatten top verbatim quotes per roadmap pain point for export.
    """
    rows: list[dict[str, Any]] = []
    for _, item in roadmap_df.head(max_topics).iterrows():
        topic = str(item["Core Pain Point"])
        priority = str(item["Priority"])
        evidence = fetch_topic_evidence(df, topic, max_rows=quotes_per_topic)
        summary = summarize_topic_evidence(df, topic)
        if evidence.empty:
            rows.append(
                {
                    "Priority": priority,
                    "Pain Point": topic,
                    "Quote Rank": 1,
                    "User Comment": "(no matching reviews)",
                    "Rage Index": "",
                    "Sentiment": "",
                    "Lifecycle Stage": "",
                    "Matching Reviews": summary["total_count"],
                }
            )
            continue
        for rank, (_, rev) in enumerate(evidence.iterrows(), start=1):
            rows.append(
                {
                    "Priority": priority,
                    "Pain Point": topic,
                    "Quote Rank": rank,
                    "User Comment": str(rev.get("text", "")),
                    "Rage Index": rev.get("rage_index", ""),
                    "Sentiment": rev.get("sentiment", ""),
                    "Lifecycle Stage": rev.get("lifecycle_stage", ""),
                    "Matching Reviews": summary["total_count"],
                }
            )
    return pd.DataFrame(rows)


def build_report_bundle(
    df: pd.DataFrame,
    *,
    source_label: str = "Unknown",
    roadmap_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Assemble all tables/metrics used by CSV / Excel / HTML exporters."""
    roadmap = roadmap_df if roadmap_df is not None else build_implementation_roadmap(df)
    matrix = compute_rage_volume_matrix(df)
    kpis = compute_export_kpis(df)
    funnel = build_funnel_summary(df)
    quotes = build_evidence_quotes(df, roadmap)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "source_label": source_label,
        "generated_at": generated_at,
        "kpis": kpis,
        "roadmap": roadmap,
        "matrix": matrix,
        "funnel": funnel,
        "quotes": quotes,
        "lifecycle_mode": kpis.get("lifecycle_mode", "active"),
    }


def export_roadmap_csv(bundle: dict[str, Any]) -> bytes:
    """CSV bytes for the implementation roadmap table."""
    buffer = io.StringIO()
    roadmap: pd.DataFrame = bundle["roadmap"]
    roadmap.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


def export_workbook_xlsx(bundle: dict[str, Any]) -> bytes:
    """
    Multi-sheet Excel workbook:
    Summary | Funnel | Roadmap | Rage-Volume | Evidence Quotes
    """
    buffer = io.BytesIO()
    kpis = bundle["kpis"]
    summary_rows = [
        {"Metric": "Generated at", "Value": bundle["generated_at"]},
        {"Metric": "Data source", "Value": bundle["source_label"]},
        {"Metric": "Total reviews", "Value": kpis["total_reviews"]},
        {"Metric": "Negative reviews", "Value": kpis["negative_count"]},
        {"Metric": "Negative sentiment rate (%)", "Value": kpis["negative_pct"]},
        {"Metric": "Avg. rage index (negative)", "Value": kpis["avg_rage"]},
        {"Metric": "Unique pain topics", "Value": kpis["unique_pain_topics"]},
        {"Metric": "Highest risk stage", "Value": kpis["highest_risk_stage"]},
        {"Metric": "Topic engine", "Value": kpis["topic_method"]},
    ]
    if kpis["languages"]:
        lang_str = ", ".join(f"{code}:{count}" for code, count in kpis["languages"].items())
        summary_rows.append({"Metric": "Language profile", "Value": lang_str})

    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if not bundle["funnel"].empty:
            bundle["funnel"].to_excel(writer, sheet_name="Drop-out Funnel", index=False)
        bundle["roadmap"].to_excel(writer, sheet_name="Roadmap", index=False)
        if not bundle["matrix"].empty:
            bundle["matrix"].to_excel(writer, sheet_name="Rage-Volume", index=False)
        if not bundle["quotes"].empty:
            bundle["quotes"].to_excel(writer, sheet_name="Evidence Quotes", index=False)

    return buffer.getvalue()


def export_html_report(bundle: dict[str, Any]) -> bytes:
    """
    Self-contained HTML executive report.

    Open in a browser and use Print → Save as PDF for a PDF deliverable.
    """
    kpis = bundle["kpis"]
    roadmap: pd.DataFrame = _roadmap_for_html(bundle["roadmap"])
    funnel: pd.DataFrame = bundle["funnel"]
    quotes: pd.DataFrame = bundle["quotes"]
    lifecycle_mode = str(bundle.get("lifecycle_mode") or kpis.get("lifecycle_mode") or "active")
    p0_count = int((roadmap["Priority"] == "P0").sum()) if not roadmap.empty and "Priority" in roadmap.columns else 0

    def table_html(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p class='muted'>No data.</p>"
        headers = "".join(f"<th>{_escape(c)}</th>" for c in df.columns)
        body_rows = []
        for _, row in df.iterrows():
            cells = "".join(f"<td>{_escape(row[c])}</td>" for c in df.columns)
            priority = str(row.get("Priority", "")) if "Priority" in df.columns else ""
            css = f' class="p-{priority.lower()}"' if priority in {"P0", "P1", "P2"} else ""
            body_rows.append(f"<tr{css}>{cells}</tr>")
        return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"

    # Evidence cards for top P0/P1 items
    quote_blocks: list[str] = []
    if not quotes.empty:
        for pain_point, group in quotes.groupby("Pain Point", sort=False):
            priority = str(group.iloc[0]["Priority"])
            matching = group.iloc[0]["Matching Reviews"]
            items = []
            for _, q in group.iterrows():
                comment = _escape(q["User Comment"])
                rage = _escape(q["Rage Index"])
                stage = _escape(q["Lifecycle Stage"])
                stage_bit = f" · {stage}" if stage and stage not in ("General feedback", "nan", "") else ""
                items.append(
                    f"<blockquote><p>“{comment}”</p>"
                    f"<footer>Rage {rage}/100{stage_bit}</footer></blockquote>"
                )
            quote_blocks.append(
                f"<section class='evidence'>"
                f"<h3><span class='badge p-{priority.lower()}'>{_escape(priority)}</span> "
                f"{_escape(pain_point)}</h3>"
                f"<p class='muted'>{_escape(matching)} matching reviews</p>"
                f"{''.join(items)}</section>"
            )

    lang_html = ""
    if kpis["languages"]:
        chips = " · ".join(f"{_escape(k)} ({v})" for k, v in kpis["languages"].items())
        lang_html = f"<p><strong>Languages:</strong> {chips}</p>"

    kpi_stage = ""
    if lifecycle_mode != "disabled" and kpis.get("highest_risk_stage") not in (None, "", "N/A"):
        kpi_stage = (
            f"<div class='kpi'><div class='label'>Highest Risk Stage</div>"
            f"<div class='value'>{_escape(kpis['highest_risk_stage'])}</div></div>"
        )

    if lifecycle_mode == "disabled":
        funnel_section = (
            "<h2>2. Lifecycle funnel</h2>"
            "<p class='muted'>No product lifecycle signals detected in this corpus — "
            "funnel framing is omitted. Themes and evidence below are the primary output.</p>"
        )
        roadmap_heading = "3. Priority draft"
        evidence_heading = "4. Evidence — verbatim quotes"
    else:
        funnel_section = f"<h2>2. Drop-out Funnel</h2>\n  {table_html(funnel)}"
        roadmap_heading = "3. Priority draft"
        evidence_heading = "4. Evidence — verbatim quotes"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>InsightOptima Report — {_escape(bundle['source_label'])}</title>
<style>
  :root {{
    --ink: #1a2332;
    --muted: #5a6a7a;
    --line: #d7dee7;
    --p0: #b71c1c;
    --p1: #e65100;
    --p2: #1b5e20;
    --accent: #1e3a5f;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: var(--ink);
    margin: 0;
    padding: 2rem 2.5rem 3rem;
    background: #f7f9fc;
    line-height: 1.5;
  }}
  header {{
    background: linear-gradient(135deg, #1e3a5f, #2d5a87);
    color: #fff;
    padding: 1.75rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.75rem;
  }}
  header h1 {{ margin: 0 0 0.35rem; font-size: 1.75rem; }}
  header .meta {{ opacity: 0.9; font-size: 0.95rem; }}
  h2 {{
    margin: 2rem 0 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--accent);
    font-size: 1.2rem;
  }}
  .kpis {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
    margin: 1rem 0 1.5rem;
  }}
  .kpi {{
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.9rem 1rem;
  }}
  .kpi .label {{ color: var(--muted); font-size: 0.8rem; }}
  .kpi .value {{ font-size: 1.35rem; font-weight: 700; margin-top: 0.2rem; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: #fff;
    border: 1px solid var(--line);
    font-size: 0.9rem;
  }}
  th, td {{
    border-bottom: 1px solid var(--line);
    padding: 0.55rem 0.7rem;
    text-align: left;
    vertical-align: top;
  }}
  th {{ background: #eef3f8; font-weight: 600; }}
  tr.p-p0 td:first-child {{ color: var(--p0); font-weight: 700; }}
  tr.p-p1 td:first-child {{ color: var(--p1); font-weight: 700; }}
  tr.p-p2 td:first-child {{ color: var(--p2); font-weight: 700; }}
  .badge {{
    display: inline-block;
    padding: 0.1rem 0.45rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #fff;
    margin-right: 0.4rem;
  }}
  .badge.p-p0 {{ background: var(--p0); }}
  .badge.p-p1 {{ background: var(--p1); }}
  .badge.p-p2 {{ background: var(--p2); }}
  .callout {{
    background: #e8f4fc;
    border-left: 4px solid var(--accent);
    padding: 0.85rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
  }}
  .evidence {{
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin: 0.85rem 0;
  }}
  blockquote {{
    margin: 0.65rem 0;
    padding: 0.55rem 0.85rem;
    background: #f4f7fb;
    border-left: 3px solid #4fc3f7;
  }}
  blockquote p {{ margin: 0; }}
  blockquote footer {{ margin-top: 0.35rem; color: var(--muted); font-size: 0.8rem; }}
  .muted {{ color: var(--muted); font-size: 0.9rem; }}
  footer.page-foot {{
    margin-top: 2.5rem;
    color: var(--muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--line);
    padding-top: 0.75rem;
  }}
  @media print {{
    body {{ background: #fff; padding: 0.5rem; }}
    header {{ break-inside: avoid; }}
    .evidence, table {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
  <header>
    <h1>InsightOptima</h1>
    <div class="meta">Feedback synthesis — priority draft with evidence</div>
    <div class="meta">Source: {_escape(bundle['source_label'])} · {_escape(bundle['generated_at'])}</div>
  </header>

  <h2>1. Executive Snapshot</h2>
  <div class="kpis">
    <div class="kpi"><div class="label">Total Reviews</div><div class="value">{kpis['total_reviews']:,}</div></div>
    <div class="kpi"><div class="label">Negative Rate</div><div class="value">{kpis['negative_pct']}%</div></div>
    <div class="kpi"><div class="label">Avg. Rage (negatives)</div><div class="value">{kpis['avg_rage']}</div></div>
    <div class="kpi"><div class="label">Pain Topics</div><div class="value">{kpis['unique_pain_topics']}</div></div>
    {kpi_stage}
  </div>
  {lang_html}
  <div class="callout">
    <strong>Priority draft:</strong> {p0_count} P0 themes among {len(roadmap)} pain points.
    Scores rank discussion urgency from rage × volume — not predicted retention, clinical, or revenue outcomes.
    Theme labels are analytical drafts — rename in the workspace before socializing.
  </div>

  {funnel_section}

  <h2>{roadmap_heading}</h2>
  {table_html(roadmap)}

  <h2>{evidence_heading}</h2>
  <p class="muted">Top quotes per theme, sorted by rage index (most frustrated first).</p>
  {''.join(quote_blocks) if quote_blocks else "<p class='muted'>No evidence quotes available.</p>"}

  <footer class="page-foot">
    Generated by InsightOptima · Topic engine: {_escape(kpis['topic_method'])} ·
    Tip: use your browser’s Print → Save as PDF for a PDF deliverable.
  </footer>
</body>
</html>
"""
    return doc.encode("utf-8")


def default_export_basename(source_label: str = "report") -> str:
    """Safe filename stem for downloads."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_label)[:40]
    safe = safe.strip("_") or "report"
    return f"InsightOptima_{safe}_{stamp}"
