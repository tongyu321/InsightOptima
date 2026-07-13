"""
InsightOptima — Research workspace for feedback synthesis
=========================================================
Streamlit app for UXR / research analysts:
discover friction themes → draft P0/P1/P2 → attach evidence → export a research brief.

Does not claim causal retention or revenue forecasts.
"""

from __future__ import annotations

import base64
import sys

# Streamlit keeps modules warm across reruns — drop service caches so UI
# strings / backend fixes always pick up the latest on-disk files.
for _mod in list(sys.modules):
    if _mod == "loader" or _mod.startswith("insightoptima.") or (
        _mod.startswith("services.") and _mod != "services"
    ):
        del sys.modules[_mod]

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from loader import (
    LOADER_VERSION,
    PreflightReport,
    apply_brief_overrides,
    apply_roadmap_overrides,
    ask_insight,
    auto_resolve_columns,
    both_strands_ready,
    build_implementation_roadmap,
    build_report_bundle,
    build_research_brief,
    build_strand_snapshot,
    clear_analysis_cache,
    compare_snapshots,
    default_export_basename,
    detect_corpus_languages,
    detect_strand_key,
    encode_reviews,
    export_html_report,
    export_roadmap_csv,
    export_workbook_xlsx,
    fetch_topic_evidence,
    format_evidence_for_display,
    generate_prd_markdown,
    is_rtl,
    list_projects,
    list_snapshots,
    list_theme_stats,
    load_behavior_dataframe,
    load_project,
    load_reviews_file,
    load_snapshot,
    merge_topics,
    normalize_behavior_df,
    normalize_raw_reviews,
    read_raw_file,
    rename_topic,
    render_language_selector,
    research_brief_to_markdown,
    run_preflight,
    save_analysis_snapshot,
    save_project,
    snapshots_to_dataframe,
    SUPPORTED_LANGUAGES,
    summarize_topic_evidence,
    t,
    topic_label_needs_review,
    translate_lifecycle_stage,
    translate_quadrant,
    translate_risk,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

PAGE_TITLE = "InsightOptima | Feedback synthesis with evidence"
PAGE_ICON = "📊"
MOCK_REVIEW_COUNT = 2000
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_XLSX = PROJECT_ROOT / "data" / "sample_reviews.xlsx"
SAMPLE_DATA_CSV = PROJECT_ROOT / "data" / "sample_reviews.csv"
SAMPLE_CASE_STUDY_CSV = PROJECT_ROOT / "data" / "case_study_drug_reviews.csv"
SAMPLE_CASE_STUDY_META = PROJECT_ROOT / "data" / "case_study_drug_reviews.SOURCE.json"
SAMPLE_CASE_STUDY_QUAL_CSV = PROJECT_ROOT / "data" / "case_study_pubpeer_qual.csv"
SAMPLE_CASE_STUDY_QUAL_META = PROJECT_ROOT / "data" / "case_study_pubpeer_qual.SOURCE.json"
SAMPLE_BEHAVIOR_CSV = PROJECT_ROOT / "data" / "sample_behavior.csv"


class LifecycleStage(str, Enum):
    """User lifecycle stages used for drop-out funnel analysis."""

    ONBOARDING = "Onboarding"
    CORE_FEATURE = "Core Feature Activation"
    DAILY_RETENTION = "Daily Retention"


@dataclass(frozen=True)
class StageMeta:
    """Metadata for each lifecycle stage in the drop-out funnel."""

    label: str
    risk_label: str
    risk_emoji: str
    risk_color: str


STAGE_METADATA: dict[LifecycleStage, StageMeta] = {
    LifecycleStage.ONBOARDING: StageMeta(
        label="Onboarding",
        risk_label="High",
        risk_emoji="🔴",
        risk_color="#E74C3C",
    ),
    LifecycleStage.CORE_FEATURE: StageMeta(
        label="Core Feature Activation",
        risk_label="Medium",
        risk_emoji="🟡",
        risk_color="#F39C12",
    ),
    LifecycleStage.DAILY_RETENTION: StageMeta(
        label="Daily Retention",
        risk_label="Low",
        risk_emoji="🟢",
        risk_color="#27AE60",
    ),
}


# ---------------------------------------------------------------------------
# Custom CSS — Nordic minimalist
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');

:root {
    --ink: #292524;
    --muted: #78716c;
    --line: #e7e5e4;
    --line-soft: #f0eeeb;
    --paper: #f7f6f3;
    --card: #ffffff;
    --accent: #0f766e;
    --accent-soft: #e6f7f4;
    --accent-mid: #99f6e4;
    --danger: #b91c1c;
    --warn: #a16207;
    --ok: #15803d;
    --radius: 10px;
    --shadow-soft: 0 1px 2px rgba(28, 25, 23, 0.04), 0 8px 24px rgba(28, 25, 23, 0.03);
}

html, body, [class*="css"] {
    font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
}

.stApp {
    background:
        radial-gradient(1000px 520px at 8% -8%, rgba(204, 251, 241, 0.45) 0%, transparent 55%),
        radial-gradient(800px 480px at 100% 0%, rgba(245, 245, 244, 0.9) 0%, transparent 50%),
        linear-gradient(180deg, #f9f8f6 0%, var(--paper) 40%, #f5f4f1 100%);
    color: var(--ink);
}

/* Main reading column — less wall-of-text feel */
div[data-testid="stAppViewContainer"] > .main .block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1080px;
}

.stMarkdown, .stCaption, p, li {
    line-height: 1.65;
}
.stCaption {
    color: var(--muted) !important;
    font-size: 0.9rem !important;
}

/* Sidebar — compact icon rail */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.92);
    border-right: 1px solid var(--line-soft);
    backdrop-filter: blur(8px);
    min-width: 88px !important;
    width: 88px !important;
    max-width: 88px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    width: 88px !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    width: 88px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.1rem;
    padding-left: 0.55rem !important;
    padding-right: 0.55rem !important;
}
.sidebar-brand {
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.02em;
    margin: 0 0 0.15rem 0;
    text-align: center;
    line-height: 1.2;
}
.sidebar-tag {
    display: none;
}
.sidebar-section {
    color: #a8a29e;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0.95rem 0 0.4rem;
    text-align: center;
}
section[data-testid="stSidebar"] .stButton > button {
    justify-content: center !important;
    text-align: center !important;
    border-radius: 10px !important;
    font-weight: 500;
    border: 1px solid var(--line) !important;
    background: #fff !important;
    color: #57534e !important;
    min-height: 2.55rem;
    width: 100%;
    padding-left: 0.25rem !important;
    padding-right: 0.25rem !important;
    margin-bottom: 0.35rem !important;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #d6d3d1 !important;
    background: #f5f5f4 !important;
    color: var(--ink) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #fff !important;
    border-color: var(--accent) !important;
    font-weight: 600;
}
section[data-testid="stSidebar"] .stButton > button:disabled {
    opacity: 0.38;
}
section[data-testid="stSidebar"] .stSelectbox label {
    font-size: 0.72rem !important;
}
/* Widen when a tool/settings panel is open */
section[data-testid="stSidebar"].sidebar-rail-wide,
section[data-testid="stSidebar"]:has(.sidebar-panel-open) {
    min-width: 240px !important;
    width: 240px !important;
    max-width: 240px !important;
}
section[data-testid="stSidebar"].sidebar-rail-wide > div:first-child,
section[data-testid="stSidebar"].sidebar-rail-wide [data-testid="stSidebarContent"],
section[data-testid="stSidebar"]:has(.sidebar-panel-open) > div:first-child,
section[data-testid="stSidebar"]:has(.sidebar-panel-open) [data-testid="stSidebarContent"] {
    width: 240px !important;
}
.sidebar-panel-open {
    margin-top: 0.35rem;
    padding-top: 0.5rem;
    border-top: 1px solid var(--line-soft);
}
.sidebar-panel-open .sidebar-panel-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--ink);
    margin: 0 0 0.45rem;
    text-align: left;
}

.dashboard-hero {
    background: linear-gradient(165deg, #ffffff 0%, #fbfaf8 100%);
    border: 1px solid var(--line-soft);
    border-radius: var(--radius);
    padding: 2.4rem 2.1rem 2.1rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-soft);
}

.dashboard-hero h1 {
    color: var(--ink);
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 2.2rem;
    font-weight: 600;
    margin: 0 0 0.55rem 0;
    letter-spacing: -0.03em;
    line-height: 1.15;
}

.dashboard-hero .tagline {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 1.15rem;
    line-height: 1.55;
    max-width: 40rem;
}

.dashboard-hero .logic-block {
    color: #57534e;
    font-size: 0.95rem;
    line-height: 1.7;
    border-left: 3px solid var(--accent-mid);
    padding-left: 1rem;
    margin-top: 0.35rem;
    max-width: 44rem;
}

div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--line-soft);
    border-radius: var(--radius);
    padding: 1rem 1.05rem;
    box-shadow: 0 1px 2px rgba(28, 25, 23, 0.03);
}

div[data-testid="stMetric"] label {
    color: var(--muted) !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-weight: 600 !important;
}

.section-header {
    color: var(--ink);
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 1.2rem;
    font-weight: 600;
    margin: 1.5rem 0 0.65rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--line-soft);
    letter-spacing: -0.02em;
}

.quadrant-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.75rem 0 1rem;
}

.quadrant-pill {
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    border: 1px solid var(--line);
    background: #fff;
    color: #57534e;
}

.pill-red { border-color: #fecaca; background: #fef2f2; color: var(--danger); }
.pill-purple { border-color: #e9d5ff; background: #faf5ff; color: #6b21a8; }
.pill-yellow { border-color: #fde68a; background: #fffbeb; color: var(--warn); }
.pill-green { border-color: #bbf7d0; background: #f0fdf4; color: var(--ok); }

.insight-answer {
    background: var(--card);
    border: 1px solid var(--line-soft);
    border-radius: var(--radius);
    padding: 1.15rem 1.3rem;
    line-height: 1.7;
    color: #44403c;
    box-shadow: var(--shadow-soft);
}

.quote-card {
    border-left: 3px solid var(--accent-mid);
    padding: 0.7rem 0.95rem;
    margin: 0.6rem 0;
    background: #f7faf9;
    color: #57534e;
    font-size: 0.92rem;
    border-radius: 0 8px 8px 0;
    line-height: 1.55;
}

.source-card {
    background: #fff;
    border: 1px solid var(--line-soft);
    border-radius: var(--radius);
    padding: 1.15rem 1.1rem;
    min-height: 148px;
    height: 148px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    box-shadow: var(--shadow-soft);
    transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
}
.source-card:hover {
    border-color: #d6d3d1;
    box-shadow: 0 4px 14px rgba(28, 25, 23, 0.06);
    transform: translateY(-2px);
}
.source-card h3 {
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 1.05rem;
    margin: 0 0 0.4rem;
    color: var(--ink);
}
.source-card p {
    color: var(--muted);
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.55;
    flex: 1;
}

/* Case Study pair — equal height, no fixed clip (Home cards keep .source-card) */
.case-pair-card {
    background: #fff;
    border: 1px solid var(--line-soft);
    border-radius: var(--radius);
    padding: 1.2rem 1.1rem;
    min-height: 236px;
    height: 100%;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    box-shadow: var(--shadow-soft);
    transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
}
.case-pair-card:hover {
    border-color: #d6d3d1;
    box-shadow: 0 4px 14px rgba(28, 25, 23, 0.06);
    transform: translateY(-2px);
}
.case-pair-card h3 {
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 1.05rem;
    margin: 0;
    color: var(--ink);
    line-height: 1.35;
}
.case-pair-card .case-pair-body {
    color: var(--muted);
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.55;
    flex: 1;
}
.case-pair-card .case-pair-source {
    margin: 0;
    padding-top: 0.35rem;
    border-top: 1px solid var(--line-soft);
    color: #78716c;
    font-size: 0.78rem;
    line-height: 1.45;
}

/* Soft hero + glyphs */
@keyframes softFadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.soft-hero {
    display: grid;
    grid-template-columns: 1.35fr 0.9fr;
    gap: 1.25rem 1.75rem;
    align-items: center;
    background:
        radial-gradient(420px 220px at 92% 18%, rgba(153, 246, 228, 0.35) 0%, transparent 60%),
        linear-gradient(145deg, #ffffff 0%, #f7fbfa 48%, #fbfaf8 100%);
    border: 1px solid var(--line-soft);
    border-radius: 14px;
    padding: 1.85rem 1.75rem 1.7rem;
    margin-bottom: 1.15rem;
    box-shadow: var(--shadow-soft);
    animation: softFadeUp 0.45s ease-out both;
}
.soft-hero__copy h1 {
    color: var(--ink);
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 2.15rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.03em;
    line-height: 1.15;
}
.soft-hero__copy .tagline {
    color: var(--muted);
    font-size: 1.02rem;
    margin: 0 0 0.85rem;
    line-height: 1.55;
    max-width: 34rem;
}
.soft-hero__copy .cta-line {
    color: #57534e;
    font-size: 0.9rem;
    line-height: 1.55;
    margin: 0;
    padding: 0.65rem 0.8rem;
    background: rgba(230, 247, 244, 0.55);
    border-radius: 8px;
    border: 1px solid #d5f0ea;
    max-width: 36rem;
}
.soft-hero__art {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 150px;
}
.soft-hero__art img {
    width: 100%;
    max-width: 260px;
    height: auto;
    display: block;
}
.case-soft-banner__art img { width: 100%; height: 100%; display: block; }
.workspace-soft__mark img { width: 20px; height: 20px; display: block; }
.soft-note {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.5;
    margin: 0 0 1.35rem;
    padding-left: 0.15rem;
}
.home-case-entry {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    background: linear-gradient(145deg, #ffffff 0%, #f7fbfa 100%);
    border: 1px solid var(--line-soft);
    border-radius: 12px;
    padding: 1.1rem 1.15rem;
    margin: 0 0 0.65rem;
    box-shadow: var(--shadow-soft);
}
.home-case-entry .glyph {
    margin-bottom: 0;
    flex-shrink: 0;
}
.home-case-entry__copy h3 {
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 1.08rem;
    font-weight: 600;
    margin: 0 0 0.3rem;
    color: var(--ink);
}
.home-case-entry__copy p {
    margin: 0;
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.55;
}
.glyph {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: var(--accent-soft);
    border: 1px solid #cceee8;
    margin-bottom: 0.55rem;
    transition: transform 0.18s ease;
}
.glyph img { width: 22px; height: 22px; display: block; }
.source-card:hover .glyph,
.case-pair-card:hover .glyph {
    transform: scale(1.06);
}
.glyph--stone {
    background: #f5f5f4;
    border-color: var(--line);
}
.glyph--warn {
    background: #fffbeb;
    border-color: #fde68a;
}

.case-soft-banner {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 1rem 1.5rem;
    align-items: center;
    background:
        radial-gradient(280px 160px at 100% 0%, rgba(153, 246, 228, 0.28) 0%, transparent 65%),
        linear-gradient(160deg, #ffffff 0%, #f8fbfa 100%);
    border: 1px solid var(--line-soft);
    border-radius: 14px;
    padding: 1.4rem 1.45rem 1.3rem;
    margin: 0 0 1.35rem;
    box-shadow: var(--shadow-soft);
    animation: softFadeUp 0.4s ease-out both;
}
.case-soft-banner .eyebrow {
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin: 0 0 0.4rem;
}
.case-soft-banner h2 {
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 1.35rem;
    font-weight: 600;
    margin: 0 0 0.45rem;
    color: var(--ink);
    letter-spacing: -0.02em;
}
.case-soft-banner .body {
    color: #57534e;
    font-size: 0.93rem;
    line-height: 1.65;
    margin: 0;
    max-width: 40rem;
}
.case-soft-banner__art {
    width: 108px;
    height: 108px;
    flex-shrink: 0;
}
.case-soft-banner__art svg { width: 100%; height: 100%; }

.workspace-soft {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    background: linear-gradient(180deg, #ffffff 0%, #f8fbfa 100%);
    border: 1px solid var(--line-soft);
    border-radius: 12px;
    padding: 0.85rem 1rem 0.9rem;
    margin: 0.15rem 0 0.85rem;
    box-shadow: 0 1px 2px rgba(28, 25, 23, 0.03);
    animation: softFadeUp 0.35s ease-out both;
}
.workspace-soft__mark {
    width: 36px;
    height: 36px;
    border-radius: 9px;
    background: var(--accent-soft);
    border: 1px solid #cceee8;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 0.1rem;
}
.workspace-soft__mark svg { width: 20px; height: 20px; }
.workspace-soft .title {
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin: 0 0 0.2rem;
}
.workspace-soft .meta {
    color: var(--muted);
    font-size: 0.86rem;
    line-height: 1.45;
    margin: 0;
}

@media (max-width: 820px) {
    .soft-hero {
        grid-template-columns: 1fr;
        padding: 1.4rem 1.2rem;
    }
    .soft-hero__art { order: -1; min-height: 120px; }
    .soft-hero__art img { max-width: 200px; }
    .case-soft-banner { grid-template-columns: 1fr; }
    .case-soft-banner__art { width: 88px; height: 88px; }
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
    min-height: 2.75rem !important;
    max-height: 2.75rem !important;
    padding: 0.35rem 0.75rem !important;
    align-items: center !important;
    border-radius: 8px !important;
}
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] > div {
    gap: 0.35rem !important;
}
div[data-testid="stFileUploader"] small {
    display: none !important;
}
div[data-testid="stFileUploader"] button {
    min-height: 2rem !important;
    padding-top: 0.25rem !important;
    padding-bottom: 0.25rem !important;
}

.workspace-bar {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 1rem;
    margin: 0.15rem 0 0.85rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid var(--line-soft);
}
.workspace-bar .title {
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.02em;
}
.workspace-bar .meta {
    color: var(--muted);
    font-size: 0.88rem;
}

.complaint-row {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.85rem 1rem;
    margin-bottom: 0.45rem;
    background: #fff;
    border: 1px solid var(--line-soft);
    border-radius: 8px;
}
.complaint-row .name { font-weight: 500; color: var(--ink); }
.complaint-row .meta { color: var(--muted); font-size: 0.85rem; }

.rigor-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 0.75rem;
    align-items: center;
    background: linear-gradient(165deg, #ffffff 0%, #f5faf8 100%);
    border: 1px solid var(--line-soft);
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin: 0.35rem 0 1.15rem;
    font-size: 0.86rem;
    color: #57534e;
    box-shadow: 0 1px 2px rgba(28, 25, 23, 0.025);
}
.rigor-strip .pill {
    display: inline-block;
    background: #fff;
    border: 1px solid var(--line-soft);
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    font-weight: 600;
    color: var(--ink);
    font-size: 0.78rem;
}
.rigor-strip .caveat {
    color: var(--muted);
    font-size: 0.8rem;
    width: 100%;
    line-height: 1.5;
}

.case-study-section {
    background: linear-gradient(165deg, #ffffff 0%, #fbfaf8 100%);
    border: 1px solid var(--line-soft);
    border-left: 3px solid var(--accent-mid);
    border-radius: var(--radius);
    padding: 1.5rem 1.5rem 1.35rem;
    margin: 0 0 1.5rem;
    box-shadow: var(--shadow-soft);
}
.case-study-section .eyebrow {
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin: 0 0 0.45rem;
}
.case-study-section h2 {
    font-family: "IBM Plex Serif", Georgia, serif;
    font-size: 1.4rem;
    font-weight: 600;
    margin: 0 0 0.55rem;
    color: var(--ink);
    letter-spacing: -0.02em;
}
.case-study-section .body {
    color: #57534e;
    font-size: 0.95rem;
    line-height: 1.7;
    margin: 0 0 0.85rem;
    max-width: 46rem;
}
.case-study-section .source {
    color: var(--muted);
    font-size: 0.8rem;
    margin: 0;
    line-height: 1.45;
}

.finding-card {
    background: #fff;
    border: 1px solid var(--line-soft);
    border-radius: var(--radius);
    padding: 1.05rem 1.15rem;
    margin: 0 0 0.7rem;
    box-shadow: 0 1px 2px rgba(28, 25, 23, 0.03);
}
.finding-card h4 {
    margin: 0 0 0.35rem;
    font-size: 1.02rem;
    color: var(--ink);
    font-family: "IBM Plex Serif", Georgia, serif;
    font-weight: 600;
}
.finding-card .meta {
    color: var(--muted);
    font-size: 0.82rem;
    margin-bottom: 0.45rem;
}

/* Soften Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent;
}
.stButton > button {
    border-radius: 8px !important;
    border: 1px solid var(--line) !important;
    font-weight: 500 !important;
    transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 1px 4px rgba(28, 25, 23, 0.06) !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
}
hr { border-color: var(--line-soft) !important; opacity: 0.7; }

/* Expandable analysis sections — less dense */
div[data-testid="stExpander"] {
    background: #fff;
    border: 1px solid var(--line-soft) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 0.65rem;
    box-shadow: 0 1px 2px rgba(28, 25, 23, 0.025);
    overflow: hidden;
}
div[data-testid="stExpander"] details {
    border: none !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    padding: 0.35rem 0.15rem !important;
}
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding-top: 0.35rem !important;
}

/* Soften alerts / info */
div[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--line-soft) !important;
}
div[data-baseweb="notification"],
div[data-testid="stNotification"] {
    border-radius: var(--radius) !important;
}

/* Inputs */
.stTextInput input, .stSelectbox [data-baseweb="select"] > div {
    border-radius: 8px !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Soft UI — inline SVG helpers (no external image assets)
# ---------------------------------------------------------------------------

def _svg_data_uri(svg: str) -> str:
    """Encode SVG for <img src> so Streamlit HTML sanitizer keeps it."""
    payload = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{payload}"


def _svg_img(svg: str, *, css_class: str = "", alt: str = "") -> str:
    cls = f' class="{css_class}"' if css_class else ""
    return f'<img{cls} src="{_svg_data_uri(svg)}" alt="{alt}" />'


def _svg_research_scene() -> str:
    """Abstract researcher metaphor: lens + quote bubbles + theme dots."""
    svg = """
<svg viewBox="0 0 260 180" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <ellipse cx="130" cy="158" rx="88" ry="10" fill="#e7e5e4" opacity="0.55"/>
  <rect x="28" y="36" width="92" height="72" rx="14" fill="#e6f7f4" stroke="#99f6e4" stroke-width="1.5"/>
  <circle cx="58" cy="64" r="16" stroke="#0f766e" stroke-width="3" fill="#fff"/>
  <path d="M69 75 L82 88" stroke="#0f766e" stroke-width="3" stroke-linecap="round"/>
  <circle cx="98" cy="52" r="4" fill="#0f766e" opacity="0.35"/>
  <circle cx="108" cy="68" r="3" fill="#0f766e" opacity="0.25"/>
  <rect x="138" y="28" width="86" height="44" rx="12" fill="#fff" stroke="#e7e5e4" stroke-width="1.5"/>
  <path d="M152 44 H198" stroke="#a8a29e" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M152 54 H184" stroke="#d6d3d1" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M160 72 L152 84 L176 72 Z" fill="#fff" stroke="#e7e5e4" stroke-width="1.5"/>
  <rect x="148" y="98" width="78" height="36" rx="11" fill="#f5f5f4" stroke="#e7e5e4" stroke-width="1.5"/>
  <path d="M160 112 H208" stroke="#a8a29e" stroke-width="2" stroke-linecap="round"/>
  <path d="M160 122 H192" stroke="#d6d3d1" stroke-width="2" stroke-linecap="round"/>
  <circle cx="48" cy="128" r="5" fill="#99f6e4"/>
  <circle cx="68" cy="136" r="4" fill="#0f766e" opacity="0.45"/>
  <circle cx="88" cy="124" r="3.5" fill="#57534e" opacity="0.25"/>
</svg>
""".strip()
    return _svg_img(svg)


def _svg_case_accent() -> str:
    """Compact mixed-methods mark for Case Study banner."""
    svg = """
<svg viewBox="0 0 108 108" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect width="108" height="108" rx="22" fill="#e6f7f4"/>
  <rect x="18" y="58" width="12" height="28" rx="3" fill="#0f766e" opacity="0.35"/>
  <rect x="36" y="44" width="12" height="42" rx="3" fill="#0f766e" opacity="0.55"/>
  <rect x="54" y="32" width="12" height="54" rx="3" fill="#0f766e"/>
  <rect x="76" y="24" width="18" height="18" rx="6" fill="#fff" stroke="#0f766e" stroke-width="2"/>
  <path d="M81 33 H89" stroke="#0f766e" stroke-width="1.8" stroke-linecap="round"/>
  <path d="M81 38 H86" stroke="#99f6e4" stroke-width="1.8" stroke-linecap="round"/>
</svg>
""".strip()
    return _svg_img(svg)


def _svg_workspace_mark() -> str:
    svg = """
<svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M4 5.5h12v9a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 4 14.5v-9Z" stroke="#0f766e" stroke-width="1.5"/>
  <path d="M7 8.5h6M7 11.5h4" stroke="#0f766e" stroke-width="1.5" stroke-linecap="round"/>
</svg>
""".strip()
    return _svg_img(svg)


def _glyph_svg(kind: str) -> str:
    """Tiny icons for connect / case cards."""
    icons = {
        "sample": """
<svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M5 6h12v11H5V6Z" stroke="#0f766e" stroke-width="1.6"/>
  <path d="M8 9.5h6M8 12.5h4" stroke="#0f766e" stroke-width="1.5" stroke-linecap="round"/>
</svg>""",
        "mock": """
<svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <circle cx="11" cy="11" r="6.5" stroke="#a16207" stroke-width="1.6"/>
  <path d="M11 8v3.2L13.2 13" stroke="#a16207" stroke-width="1.5" stroke-linecap="round"/>
</svg>""",
        "upload": """
<svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M11 14V6.5M11 6.5 8.2 9.2M11 6.5l2.8 2.7" stroke="#57534e" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M5.5 15.5v1A1.5 1.5 0 0 0 7 18h8a1.5 1.5 0 0 0 1.5-1.5v-1" stroke="#57534e" stroke-width="1.6" stroke-linecap="round"/>
</svg>""",
        "quant": """
<svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M5 16V10M10 16V7M15 16V5" stroke="#0f766e" stroke-width="2" stroke-linecap="round"/>
</svg>""",
        "qual": """
<svg viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M5 6.5h12v7.5H9L5.5 17V6.5Z" stroke="#0f766e" stroke-width="1.6" stroke-linejoin="round"/>
  <path d="M8 9.5h6M8 12h4" stroke="#0f766e" stroke-width="1.4" stroke-linecap="round"/>
</svg>""",
    }
    return icons.get(kind, icons["sample"]).strip()


def _glyph_html(kind: str, extra_class: str = "") -> str:
    cls = f"glyph {extra_class}".strip()
    return f'<div class="{cls}">{_svg_img(_glyph_svg(kind))}</div>'


# ---------------------------------------------------------------------------
# Mock Data Generation
# ---------------------------------------------------------------------------

PAIN_POINT_CATALOG: list[dict[str, Any]] = [
    {
        "topic": "Confusing onboarding tutorial",
        "stage": LifecycleStage.ONBOARDING,
        "base_rage": 78,
        "base_volume": 320,
        "diagnosis": "Users abandon before completing step 3 due to unclear value proposition.",
        "action": "Replace linear tutorial with interactive 60-second quick-win flow.",
        "priority": "P0",
        "retention_lift": 12.4,
    },
    {
        "topic": "Account verification timeout",
        "stage": LifecycleStage.ONBOARDING,
        "base_rage": 85,
        "base_volume": 180,
        "diagnosis": "Email OTP delays exceed 5 minutes, triggering immediate churn.",
        "action": "Add SMS fallback and real-time verification status indicator.",
        "priority": "P0",
        "retention_lift": 9.8,
    },
    {
        "topic": "Dashboard load performance",
        "stage": LifecycleStage.CORE_FEATURE,
        "base_rage": 62,
        "base_volume": 410,
        "diagnosis": "Initial render takes 8+ seconds on mobile networks.",
        "action": "Implement skeleton loading and lazy-load non-critical widgets.",
        "priority": "P0",
        "retention_lift": 8.2,
    },
    {
        "topic": "Export feature broken",
        "stage": LifecycleStage.CORE_FEATURE,
        "base_rage": 91,
        "base_volume": 95,
        "diagnosis": "CSV export fails silently for datasets > 500 rows — hidden killer.",
        "action": "Fix export pipeline; add progress bar and error toast notifications.",
        "priority": "P0",
        "retention_lift": 6.5,
    },
    {
        "topic": "Notification overload",
        "stage": LifecycleStage.DAILY_RETENTION,
        "base_rage": 55,
        "base_volume": 280,
        "diagnosis": "Push frequency exceeds user tolerance threshold (4+/day).",
        "action": "Introduce smart digest mode and granular notification preferences.",
        "priority": "P1",
        "retention_lift": 5.1,
    },
    {
        "topic": "Search relevance poor",
        "stage": LifecycleStage.CORE_FEATURE,
        "base_rage": 48,
        "base_volume": 350,
        "diagnosis": "Semantic search returns irrelevant results for domain-specific queries.",
        "action": "Fine-tune embedding model on product corpus; add query suggestions.",
        "priority": "P1",
        "retention_lift": 4.7,
    },
    {
        "topic": "Dark mode inconsistency",
        "stage": LifecycleStage.DAILY_RETENTION,
        "base_rage": 35,
        "base_volume": 120,
        "diagnosis": "Partial dark theme causes eye strain during extended sessions.",
        "action": "Audit all components for theme token compliance.",
        "priority": "P2",
        "retention_lift": 2.3,
    },
    {
        "topic": "Billing page confusion",
        "stage": LifecycleStage.CORE_FEATURE,
        "base_rage": 72,
        "base_volume": 210,
        "diagnosis": "Pricing tiers lack feature comparison matrix; users fear hidden fees.",
        "action": "Add transparent pricing table with annual/monthly toggle.",
        "priority": "P1",
        "retention_lift": 5.9,
    },
    {
        "topic": "Mobile gesture conflicts",
        "stage": LifecycleStage.DAILY_RETENTION,
        "base_rage": 58,
        "base_volume": 165,
        "diagnosis": "Swipe-to-delete conflicts with system back gesture on Android.",
        "action": "Replace swipe actions with long-press context menu.",
        "priority": "P1",
        "retention_lift": 3.8,
    },
    {
        "topic": "Password reset loop",
        "stage": LifecycleStage.ONBOARDING,
        "base_rage": 88,
        "base_volume": 45,
        "diagnosis": "Reset link expires before users check email — low volume, high rage.",
        "action": "Extend token TTL to 30 min; add in-app reset option.",
        "priority": "P0",
        "retention_lift": 4.2,
    },
]

POSITIVE_TOPICS: list[str] = [
    "Fast customer support response",
    "Clean UI design",
    "Reliable sync across devices",
    "Helpful in-app tips",
    "Fair pricing",
]


def generate_mock_reviews(n: int = MOCK_REVIEW_COUNT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic user review dataset simulating post-AI semantic coding output.

    Each row represents one user comment with pre-computed sentiment metrics
    that would normally come from LLM-based semantic encoding.

    Parameters
    ----------
    n : int
        Number of review records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: review_id, text, lifecycle_stage, sentiment, is_negative,
                  rage_index, topic, created_at
    """
    rng = np.random.default_rng(seed)

    stage_weights = [
        (LifecycleStage.ONBOARDING, 0.35),
        (LifecycleStage.CORE_FEATURE, 0.40),
        (LifecycleStage.DAILY_RETENTION, 0.25),
    ]
    stages = [s for s, _ in stage_weights]
    probs = np.asarray([w for _, w in stage_weights], dtype=float)

    rows: list[dict[str, Any]] = []

    for i in range(n):
        # Index-based choice: np.choice on Enum list returns numpy.str_ (no .value)
        stage = stages[int(rng.choice(len(stages), p=probs))]

        # 62% of reviews carry identifiable pain points (negative signal)
        is_negative = bool(rng.random() < 0.62)

        if is_negative:
            pain = PAIN_POINT_CATALOG[int(rng.integers(0, len(PAIN_POINT_CATALOG)))]
            topic = str(pain["topic"])
            rage_base = float(pain["base_rage"])
            rage_index = float(np.clip(rage_base + rng.normal(0, 12), 10, 100))
            sentiment = float(np.clip(-0.9 + rng.normal(0, 0.15), -1.0, -0.2))
        else:
            topic = str(POSITIVE_TOPICS[int(rng.integers(0, len(POSITIVE_TOPICS)))])
            rage_index = float(np.clip(rng.normal(15, 10), 0, 40))
            sentiment = float(np.clip(0.3 + rng.normal(0, 0.2), 0.1, 1.0))

        rows.append(
            {
                "review_id": f"REV-{i + 1:05d}",
                "text": _generate_review_text(topic, is_negative, rng),
                "lifecycle_stage": stage.value,
                "sentiment": round(sentiment, 3),
                "is_negative": is_negative,
                "rage_index": round(rage_index, 1),
                "topic": topic,
                "created_at": pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 365))),
            }
        )

    return pd.DataFrame(rows)


def _generate_review_text(topic: str, is_negative: bool, rng: np.random.Generator) -> str:
    """Generate plausible review snippet for mock data."""
    if is_negative:
        templates = [
            f"I tried to use the app but {topic.lower()} really frustrated me. Almost deleted it.",
            f"Seriously annoyed — {topic.lower()} made me want to quit immediately.",
            f"The worst part is {topic.lower()}. Fix this or I'm gone.",
            f"Can't believe {topic.lower()} is still an issue in 2025.",
        ]
    else:
        templates = [
            f"Love this app! Especially {topic.lower()} — keeps me coming back.",
            f"{topic.capitalize()} is exactly why I recommend this to friends.",
            f"Been using for 3 months. {topic.capitalize()} makes daily use a breeze.",
        ]
    return rng.choice(templates)


# ---------------------------------------------------------------------------
# Metrics Calculation (Dashboard Aggregations)
# ---------------------------------------------------------------------------


def compute_funnel_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate negative sentiment ratio per lifecycle stage for the drop-out funnel.

    Returns localized stage and risk labels for display.
    """
    records: list[dict[str, Any]] = []

    for stage_enum, meta in STAGE_METADATA.items():
        stage_df = df[df["lifecycle_stage"] == stage_enum.value]
        total = len(stage_df)
        negative_count = int(stage_df["is_negative"].sum())
        negative_pct = (negative_count / total * 100) if total > 0 else 0.0

        records.append(
            {
                "stage": translate_lifecycle_stage(stage_enum.value),
                "stage_key": stage_enum.value,
                "negative_pct": round(negative_pct, 1),
                "negative_count": negative_count,
                "total_count": total,
                "risk_label": translate_risk(meta.risk_label),
                "risk_emoji": meta.risk_emoji,
                "color": meta.risk_color,
            }
        )

    return pd.DataFrame(records)


def compute_rage_volume_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build topic-level rage-volume matrix for quadrant scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded review dataframe.

    Returns
    -------
    pd.DataFrame
        Columns: topic, volume, avg_rage, quadrant, priority_hint
    """
    negative_df = df[df["is_negative"]].copy()
    if negative_df.empty:
        return pd.DataFrame(columns=["topic", "volume", "avg_rage", "quadrant"])

    grouped = (
        negative_df.groupby("topic")
        .agg(volume=("review_id", "count"), avg_rage=("rage_index", "mean"))
        .reset_index()
    )
    grouped["avg_rage"] = grouped["avg_rage"].round(1)

    volume_median = grouped["volume"].median()
    rage_median = grouped["avg_rage"].median()

    def assign_quadrant(row: pd.Series) -> str:
        high_vol = row["volume"] >= volume_median
        high_rage = row["avg_rage"] >= rage_median
        if high_vol and high_rage:
            return "Red Core Blast Zone"
        if not high_vol and high_rage:
            return "Purple Hidden Sting Zone"
        if high_vol and not high_rage:
            return "Yellow Monitor Zone"
        return "Green Low Priority Zone"

    grouped["quadrant"] = grouped.apply(assign_quadrant, axis=1)
    return grouped.sort_values("avg_rage", ascending=False)


# build_implementation_roadmap imported from services.roadmap_generator


def compute_summary_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Compute top-level KPI metrics for dashboard header cards."""
    total = len(df)
    negative_pct = df["is_negative"].mean() * 100 if total > 0 else 0
    avg_rage = df.loc[df["is_negative"], "rage_index"].mean() if df["is_negative"].any() else 0
    unique_topics = df.loc[df["is_negative"], "topic"].nunique()

    lifecycle_mode = str(df.attrs.get("lifecycle_mode") or "active")
    if "lifecycle_stage" in df.columns and total:
        if float((df["lifecycle_stage"].astype(str) == "General feedback").mean()) >= 0.85:
            lifecycle_mode = "disabled"

    highest_risk_stage = "N/A"
    if lifecycle_mode != "disabled":
        funnel = compute_funnel_metrics(df)
        highest_risk_stage = (
            translate_lifecycle_stage(funnel.loc[funnel["negative_pct"].idxmax(), "stage_key"])
            if len(funnel) > 0
            else "N/A"
        )

    return {
        "total_reviews": total,
        "negative_pct": round(negative_pct, 1),
        "avg_rage": round(avg_rage, 1),
        "unique_pain_topics": unique_topics,
        "highest_risk_stage": highest_risk_stage,
        "lifecycle_mode": lifecycle_mode,
    }


# ---------------------------------------------------------------------------
# Visualization Components
# ---------------------------------------------------------------------------

QUADRANT_COLORS: dict[str, str] = {
    "Red Core Blast Zone": "#E74C3C",
    "Purple Hidden Sting Zone": "#9B59B6",
    "Yellow Monitor Zone": "#F39C12",
    "Green Low Priority Zone": "#27AE60",
}


def render_hero_header() -> None:
    """Render the professional dashboard hero section."""
    st.markdown(
        f"""
        <div class="dashboard-hero">
            <h1>{t("hero_title")}</h1>
            <div class="tagline">{t("hero_tagline")}</div>
            <div class="logic-block">{t("hero_logic")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_dropout_funnel_chart(funnel_df: pd.DataFrame) -> go.Figure:
    """
    Create horizontal bar chart showing negative sentiment % per lifecycle stage.

    Parameters
    ----------
    funnel_df : pd.DataFrame
        Output of compute_funnel_metrics().

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    labels = [
        f"{row['risk_emoji']} {row['stage']}  ({t('funnel_chart_risk', risk=row['risk_label'])})"
        for _, row in funnel_df.iterrows()
    ]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=funnel_df["negative_pct"],
            orientation="h",
            marker=dict(
                color=funnel_df["color"],
                line=dict(color="rgba(28,25,23,0.08)", width=1),
            ),
            text=[f"{v:.1f}%" for v in funnel_df["negative_pct"]],
            textposition="outside",
            textfont=dict(color="#1c1917", size=12),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + t("funnel_hover_negative", value="%{x:.1f}") + "<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=t("chart_funnel_title"),
            font=dict(size=15, color="#1c1917", family="IBM Plex Serif"),
        ),
        xaxis=dict(
            title=t("chart_funnel_x"),
            range=[0, max(funnel_df["negative_pct"].max() * 1.25, 10)],
            gridcolor="#e7e5e4",
            color="#78716c",
        ),
        yaxis=dict(color="#44403c"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=320,
        margin=dict(l=20, r=80, t=50, b=40),
        font=dict(color="#44403c"),
    )

    return fig


def create_rage_volume_scatter(matrix_df: pd.DataFrame) -> go.Figure:
    """
    Create quadrant scatter plot for rage-volume matrix analysis.

    Parameters
    ----------
    matrix_df : pd.DataFrame
        Output of compute_rage_volume_matrix().

    Returns
    -------
    go.Figure
        Plotly figure with quadrant reference lines and zone annotations.
    """
    if matrix_df.empty:
        fig = go.Figure()
        fig.add_annotation(text=t("chart_no_negative"), showarrow=False)
        return fig

    plot_df = matrix_df.copy()
    plot_df["quadrant_display"] = plot_df["quadrant"].map(translate_quadrant)

    volume_median = plot_df["volume"].median()
    rage_median = plot_df["avg_rage"].median()
    max_vol = plot_df["volume"].max() * 1.15
    max_rage = min(plot_df["avg_rage"].max() * 1.1, 100)

    translated_colors = {
        translate_quadrant(k): v for k, v in QUADRANT_COLORS.items()
    }

    fig = px.scatter(
        plot_df,
        x="volume",
        y="avg_rage",
        color="quadrant_display",
        color_discrete_map=translated_colors,
        hover_name="topic",
        size="volume",
        size_max=40,
        labels={
            "volume": t("chart_matrix_x"),
            "avg_rage": t("chart_matrix_y"),
            "quadrant_display": t("chart_zone"),
        },
    )

    fig.add_hline(
        y=rage_median,
        line_dash="dash",
        line_color="rgba(28,25,23,0.25)",
        annotation_text=t("chart_rage_median", value=rage_median),
        annotation_font_color="#78716c",
    )
    fig.add_vline(
        x=volume_median,
        line_dash="dash",
        line_color="rgba(28,25,23,0.25)",
        annotation_text=t("chart_volume_median", value=volume_median),
        annotation_font_color="#78716c",
    )

    annotations = [
        dict(x=max_vol * 0.85, y=max_rage * 0.95, text=f"● {translate_quadrant('Red Core Blast Zone')[:28]}", showarrow=False, font=dict(color="#b91c1c", size=11)),
        dict(x=max_vol * 0.08, y=max_rage * 0.95, text=f"● {translate_quadrant('Purple Hidden Sting Zone')[:28]}", showarrow=False, font=dict(color="#6b21a8", size=11)),
        dict(x=max_vol * 0.85, y=max_rage * 0.08, text=f"● {translate_quadrant('Yellow Monitor Zone')[:28]}", showarrow=False, font=dict(color="#a16207", size=11)),
        dict(x=max_vol * 0.08, y=max_rage * 0.08, text=f"● {translate_quadrant('Green Low Priority Zone')[:28]}", showarrow=False, font=dict(color="#15803d", size=11)),
    ]
    fig.update_layout(annotations=annotations)

    fig.update_layout(
        title=dict(text=t("chart_matrix_title"), font=dict(size=15, color="#1c1917", family="IBM Plex Serif")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=480,
        legend=dict(title=t("chart_zone"), font=dict(color="#44403c")),
        xaxis=dict(gridcolor="#e7e5e4", color="#78716c"),
        yaxis=dict(gridcolor="#e7e5e4", color="#78716c", range=[0, max_rage]),
        font=dict(color="#44403c"),
    )

    return fig


def style_roadmap_dataframe(roadmap_df: pd.DataFrame) -> Any:
    """
    Apply conditional styling to the implementation roadmap table.

    Parameters
    ----------
    roadmap_df : pd.DataFrame
        Roadmap dataframe.

    Returns
    -------
    Styler
        Pandas Styler object for st.dataframe.
    """

    def highlight_priority(val: str) -> str:
        colors = {"P0": "#ffebee", "P1": "#fff8e1", "P2": "#e8f5e9"}
        text_colors = {"P0": "#b71c1c", "P1": "#e65100", "P2": "#1b5e20"}
        bg = colors.get(val, "")
        fg = text_colors.get(val, "")
        if bg:
            return f"background-color: {bg}; color: {fg}; font-weight: 700;"
        return ""

    return roadmap_df.style.map(highlight_priority, subset=["Priority"])


def render_evidence_chain(df: pd.DataFrame, roadmap_df: pd.DataFrame) -> None:
    """
    Render evidence chain panel — links roadmap items to verbatim source reviews.

    Parameters
    ----------
    df : pd.DataFrame
        Full encoded review dataframe.
    roadmap_df : pd.DataFrame
        Implementation roadmap table.
    """
    if roadmap_df.empty:
        return

    st.markdown(
        f'<p class="section-header">{t("evidence_title")}</p>',
        unsafe_allow_html=True,
    )
    st.caption(t("evidence_caption"))

    # Build select options with priority + mention count
    topic_options: list[str] = []
    topic_labels: list[str] = []
    for _, row in roadmap_df.iterrows():
        topic = str(row["Core Pain Point"])
        priority = str(row["Priority"])
        count = summarize_topic_evidence(df, topic)["total_count"]
        topic_options.append(topic)
        topic_labels.append(f"[{priority}] {topic}  ({count:,} reviews)")

    # Default selection: first P0, else first row
    default_index = 0
    for i, (_, row) in enumerate(roadmap_df.iterrows()):
        if row["Priority"] == "P0":
            default_index = i
            break

    selected_label = st.selectbox(
        t("evidence_select"),
        options=topic_labels,
        index=default_index,
        key="evidence_topic_selector",
    )
    selected_topic = topic_options[topic_labels.index(selected_label)]
    selected_roadmap_row = roadmap_df.loc[
        roadmap_df["Core Pain Point"] == selected_topic
    ].iloc[0]

    # Summary header for selected pain point
    summary = summarize_topic_evidence(df, selected_topic)
    evidence = fetch_topic_evidence(df, selected_topic, max_rows=15)

    h1, h2, h3, h4 = st.columns(4)
    h1.metric(t("evidence_matching"), f"{summary['total_count']:,}")
    h2.metric(t("evidence_avg_rage"), summary["avg_rage"])
    h3.metric(t("evidence_avg_sentiment"), summary["avg_sentiment"])
    if summary["has_rating"] and summary["avg_rating"] is not None:
        h4.metric(t("evidence_avg_rating"), summary["avg_rating"])
    else:
        h4.metric(t("evidence_priority"), selected_roadmap_row["Priority"])

    # Lifecycle breakdown
    if summary["lifecycle_breakdown"]:
        breakdown_parts = [
            f"**{translate_lifecycle_stage(stage)}**: {count}"
            for stage, count in summary["lifecycle_breakdown"].items()
        ]
        st.markdown(f"**{t('evidence_lifecycle')}:** " + " · ".join(breakdown_parts))

    with st.container(border=True):
        st.markdown(f"**{t('evidence_diagnosis')}:** {selected_roadmap_row['AI Diagnosis']}")
        st.markdown(f"**{t('evidence_action')}:** {selected_roadmap_row['Recommended Action']}")

    if evidence.empty:
        st.warning(t("evidence_no_reviews"))
        return

    display_df = format_evidence_for_display(evidence)
    shown = len(evidence)
    total = summary["total_count"]
    st.markdown(
        f"**{t('evidence_showing', shown=shown, total=total)}**"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(60 + shown * 42, 480),
        column_config={
            "User Comment": st.column_config.TextColumn(
                t("col_user_comment"),
                width="large",
                help=t("help_user_comment"),
            ),
            "Rage Index": st.column_config.ProgressColumn(
                t("col_rage_index"),
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Sentiment": st.column_config.NumberColumn(
                t("col_sentiment"),
                format="%.2f",
                help=t("help_sentiment"),
            ),
        },
    )

    with st.expander(t("evidence_quotes")):
        for i, (_, row) in enumerate(evidence.head(3).iterrows(), start=1):
            rage = row["rage_index"]
            stage = translate_lifecycle_stage(str(row.get("lifecycle_stage", "N/A")))
            st.markdown(
                f"**{t('evidence_quote_line', n=i, rage=int(rage), stage=stage)}**  \n"
                f"> \"{row['text']}\""
            )


# ---------------------------------------------------------------------------
# Analysis Pipeline
# ---------------------------------------------------------------------------


def run_analysis_pipeline(raw_df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Execute full semantic encoding pipeline on normalized review data.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Normalized reviews from data_loader (review_id, text, rating, etc.).
    source_label : str
        Label for session state tracking (upload / sample / mock).

    Returns
    -------
    pd.DataFrame
        Fully encoded review dataframe ready for dashboard rendering.
    """
    progress_bar = st.progress(0, text=t("progress_init"))
    status_text = st.empty()

    def on_progress(pct: float, message: str) -> None:
        progress_bar.progress(min(pct, 1.0), text=message)
        status_text.caption(message)

    encoded_df = encode_reviews(raw_df, progress_callback=on_progress)

    progress_bar.progress(1.0, text=t("progress_complete"))
    topic_engine = encoded_df.attrs.get("topic_method", "unknown")
    engine_label = {
        "bertopic": t("engine_bertopic"),
        "tfidf": t("engine_tfidf"),
    }.get(topic_engine, topic_engine)
    status_text.success(
        t(
            "analysis_done",
            total=len(encoded_df),
            negative=int(encoded_df["is_negative"].sum()),
            topics=encoded_df["topic"].nunique(),
            engine=engine_label,
        )
    )

    st.session_state["review_data"] = encoded_df
    st.session_state["data_source"] = source_label
    if encoded_df.attrs.get("cache_hit"):
        st.caption(f"⚡ Cache hit · loader {LOADER_VERSION}")
    return encoded_df


# ---------------------------------------------------------------------------
# Upload Wizard — Preflight + Column Mapping
# ---------------------------------------------------------------------------

FIELD_LABELS: dict[str, str] = {}  # populated via _field_labels()


def _field_labels() -> dict[str, str]:
    return {
        "text": t("field_review_text"),
        "summary": t("field_summary"),
        "rating": t("field_rating"),
        "created_at": t("field_date"),
        "review_id": t("field_review_id"),
        "lifecycle_stage": t("field_lifecycle"),
        "topic": t("field_topic"),
    }


REQUIRED_UPLOAD_FIELDS = {"text"}


def _none_option() -> str:
    return t("none_skip")


def _column_options(raw_df: pd.DataFrame) -> list[str]:
    """Build selectbox options: None + all dataframe columns."""
    return [_none_option()] + [str(c) for c in raw_df.columns]


def _default_index(options: list[str], mapped_col: str | None) -> int:
    """Find selectbox default index for a mapped column."""
    if mapped_col and mapped_col in options:
        return options.index(mapped_col)
    return 0


def render_preflight_report(report: PreflightReport) -> None:
    """Display pre-analysis validation report."""
    st.markdown(f"#### 🔍 {t('preflight_title')}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("preflight_total"), f"{report.total_rows:,}")
    c2.metric(t("preflight_valid"), f"{report.valid_rows:,}")
    c3.metric(t("preflight_platform"), report.detected_platform[:22])
    c4.metric(t("preflight_confidence"), report.mapping_confidence.upper())

    if report.ready:
        st.success(t("preflight_ready", count=report.valid_rows))
    else:
        st.error(t("preflight_not_ready"))

    for warning in report.warnings:
        st.warning(warning)


def render_column_mapping_ui(raw_df: pd.DataFrame, auto_mapping: dict[str, str | None]) -> dict[str, str | None]:
    """
    Render manual column mapping form with auto-detected defaults.

    Returns
    -------
    dict[str, str | None]
        User-confirmed column mapping.
    """
    st.markdown(f"#### 🗂️ {t('column_mapping')}")
    st.caption(t("column_mapping_caption"))

    options = _column_options(raw_df)
    confirmed: dict[str, str | None] = {}
    labels = _field_labels()

    primary_fields = ["text", "summary", "rating", "created_at", "review_id"]
    cols = st.columns(2)
    for i, field in enumerate(primary_fields):
        with cols[i % 2]:
            label = labels.get(field, field)
            selected = st.selectbox(
                label,
                options=options,
                index=_default_index(options, auto_mapping.get(field)),
                key=f"map_{field}",
                help=t("help_required") if field in REQUIRED_UPLOAD_FIELDS else t("help_optional"),
            )
            confirmed[field] = None if selected == _none_option() else selected

    with st.expander(t("advanced_mapping")):
        for field in ["lifecycle_stage", "topic"]:
            label = labels.get(field, field)
            selected = st.selectbox(
                label,
                options=options,
                index=_default_index(options, auto_mapping.get(field)),
                key=f"map_{field}",
            )
            confirmed[field] = None if selected == _none_option() else selected

    text_col = confirmed.get("text")
    if text_col and text_col in raw_df.columns:
        with st.expander(t("preview_mapped")):
            preview = raw_df[text_col].astype(str).head(3)
            for i, val in enumerate(preview):
                st.text(f"{t('preview_row', n=i + 1)}: {str(val)[:200]}{'...' if len(str(val)) > 200 else ''}")

    return confirmed


def _clear_upload_session() -> None:
    """Drop the in-progress upload wizard and reset the file picker widget."""
    for key in (
        "raw_upload_df",
        "upload_filename",
        "auto_column_mapping",
        "column_mapping",
        "last_upload_key",
        "upload_wizard_active",
    ):
        st.session_state.pop(key, None)
    # Bump widget key so Streamlit forgets the selected file
    st.session_state["file_uploader_version"] = int(
        st.session_state.get("file_uploader_version") or 0
    ) + 1


def handle_file_upload(uploaded: Any) -> None:
    """
    Process uploaded file: parse → preflight → column mapping → analyze on confirm.
    """
    file_key = f"{uploaded.name}_{uploaded.size}"

    # New file uploaded — reset wizard state
    if st.session_state.get("last_upload_key") != file_key:
        try:
            raw_df = read_raw_file(uploaded)
            auto_mapping, platform, confidence = auto_resolve_columns(raw_df)
            st.session_state["raw_upload_df"] = raw_df
            st.session_state["upload_filename"] = uploaded.name
            st.session_state["auto_column_mapping"] = auto_mapping
            st.session_state["column_mapping"] = auto_mapping
            st.session_state["last_upload_key"] = file_key
            st.session_state["upload_wizard_active"] = True
            # Clear previous analysis when new file arrives
            st.session_state.pop("review_data", None)
        except Exception as exc:
            st.error(t("error_read_file", error=exc))
            return

    if not st.session_state.get("upload_wizard_active"):
        return

    raw_df: pd.DataFrame = st.session_state["raw_upload_df"]
    auto_mapping: dict[str, str | None] = st.session_state.get("auto_column_mapping", {})

    st.markdown("---")
    head_l, head_r = st.columns([4, 1])
    with head_l:
        st.markdown(f"### 📂 {t('uploaded_file')}: `{st.session_state.get('upload_filename', 'file')}`")
        st.caption(t("upload_rows_cols", rows=len(raw_df), cols=len(raw_df.columns)))
    with head_r:
        if st.button(
            _label("upload_remove", "Remove upload"),
            use_container_width=True,
            key="upload_remove_btn",
            help=_label(
                "upload_remove_help",
                "Clear this file and start over — does not delete files on disk.",
            ),
        ):
            _clear_upload_session()
            st.rerun()

    # Column mapping form
    confirmed_mapping = render_column_mapping_ui(raw_df, auto_mapping)
    st.session_state["column_mapping"] = confirmed_mapping

    # Preflight check with confirmed mapping
    report = run_preflight(raw_df, confirmed_mapping)
    render_preflight_report(report)

    # Analyze / remove
    analyze_col, remove_col, _ = st.columns([1, 1, 2])
    with analyze_col:
        analyze_clicked = st.button(
            f"✅ {t('confirm_analyze')}",
            type="primary",
            disabled=not report.ready,
            use_container_width=True,
            key="confirm_analyze_btn",
        )
    with remove_col:
        if st.button(
            _label("upload_remove", "Remove upload"),
            use_container_width=True,
            key="upload_remove_btn_bottom",
        ):
            _clear_upload_session()
            st.rerun()

    if analyze_clicked and report.ready:
        try:
            normalized = normalize_raw_reviews(raw_df, confirmed_mapping)
            filename = st.session_state.get("upload_filename", "upload")
            run_analysis_pipeline(normalized, source_label=f"Upload ({filename})")
            st.session_state["upload_wizard_active"] = False
            st.session_state["nav_page"] = NAV_ANALYSIS
            st.rerun()
        except Exception as exc:
            st.error(t("error_analysis_failed", error=exc))


# ---------------------------------------------------------------------------
# Data Ingestion Layer (UI)
# ---------------------------------------------------------------------------


def _language_display_name(code: str) -> str:
    """Human-readable language name for profile chart."""
    return SUPPORTED_LANGUAGES.get(code, code.upper() if code != "unknown" else "Unknown")


def render_language_profile(df: pd.DataFrame) -> None:
    """Show detected review languages when language column is present."""
    if "language" not in df.columns:
        return

    st.markdown(
        f'<p class="section-header">🌐 {t("lang_detect_title")}</p>',
        unsafe_allow_html=True,
    )
    st.caption(t("lang_detect_caption"))

    lang_counts = df["language"].value_counts().head(8)
    profile_df = pd.DataFrame(
        {
            "language": [_language_display_name(str(c)) for c in lang_counts.index],
            "count": lang_counts.values,
        }
    )
    fig = px.bar(
        profile_df,
        x="language",
        y="count",
        labels={"language": t("ui_language"), "count": t("metric_total")},
        color="count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=280,
        showlegend=False,
        font=dict(color="#44403c"),
        xaxis=dict(color="#78716c"),
        yaxis=dict(color="#78716c", gridcolor="#e7e5e4"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_credibility_panel(df: pd.DataFrame) -> dict:
    """Show analysis quality / confidence so users don't over-trust noisy runs."""
    cred = df.attrs.get("credibility")
    if not isinstance(cred, dict):
        from loader import compute_credibility_report

        cred = compute_credibility_report(df)
        df.attrs["credibility"] = cred

    st.markdown(
        f'<p class="section-header">🧪 {t("credibility_title")}</p>',
        unsafe_allow_html=True,
    )
    st.caption(t("credibility_caption"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("credibility_score"), f"{cred.get('confidence', 0)}/100")
    c2.metric(t("credibility_grade"), cred.get("grade", "?"))
    metrics = cred.get("metrics") or {}
    c3.metric(t("credibility_uncat"), f"{metrics.get('uncategorized_negative_pct', 0)}%")
    c4.metric(t("credibility_merged"), metrics.get("topics_merged", 0))

    if metrics.get("topics_before_merge") and metrics.get("topics_after_merge"):
        st.caption(
            t(
                "credibility_merge_note",
                before=metrics["topics_before_merge"],
                after=metrics["topics_after_merge"],
            )
        )

    for warning in cred.get("warnings") or []:
        st.warning(warning)

    estimate_note = metrics.get("analysis_purpose", "discover + prioritize + evidence")
    st.info(t("credibility_retention_note", kind=estimate_note))
    return cred


def render_behavior_controls() -> None:
    """Optional behavior/retention CSV join controls in the data area."""
    st.markdown(
        f'<p class="section-header">📈 {t("behavior_title")}</p>',
        unsafe_allow_html=True,
    )
    st.caption(t("behavior_caption"))

    b1, b2 = st.columns(2)
    with b1:
        uploaded = st.file_uploader(
            t("behavior_upload"),
            type=["csv", "xlsx", "xls"],
            key="behavior_uploader",
        )
        if uploaded is not None:
            try:
                raw = load_behavior_dataframe(uploaded)
                behavior = normalize_behavior_df(raw)
                st.session_state["behavior_data"] = behavior
                st.success(t("behavior_loaded", count=len(behavior)))
            except Exception as exc:
                st.error(t("error_analysis_failed", error=exc))
    with b2:
        if SAMPLE_BEHAVIOR_CSV.exists():
            if st.button(t("behavior_sample_btn"), use_container_width=True, key="load_behavior_sample"):
                try:
                    raw = load_behavior_dataframe(SAMPLE_BEHAVIOR_CSV)
                    behavior = normalize_behavior_df(raw)
                    st.session_state["behavior_data"] = behavior
                    st.success(t("behavior_loaded", count=len(behavior)))
                except Exception as exc:
                    st.error(t("error_analysis_failed", error=exc))
        if st.session_state.get("behavior_data") is not None:
            if st.button(t("behavior_clear"), use_container_width=True, key="clear_behavior"):
                st.session_state.pop("behavior_data", None)
                st.rerun()


def render_history_panel(df: pd.DataFrame, roadmap_df: pd.DataFrame, credibility: dict) -> None:
    """Save snapshots and compare two analysis runs."""
    st.markdown(
        f'<p class="section-header">🕘 {t("history_title")}</p>',
        unsafe_allow_html=True,
    )
    st.caption(t("history_caption"))

    h1, h2 = st.columns([1, 2])
    with h1:
        if st.button(t("history_save"), type="primary", use_container_width=True, key="save_snapshot_btn"):
            snap = save_analysis_snapshot(
                df,
                roadmap_df,
                source_label=str(st.session_state.get("data_source", "Unknown")),
                credibility=credibility,
            )
            st.success(t("history_saved", run_id=snap["id"]))

        if st.button(t("cache_clear"), use_container_width=True, key="clear_cache_btn"):
            n = clear_analysis_cache()
            st.info(t("cache_cleared", count=n))

    snapshots = list_snapshots(limit=15)
    if not snapshots:
        st.caption(t("history_empty"))
        return

    st.dataframe(snapshots_to_dataframe(snapshots), use_container_width=True, hide_index=True)

    ids = [s["id"] for s in snapshots]
    if len(ids) < 2:
        st.caption(t("history_need_two"))
        return

    c1, c2 = st.columns(2)
    with c1:
        older_id = st.selectbox(t("history_older"), ids[1:], key="hist_older")
    with c2:
        newer_id = st.selectbox(t("history_newer"), ids, key="hist_newer")

    if st.button(t("history_compare"), use_container_width=True, key="compare_btn"):
        older = load_snapshot(older_id)
        newer = load_snapshot(newer_id)
        if not older or not newer:
            st.error(t("history_load_failed"))
            return
        diff = compare_snapshots(older, newer)
        k = diff["kpi_delta"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Δ Negative %", k["negative_pct"])
        m2.metric("Δ Avg Rage", k["avg_rage"])
        m3.metric("Δ Topics", k["unique_topics"])
        m4.metric("Δ Reviews", k["total_reviews"])

        if diff["new_topics"]:
            st.markdown(f"**{t('history_new_topics')}:** " + ", ".join(diff["new_topics"][:12]))
        if diff["resolved_topics"]:
            st.markdown(f"**{t('history_resolved_topics')}:** " + ", ".join(diff["resolved_topics"][:12]))
        if diff["priority_changes"]:
            st.markdown(f"**{t('history_priority_changes')}**")
            st.dataframe(pd.DataFrame(diff["priority_changes"]), use_container_width=True, hide_index=True)
        elif diff["shared_topic_changes"]:
            st.markdown(f"**{t('history_volume_changes')}**")
            st.dataframe(
                pd.DataFrame(diff["shared_topic_changes"]).head(8),
                use_container_width=True,
                hide_index=True,
            )


def _label(key: str, fallback: str) -> str:
    """Translate with hard fallback so UI never shows raw key names."""
    value = t(key)
    return fallback if (not value or value == key) else value


def _material_label(icon: str, text: str) -> str:
    """Streamlit Material Symbol + label (standard nav affordance)."""
    return f":material/{icon}: {text}"


def _material_icon(icon: str) -> str:
    """Icon-only Material Symbol (compact sidebar rail)."""
    return f":material/{icon}:"


NAV_HOME = "home"
NAV_CASE = "case"
NAV_ANALYSIS = "analysis"


def _go_home() -> None:
    """Site Home — landing / connect sources."""
    st.session_state["nav_page"] = NAV_HOME
    st.rerun()


def _go_case() -> None:
    """Site Case Study page."""
    st.session_state["nav_page"] = NAV_CASE
    st.rerun()


def _go_analysis() -> None:
    """Single-page analysis workspace (after data is loaded)."""
    st.session_state["nav_page"] = NAV_ANALYSIS
    st.rerun()


def _go_back() -> None:
    """
    Back lives in the main content chrome (not the site nav).
    Prefer previous site page; otherwise Home.
    """
    history = list(st.session_state.get("nav_history") or [])
    if history:
        prev = history.pop()
        st.session_state["nav_history"] = history
        st.session_state["nav_page"] = prev
        st.rerun()
    current = st.session_state.get("nav_page", NAV_HOME)
    if current == NAV_ANALYSIS:
        st.session_state["nav_page"] = NAV_HOME
    elif current == NAV_CASE:
        st.session_state["nav_page"] = NAV_HOME
    else:
        st.session_state["nav_page"] = NAV_HOME
    st.rerun()


def _push_nav_history(page: str) -> None:
    history = list(st.session_state.get("nav_history") or [])
    if not history or history[-1] != page:
        history.append(page)
        st.session_state["nav_history"] = history[-20:]


def _go_to_page(page: str, *, record: bool = True) -> None:
    current = st.session_state.get("nav_page")
    if record and current and current != page:
        _push_nav_history(current)
    st.session_state["nav_page"] = page
    st.rerun()


def _default_nav_page(has_data: bool) -> str:
    page = st.session_state.get("nav_page")
    if page in (NAV_HOME, NAV_CASE, NAV_ANALYSIS):
        if page == NAV_ANALYSIS and not has_data:
            return NAV_HOME
        return page
    # Migrate legacy process-step pages → single analysis page
    if has_data:
        return NAV_ANALYSIS
    return NAV_HOME


def _clear_analysis_session(*, clear_strand_snapshots: bool = False) -> None:
    """Drop the loaded corpus and analysis-scoped session state."""
    for key in (
        "review_data",
        "data_source",
        "qa_result",
        "upload_wizard_active",
        "last_upload_key",
        "raw_upload_df",
        "upload_filename",
        "auto_column_mapping",
        "column_mapping",
        "nav_history",
        "brief_overrides",
        "brief_edit_nonce",
        "roadmap_overrides",
        "roadmap_edit_nonce",
        "sidebar_panel",
        "current_project_id",
        "current_project_name",
        "behavior_data",
    ):
        st.session_state.pop(key, None)
    st.session_state["file_uploader_version"] = int(
        st.session_state.get("file_uploader_version") or 0
    ) + 1
    if clear_strand_snapshots:
        st.session_state.pop("strand_snapshots", None)


def _save_current_project(*, as_new: bool = False) -> None:
    """Persist the in-session corpus + edits to data/projects/."""
    df = st.session_state.get("review_data")
    if df is None:
        st.warning(_label("project_need_data", "Load a dataset before saving a project."))
        return
    name = str(st.session_state.get("project_name_input") or "").strip()
    if not name:
        name = str(st.session_state.get("current_project_name") or "").strip()
    if not name:
        name = str(st.session_state.get("data_source") or "Untitled project").strip()
    if not name:
        st.warning(_label("project_need_name", "Enter a project name."))
        return

    pid = None if as_new else st.session_state.get("current_project_id")
    brief_ov = st.session_state.get("brief_overrides") or {}
    study_title = brief_ov.get("study_title") if isinstance(brief_ov, dict) else None
    meta = save_project(
        name=name,
        review_df=df,
        source_label=str(st.session_state.get("data_source") or ""),
        behavior_df=st.session_state.get("behavior_data"),
        brief_overrides=brief_ov if isinstance(brief_ov, dict) else {},
        roadmap_overrides=st.session_state.get("roadmap_overrides") or {},
        study_title=study_title,
        project_id=pid,
    )
    st.session_state["current_project_id"] = meta["id"]
    st.session_state["current_project_name"] = meta["name"]
    st.success(_label("project_saved", "Project saved."))


def _open_project(project_id: str) -> None:
    """Load a project from disk into session and open Analysis."""
    payload = load_project(project_id)
    meta = payload["meta"]
    st.session_state["review_data"] = payload["review_df"]
    st.session_state["data_source"] = meta.get("source_label") or meta.get("name") or project_id
    if payload.get("behavior_df") is not None:
        st.session_state["behavior_data"] = payload["behavior_df"]
    else:
        st.session_state.pop("behavior_data", None)
    st.session_state["brief_overrides"] = payload.get("brief_overrides") or {}
    st.session_state["roadmap_overrides"] = payload.get("roadmap_overrides") or {}
    st.session_state["brief_edit_nonce"] = int(st.session_state.get("brief_edit_nonce") or 0) + 1
    st.session_state["roadmap_edit_nonce"] = int(st.session_state.get("roadmap_edit_nonce") or 0) + 1
    st.session_state["current_project_id"] = meta.get("id") or project_id
    st.session_state["current_project_name"] = meta.get("name") or project_id
    st.session_state["project_name_input"] = meta.get("name") or project_id
    st.session_state["sidebar_panel"] = None
    st.session_state["nav_page"] = NAV_ANALYSIS
    st.success(_label("project_opened", "Project opened."))
    st.rerun()


def _start_new_project() -> None:
    """Clear session and return Home for a fresh project shell."""
    _clear_analysis_session(clear_strand_snapshots=False)
    st.session_state.pop("project_name_input", None)
    st.session_state["nav_page"] = NAV_HOME
    st.session_state["sidebar_panel"] = "workspace"
    st.rerun()


def _render_project_controls(*, has_data: bool) -> None:
    """New / Open / Save controls for the local research workspace."""
    current_name = st.session_state.get("current_project_name")
    if current_name:
        st.caption(f"{_label('project_current', 'Current project')}: {current_name}")
    else:
        st.caption(_label("project_none", "No project saved yet"))
    st.caption(
        _label(
            "workspace_caption",
            "Save and reopen this analysis as a local project.",
        )
    )
    default_name = str(
        st.session_state.get("current_project_name")
        or st.session_state.get("data_source")
        or ""
    )
    if "project_name_input" not in st.session_state:
        st.session_state["project_name_input"] = default_name
    st.text_input(
        _label("project_name", "Project name"),
        key="project_name_input",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            _label("project_save", "Save project"),
            use_container_width=True,
            key="project_save_btn",
            disabled=not has_data,
        ):
            _save_current_project(as_new=False)
            st.rerun()
    with c2:
        if st.button(
            _label("project_save_as", "Save as new"),
            use_container_width=True,
            key="project_save_as_btn",
            disabled=not has_data,
        ):
            _save_current_project(as_new=True)
            st.rerun()

    if st.button(
        _label("project_new", "New project"),
        use_container_width=True,
        key="project_new_btn",
        help=_label(
            "project_new_hint",
            "Clears the current session and starts a fresh project shell.",
        ),
    ):
        _start_new_project()

    projects = list_projects()
    if projects:
        labels = {
            p["id"]: f"{p.get('name', p['id'])} · {str(p.get('updated_at', ''))[:16]}"
            for p in projects
        }
        pick = st.selectbox(
            _label("project_select", "Saved projects"),
            options=list(labels.keys()),
            format_func=lambda pid: labels.get(pid, pid),
            key="project_open_select",
        )
        if st.button(
            _label("project_open_btn", "Open selected"),
            use_container_width=True,
            key="project_open_btn",
        ):
            _open_project(pick)
    else:
        st.caption(_label("project_open", "Open project") + " — —")


def _exit_current_analysis() -> None:
    """Exit the current analysis task and return to Home."""
    _clear_analysis_session(clear_strand_snapshots=False)
    st.session_state["nav_page"] = NAV_HOME
    st.rerun()


def _stash_strand_snapshot(df: pd.DataFrame, source_label: str, *, strand: str | None = None) -> None:
    """Remember a lightweight quant/qual snapshot for mixed-methods compare."""
    key = strand or detect_strand_key(source_label)
    if key not in ("quant", "qual"):
        return
    snap = build_strand_snapshot(df, source_label=source_label, strand=key)  # type: ignore[arg-type]
    store = dict(st.session_state.get("strand_snapshots") or {})
    store[key] = snap
    st.session_state["strand_snapshots"] = store


def _render_main_back_button() -> None:
    """Back control in the main content area (not sidebar site nav)."""
    can_go_back = bool(st.session_state.get("nav_history")) or (
        st.session_state.get("nav_page") not in (None, NAV_HOME)
    )
    if st.button(
        _material_label("arrow_back", _label("nav_back", "Back")),
        key="main_btn_back",
        disabled=not can_go_back,
        help=_label("nav_back_hint", "Go to the previous page"),
    ):
        _go_back()


def _render_analysis_exit_button() -> None:
    """Exit ends the current analysis task (clears data), unlike Back."""
    if st.button(
        _material_label("logout", _label("nav_exit", "Exit")),
        key="main_btn_exit",
        help=_label(
            "nav_exit_hint",
            "Exit this analysis task and clear the loaded dataset",
        ),
    ):
        _exit_current_analysis()


def render_app_sidebar(*, has_data: bool) -> str:
    """
    Compact icon rail — Home / Case / (Analysis) / Workspace / Settings.
    """
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand" title="InsightOptima">IO</div>
            """,
            unsafe_allow_html=True,
        )

        current = _default_nav_page(has_data)
        st.session_state["nav_page"] = current
        panel = st.session_state.get("sidebar_panel")  # "workspace" | "settings" | None

        def _toggle_panel(name: str) -> None:
            cur = st.session_state.get("sidebar_panel")
            st.session_state["sidebar_panel"] = None if cur == name else name
            st.rerun()

        if st.button(
            _material_icon("home"),
            use_container_width=True,
            type="primary" if current == NAV_HOME and panel is None else "secondary",
            key="nav_btn_home",
            help=_label("nav_home", "Home"),
        ):
            st.session_state["sidebar_panel"] = None
            _go_to_page(NAV_HOME)

        if st.button(
            _material_icon("science"),
            use_container_width=True,
            type="primary" if current == NAV_CASE and panel is None else "secondary",
            key="nav_btn_case",
            help=_label("nav_case", "Case Study"),
        ):
            st.session_state["sidebar_panel"] = None
            _go_to_page(NAV_CASE)

        if has_data:
            if st.button(
                _material_icon("analytics"),
                use_container_width=True,
                type="primary" if current == NAV_ANALYSIS and panel is None else "secondary",
                key="nav_btn_analysis",
                help=_label("nav_analysis", "Analysis"),
            ):
                st.session_state["sidebar_panel"] = None
                _go_to_page(NAV_ANALYSIS)

        if st.button(
            _material_icon("folder_open"),
            use_container_width=True,
            type="primary" if panel == "workspace" else "secondary",
            key="nav_btn_workspace",
            help=_label("nav_workspace", "Workspace"),
        ):
            _toggle_panel("workspace")

        if st.button(
            _material_icon("settings"),
            use_container_width=True,
            type="primary" if panel == "settings" else "secondary",
            key="nav_btn_settings",
            help=_label("nav_section_settings", "Settings"),
        ):
            _toggle_panel("settings")

        if has_data:
            if st.button(
                _material_icon("logout"),
                use_container_width=True,
                type="secondary",
                key="change_source_btn",
                help=_label("sidebar_change_source", "Exit analysis / change feedback source"),
            ):
                st.session_state["sidebar_panel"] = None
                _exit_current_analysis()

        if panel == "workspace":
            st.markdown(
                f'<div class="sidebar-panel-open"><div class="sidebar-panel-title">'
                f'{_label("nav_workspace", "Workspace")}</div></div>',
                unsafe_allow_html=True,
            )
            _render_project_controls(has_data=has_data)

        elif panel == "settings":
            st.markdown(
                f'<div class="sidebar-panel-open"><div class="sidebar-panel-title">'
                f'{_label("nav_section_settings", "Settings")}</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(_label("ui_language", "Interface Language"))
            render_language_selector(in_sidebar=False)

    return _default_nav_page(has_data)


def render_dashboard(df: pd.DataFrame) -> None:
    """Route to the sidebar-driven product workspace."""
    render_workspace(df)


def _top_complaints(df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    neg = df[df["is_negative"]] if "is_negative" in df.columns else df
    if neg.empty or "topic" not in neg.columns:
        return pd.DataFrame(columns=["topic", "count", "avg_rage"])
    g = (
        neg.groupby("topic")
        .agg(count=("review_id", "count"), avg_rage=("rage_index", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(n)
    )
    g["avg_rage"] = g["avg_rage"].round(1)
    return g


def render_research_rigor_strip(df: pd.DataFrame) -> None:
    """Always-visible sample size and anti-overclaim caveat."""
    total = len(df)
    neg_n = int(df["is_negative"].sum()) if "is_negative" in df.columns else 0
    themes = int(df["topic"].nunique()) if "topic" in df.columns else 0
    st.markdown(
        f"""
        <div class="rigor-strip">
            <span class="pill">n = {total:,}</span>
            <span class="pill">{neg_n:,} negative</span>
            <span class="pill">{themes} themes</span>
            <div class="caveat">{_label("rigor_caveat", "Draft for discussion — not causal proof. Validate with evidence before socializing.")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_brief(df: pd.DataFrame, roadmap_df: pd.DataFrame, *, show_rigor: bool = True) -> None:
    """Stakeholder brief: background → findings → sources → method → next steps."""
    if show_rigor:
        render_research_rigor_strip(df)
    st.caption(
        _label(
            "brief_caption",
            "Research brief order: background → findings → sources → method → recommended next steps.",
        )
    )
    st.caption(
        _label(
            "brief_finalize_hint",
            "Edit findings below. Save actions are at the bottom with Download — nothing auto-saves.",
        )
    )

    source_label = str(st.session_state.get("data_source", "Product"))
    default_title = _label(
        "brief_default_title",
        "Public health feedback synthesis — priority draft with evidence",
    )
    overrides = st.session_state.get("brief_overrides") or {}
    saved_by = dict(overrides.get("by_theme") or {})
    has_saved = bool(overrides)

    base_brief = build_research_brief(
        df,
        roadmap_df,
        study_title=str(overrides.get("study_title") or default_title),
        product_context=source_label,
    )
    brief = apply_brief_overrides(base_brief, overrides)
    bg = brief.get("background") or {}
    # Always edit against base quotes so indices stay stable after Apply
    base_sources_by_theme = {str(s.get("theme")): s for s in (base_brief.get("sources") or [])}

    edit_nonce = int(st.session_state.get("brief_edit_nonce") or 0)

    # 1. Background (editable)
    st.markdown(f"### {_label('brief_bg_title', '1. Research background')}")
    study_title_draft = st.text_input(
        _label("brief_study_title", "Study title"),
        value=str(brief.get("study_title") or default_title),
        key=f"brief_study_title_{edit_nonce}",
    )
    background_draft = st.text_area(
        _label("brief_background_edit", "Background summary"),
        value=str(bg.get("summary") or source_label),
        key=f"brief_bg_summary_{edit_nonce}",
        height=100,
    )
    st.markdown(f"**{_label('brief_objectives_title', 'Objectives')}**")
    for obj in bg.get("objectives") or []:
        st.markdown(f"- {obj}")
    st.markdown(f"**{_label('brief_rq_title', 'Research questions')}**")
    for i, q in enumerate(bg.get("research_questions") or brief.get("research_questions") or [], 1):
        st.markdown(f"{i}. {q}")

    # 2–3. Findings + sources with finalize controls
    st.markdown(f"### {_label('brief_findings_title', '2. Findings')} / {_label('brief_sources_title', '3. Sources')}")
    if not base_brief["findings"]:
        st.info(_label("no_complaints", "No negative themes found yet."))

    draft_by_theme: dict[str, dict] = {}
    status_options = ["keep", "needs_followup", "drop"]
    status_labels = {
        "keep": _label("brief_status_keep", "Keep"),
        "needs_followup": _label("brief_status_followup", "Needs follow-up"),
        "drop": _label("brief_status_drop", "Drop"),
    }

    for i, finding in enumerate(base_brief["findings"], 1):
        theme = str(finding["theme"])
        ov = saved_by.get(theme) or {}
        status = str(ov.get("status") or "keep")
        src = base_sources_by_theme.get(theme) or {"quotes": []}
        quotes = list(src.get("quotes") or [])
        claim_default = str(ov.get("claim") or finding.get("claim") or "")

        with st.expander(
            f"{i}. {theme} · n={finding['n']} · {status}",
            expanded=(i == 1),
        ):
            st.caption(
                f"n = {finding['n']} · rage {finding['avg_rage']} · priority {finding['priority']}"
            )
            new_status = st.selectbox(
                _label("brief_status_label", "Finding status"),
                options=status_options,
                index=status_options.index(status) if status in status_options else 0,
                format_func=lambda x: status_labels.get(x, x),
                key=f"brief_status_{edit_nonce}_{i}_{theme[:40]}",
            )
            new_claim = st.text_area(
                _label("brief_claim_label", "Claim"),
                value=claim_default,
                key=f"brief_claim_{edit_nonce}_{i}_{theme[:40]}",
                height=80,
            )
            kept_idxs: list[int] = []
            if quotes:
                st.markdown(f"**{_label('brief_quotes_label', 'Keep quotes')}**")
                default_kept = ov.get("kept_quote_idxs")
                if default_kept is None:
                    default_kept = list(range(len(quotes)))
                for qi, q in enumerate(quotes):
                    checked = qi in default_kept
                    if st.checkbox(
                        f"“{str(q.get('text', ''))[:120]}…”",
                        value=checked,
                        key=f"brief_q_{edit_nonce}_{i}_{qi}_{theme[:30]}",
                    ):
                        kept_idxs.append(qi)
            else:
                st.caption(_label("brief_no_quotes", "No matching quotes retrieved."))
                kept_idxs = []

            draft_by_theme[theme] = {
                "claim": new_claim,
                "status": new_status,
                "kept_quote_idxs": kept_idxs,
            }

    def _canonical_theme(by_theme: dict) -> dict:
        out: dict[str, dict] = {}
        for theme, v in (by_theme or {}).items():
            idxs = [int(i) for i in (v.get("kept_quote_idxs") or [])]
            out[str(theme)] = {
                "claim": str(v.get("claim") or "").strip(),
                "status": str(v.get("status") or "keep"),
                "kept_quote_idxs": idxs,
            }
        return out

    def _baseline_theme() -> dict:
        baseline: dict[str, dict] = {}
        for finding in base_brief.get("findings") or []:
            theme = str(finding["theme"])
            if theme in saved_by:
                baseline[theme] = saved_by[theme]
                continue
            n_quotes = len((base_sources_by_theme.get(theme) or {}).get("quotes") or [])
            baseline[theme] = {
                "claim": str(finding.get("claim") or ""),
                "status": "keep",
                "kept_quote_idxs": list(range(n_quotes)),
            }
        return baseline

    def _lines(text: str) -> list[str]:
        return [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]

    # 4–5. Method + next (editable limitations / next steps)
    m = brief["method"]
    with st.expander(_label("brief_method_title", "4. Research method"), expanded=True):
        st.markdown(
            f"- {_label('brief_sample_line', 'Sample')}: **n = {m['n_total']:,}** "
            f"({m['n_negative']:,} negative · {m['negative_pct']}%)  \n"
            f"- {m['approach']}"
        )
        limits_default = "\n".join(brief.get("limitations") or [])
        limits_draft = st.text_area(
            _label("brief_limits_edit", "Limitations (one per line)"),
            value=limits_default,
            key=f"brief_limits_{edit_nonce}",
            height=120,
        )
        next_default = "\n".join(brief.get("next_research") or [])
        next_draft = st.text_area(
            _label("brief_next_edit", "Next research steps (one per line)"),
            value=next_default,
            key=f"brief_next_{edit_nonce}",
            height=100,
        )

    draft_bundle = {
        "study_title": str(study_title_draft or "").strip(),
        "background_summary": str(background_draft or "").strip(),
        "limitations": _lines(limits_draft),
        "next_research": _lines(next_draft),
        "by_theme": draft_by_theme,
    }
    baseline_bundle = {
        "study_title": str(
            overrides.get("study_title") or brief.get("study_title") or default_title
        ).strip(),
        "background_summary": str(
            overrides.get("background_summary")
            or (base_brief.get("background") or {}).get("summary")
            or ""
        ).strip(),
        "limitations": list(
            overrides.get("limitations")
            if isinstance(overrides.get("limitations"), list)
            else (base_brief.get("limitations") or [])
        ),
        "next_research": list(
            overrides.get("next_research")
            if isinstance(overrides.get("next_research"), list)
            else (base_brief.get("next_research") or [])
        ),
        "by_theme": _baseline_theme(),
    }
    dirty = (
        draft_bundle["study_title"] != baseline_bundle["study_title"]
        or draft_bundle["background_summary"] != baseline_bundle["background_summary"]
        or draft_bundle["limitations"]
        != [str(x).strip() for x in baseline_bundle["limitations"] if str(x).strip()]
        or draft_bundle["next_research"]
        != [str(x).strip() for x in baseline_bundle["next_research"] if str(x).strip()]
        or _canonical_theme(draft_by_theme) != _canonical_theme(baseline_bundle["by_theme"])
    )

    # Finalize bar — sits with download at the bottom of the brief
    st.markdown(f"### {_label('brief_finalize_title', 'Finalize & export')}")
    st.caption(
        _label(
            "brief_finalize_bar_hint",
            "Not auto-saved. Apply locks edits into the download; Discard cancels unsaved form changes.",
        )
    )

    if has_saved and not dirty:
        st.success(
            _label(
                "brief_status_saved",
                "Saved — download uses your applied edits.",
            )
        )
    elif dirty:
        st.warning(
            _label(
                "brief_status_unsaved",
                "Unsaved draft — Apply to keep, or Discard to revert.",
            )
        )
    else:
        st.caption(
            _label(
                "brief_status_clean",
                "No custom judgment yet — download uses the auto-generated brief.",
            )
        )

    b1, b2, b3 = st.columns(3)
    with b1:
        apply_clicked = st.button(
            _label("brief_apply_edits", "Apply brief edits"),
            type="primary",
            use_container_width=True,
            key="brief_apply_btn",
            disabled=not dirty and has_saved,
        )
    with b2:
        discard_clicked = st.button(
            _label("brief_discard_edits", "Discard changes"),
            use_container_width=True,
            key="brief_discard_btn",
            disabled=not dirty,
            help=_label(
                "brief_discard_help",
                "Reload the form from the last Apply (or from auto-generated if you never applied).",
            ),
        )
    with b3:
        reset_clicked = st.button(
            _label("brief_reset_auto", "Reset to auto"),
            use_container_width=True,
            key="brief_reset_btn",
            disabled=not has_saved and not dirty,
            help=_label(
                "brief_reset_help",
                "Clear all applied judgment and restore generated claims / all quotes.",
            ),
        )

    if apply_clicked:
        st.session_state["brief_overrides"] = {
            "study_title": draft_bundle["study_title"],
            "background_summary": draft_bundle["background_summary"],
            "limitations": draft_bundle["limitations"],
            "next_research": draft_bundle["next_research"],
            "by_theme": draft_by_theme,
        }
        st.session_state["brief_edit_nonce"] = edit_nonce + 1
        st.success(_label("brief_edits_saved", "Applied. Download now uses this judgment."))
        st.rerun()
    if discard_clicked:
        st.session_state["brief_edit_nonce"] = edit_nonce + 1
        st.rerun()
    if reset_clicked:
        st.session_state.pop("brief_overrides", None)
        st.session_state["brief_edit_nonce"] = edit_nonce + 1
        st.rerun()

    finalized = apply_brief_overrides(base_brief, st.session_state.get("brief_overrides"))
    md = research_brief_to_markdown(finalized)
    if dirty:
        st.caption(
            _label(
                "brief_download_note_draft",
                "Download still uses the last Apply (or auto) until you Apply this draft.",
            )
        )
    else:
        st.caption(
            _label(
                "brief_download_note_saved",
                "Download matches the current applied brief.",
            )
        )

    st.download_button(
        label=_label("brief_download", "Download Research Brief (.md)"),
        type="primary" if has_saved and not dirty else "secondary",
        data=md.encode("utf-8"),
        file_name="InsightOptima_Research_Brief.md",
        mime="text/markdown",
        use_container_width=True,
        key="brief_dl_btn",
    )



def render_page_feedback(df: pd.DataFrame, kpis: dict, *, show_rigor: bool = True) -> None:
    if show_rigor:
        render_research_rigor_strip(df)
    st.caption(_label("tab_feedback_caption", "Auto-sorted top complaints from your connected source."))
    c1, c2, c3 = st.columns(3)
    c1.metric(t("metric_total"), f"{kpis['total_reviews']:,}")
    c2.metric(t("metric_negative_rate"), f"{kpis['negative_pct']}%")
    c3.metric(t("metric_avg_rage"), f"{kpis['avg_rage']}")

    st.markdown(f"#### {_label('top_complaints_title', 'Top complaints')}")
    complaints = _top_complaints(df)
    if complaints.empty:
        st.info(_label("no_complaints", "No negative themes found yet."))
    else:
        for _, row in complaints.iterrows():
            st.markdown(
                f"""
                <div class="complaint-row">
                    <div class="name">{row['topic']}</div>
                    <div class="meta">{int(row['count'])} mentions · rage {row['avg_rage']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_page_themes(df: pd.DataFrame) -> None:
    st.markdown(f"#### {_label('theme_edit_title', 'Theme coding edits')}")
    st.caption(
        _label(
            "theme_edit_caption",
            "Rename or merge themes — writes back to the corpus so roadmap and brief recompute.",
        )
    )
    theme_stats = list_theme_stats(df, negative_only=True, n=20)
    if theme_stats.empty:
        st.info(_label("no_complaints", "No negative themes found yet."))
    else:
        draft_n = sum(
            1 for tname in theme_stats["topic"].astype(str).tolist() if topic_label_needs_review(tname)
        )
        if draft_n:
            st.info(
                _label(
                    "theme_draft_hint",
                    "Auto theme names are drafts ({n} look incomplete or ironic). "
                    "Rename/merge into human pain labels before sharing the brief.",
                ).replace("{n}", str(draft_n))
            )
        st.dataframe(theme_stats, use_container_width=True, hide_index=True, height=220)
        theme_names = theme_stats["topic"].astype(str).tolist()

        c_ren1, c_ren2 = st.columns(2)
        with c_ren1:
            rename_from = st.selectbox(
                _label("theme_rename_from", "Rename from"),
                options=theme_names,
                key="theme_rename_from",
            )
        with c_ren2:
            rename_to = st.text_input(
                _label("theme_rename_to", "New name"),
                value=rename_from,
                key="theme_rename_to",
            )
        if st.button(_label("theme_rename_btn", "Apply rename"), key="theme_rename_btn"):
            updated = rename_topic(df, rename_from, rename_to)
            st.session_state["review_data"] = updated
            st.session_state.pop("brief_overrides", None)
            st.session_state.pop("roadmap_overrides", None)
            st.success(_label("theme_edit_saved", "Theme updated."))
            st.rerun()

        merge_sources = st.multiselect(
            _label("theme_merge_sources", "Merge these themes"),
            options=theme_names,
            key="theme_merge_sources",
        )
        merge_into = st.text_input(
            _label("theme_merge_into", "Into label"),
            value=merge_sources[0] if merge_sources else "",
            key="theme_merge_into",
        )
        if st.button(_label("theme_merge_btn", "Apply merge"), key="theme_merge_btn"):
            if len(merge_sources) < 2 or not str(merge_into).strip():
                st.warning(
                    _label("theme_merge_need", "Select at least two themes and a target label.")
                )
            else:
                updated = merge_topics(df, merge_sources, merge_into)
                st.session_state["review_data"] = updated
                st.session_state.pop("brief_overrides", None)
                st.session_state.pop("roadmap_overrides", None)
                st.success(_label("theme_edit_saved", "Theme updated."))
                st.rerun()

    st.markdown(f"#### {_label('matrix_short_title', 'Rage × volume map')}")
    st.markdown(
        f"""
        <div class="quadrant-legend">
            <span class="quadrant-pill pill-red">{t("quad_red")}</span>
            <span class="quadrant-pill pill-purple">{t("quad_purple")}</span>
            <span class="quadrant-pill pill-yellow">{t("quad_yellow")}</span>
            <span class="quadrant-pill pill-green">{t("quad_green")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    matrix_df = compute_rage_volume_matrix(df)
    st.plotly_chart(create_rage_volume_scatter(matrix_df), use_container_width=True)


def render_page_priority(df: pd.DataFrame, roadmap_df: pd.DataFrame, *, show_rigor: bool = True) -> None:
    if show_rigor:
        render_research_rigor_strip(df)
    st.caption(t("module3_caption"))
    st.caption(
        _label(
            "roadmap_edit_caption",
            "Edit priority and research notes — Apply keeps them for this session and for Save project.",
        )
    )

    edit_nonce = int(st.session_state.get("roadmap_edit_nonce") or 0)
    display_cols = [
        c
        for c in (
            "Priority",
            "Core Pain Point",
            "AI Diagnosis",
            "Recommended Action",
            "Priority Score",
            "Mentions",
            "Avg Rage",
            "Supporting Signal",
        )
        if c in roadmap_df.columns
    ]
    editor_df = roadmap_df[display_cols].copy() if display_cols else roadmap_df.copy()
    # Friendly headers in the editor only
    rename_map = {
        "AI Diagnosis": _label("evidence_diagnosis", "Pattern note"),
        "Recommended Action": _label("evidence_action", "Next research note"),
    }
    editor_view = editor_df.rename(columns=rename_map)
    pattern_col = rename_map["AI Diagnosis"]
    action_col = rename_map["Recommended Action"]

    column_config = {
        "Priority": st.column_config.SelectboxColumn(
            "Priority",
            options=["P0", "P1", "P2"],
            required=True,
        ),
        "Core Pain Point": st.column_config.TextColumn("Core Pain Point", disabled=True),
        pattern_col: st.column_config.TextColumn(pattern_col, width="large"),
        action_col: st.column_config.TextColumn(action_col, width="large"),
    }
    if "Priority Score" in editor_view.columns:
        column_config["Priority Score"] = st.column_config.NumberColumn(
            "Priority Score", disabled=True
        )
    if "Mentions" in editor_view.columns:
        column_config["Mentions"] = st.column_config.NumberColumn("Mentions", disabled=True)
    if "Avg Rage" in editor_view.columns:
        column_config["Avg Rage"] = st.column_config.NumberColumn("Avg Rage", disabled=True)
    if "Supporting Signal" in editor_view.columns:
        column_config["Supporting Signal"] = st.column_config.TextColumn(
            "Supporting Signal", disabled=True
        )

    edited = st.data_editor(
        editor_view,
        use_container_width=True,
        hide_index=True,
        height=min(50 + max(len(editor_view), 1) * 38, 520),
        column_config=column_config,
        key=f"roadmap_editor_{edit_nonce}",
        disabled=["Core Pain Point"]
        + [c for c in ("Priority Score", "Mentions", "Avg Rage", "Supporting Signal") if c in editor_view.columns],
    )

    if st.button(
        _label("roadmap_apply_edits", "Apply roadmap edits"),
        type="primary",
        key=f"roadmap_apply_{edit_nonce}",
    ):
        overrides: dict = {}
        for _, row in edited.iterrows():
            theme = str(row.get("Core Pain Point", "")).strip()
            if not theme:
                continue
            overrides[theme] = {
                "Priority": str(row.get("Priority", "P2")).strip().upper(),
                "diagnosis": str(row.get(pattern_col, "")).strip(),
                "action": str(row.get(action_col, "")).strip(),
            }
        st.session_state["roadmap_overrides"] = overrides
        st.session_state["roadmap_edit_nonce"] = edit_nonce + 1
        st.success(_label("roadmap_edits_saved", "Roadmap judgment applied."))
        st.rerun()

    if "Supporting Signal" in roadmap_df.columns and (roadmap_df["Supporting Signal"] != "review-only").any():
        st.caption(t("behavior_context_note"))
    p0_n = int((roadmap_df["Priority"] == "P0").sum()) if not roadmap_df.empty else 0
    st.info(t("roadmap_draft_note", p0=p0_n, total=len(roadmap_df)))



def render_workspace(df: pd.DataFrame) -> None:
    """Single analysis page — all modules as expandable sections."""
    source_label = st.session_state.get("data_source", t("source_unknown"))
    kpis = compute_summary_kpis(df)
    roadmap_df = build_implementation_roadmap(
        df,
        behavior_df=st.session_state.get("behavior_data"),
    )
    roadmap_df = apply_roadmap_overrides(
        roadmap_df,
        st.session_state.get("roadmap_overrides") or {},
    )

    chrome_l, title_col = st.columns([2, 5])
    with chrome_l:
        back_col, exit_col = st.columns(2)
        with back_col:
            _render_main_back_button()
        with exit_col:
            _render_analysis_exit_button()
    project_name = st.session_state.get("current_project_name")
    meta_bits = f"{t('dashboard_source', source=source_label)} · {kpis['total_reviews']:,} reviews"
    if project_name:
        meta_bits = f"{project_name} · {meta_bits}"
    with title_col:
        st.markdown(
            f"""
            <div class="workspace-soft">
                <div class="workspace-soft__mark">{_svg_workspace_mark()}</div>
                <div>
                    <div class="title">{_label("workspace_title", "Analysis")}</div>
                    <p class="meta">{meta_bits}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_research_rigor_strip(df)
    st.caption(
        _label(
            "analysis_page_caption",
            "Everything for this dataset is on one page — expand a section to dig in.",
        )
    )

    with st.expander(
        _material_label("description", _label("tab_brief", "Research Brief")),
        expanded=True,
    ):
        render_page_brief(df, roadmap_df, show_rigor=False)

    with st.expander(
        _material_label("forum", _label("tab_feedback", "Feedback overview")),
        expanded=False,
    ):
        render_page_feedback(df, kpis, show_rigor=False)

    with st.expander(
        _material_label("account_tree", _label("tab_themes", "Themes")),
        expanded=False,
    ):
        render_page_themes(df)

    if both_strands_ready(st.session_state.get("strand_snapshots")):
        with st.expander(
            _material_label("compare_arrows", _label("tab_mm_compare", "Mixed-methods compare")),
            expanded=False,
        ):
            render_mixed_methods_compare(show_reload=True)

    with st.expander(
        _material_label("low_priority", _label("tab_priority", "Priority draft")),
        expanded=False,
    ):
        render_page_priority(df, roadmap_df, show_rigor=False)

    with st.expander(
        _material_label("format_quote", _label("tab_evidence", "Evidence")),
        expanded=False,
    ):
        st.caption(t("evidence_caption"))
        render_evidence_chain(df, roadmap_df)

    with st.expander(
        _material_label("download", _label("tab_export", "Export")),
        expanded=False,
    ):
        render_export_section(df, roadmap_df)


def render_mixed_methods_compare(*, show_reload: bool = True) -> None:
    """Side-by-side quant vs qual strand snapshots (when both have been run)."""
    snaps = st.session_state.get("strand_snapshots") or {}
    if not both_strands_ready(snaps):
        st.caption(
            _label(
                "mm_compare_need_both",
                "Run both Case Study strands (quant + qual) to unlock this compare view.",
            )
        )
        return

    quant = snaps["quant"]
    qual = snaps["qual"]
    st.markdown(f"#### {_label('mm_compare_title', 'Mixed-methods compare')}")
    st.caption(
        _label(
            "mm_compare_caption",
            "Same research workflow, different evidence types. Snapshots are lightweight theme summaries — not two live corpora at once.",
        )
    )

    col_q, col_ql = st.columns(2, gap="large")
    for col, snap, title in (
        (col_q, quant, _label("mm_quant_label", "Quantitative strand")),
        (col_ql, qual, _label("mm_qual_label", "Qualitative strand")),
    ):
        with col:
            st.markdown(f"**{title}**")
            st.caption(snap.get("source_label", ""))
            st.markdown(
                f"- n = **{snap.get('n', 0):,}** ({snap.get('neg_n', 0):,} negative · {snap.get('neg_pct', 0)}%)  \n"
                f"- themes = **{snap.get('n_themes', 0)}**  \n"
                f"- {snap.get('evidence_type', '')}"
            )
            themes = snap.get("themes") or []
            if themes:
                rows = [
                    {"theme": t["theme"], "n": t["n"], "avg_rage": t.get("avg_rage", 0)}
                    for t in themes
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=240)
            quotes = snap.get("top_quotes") or []
            if quotes:
                with st.expander(_label("mm_sample_quotes", "Sample quotes"), expanded=False):
                    for q in quotes[:5]:
                        st.markdown(
                            f'<div class="quote-card"><strong>{q.get("theme", "")}</strong><br>“{q.get("text", "")}”</div>',
                            unsafe_allow_html=True,
                        )

    if show_reload:
        r1, r2 = st.columns(2)
        with r1:
            if st.button(
                _label("mm_reload_quant", "Load quant into Analysis"),
                use_container_width=True,
                key="mm_reload_quant",
            ):
                _load_case_into_workspace(
                    SAMPLE_CASE_STUDY_CSV,
                    source_label=_label(
                        "case_loaded_quant",
                        "Public quant case — UCI Drugs.com patient reviews (CC BY 4.0)",
                    ),
                    missing_hint=_label(
                        "case_not_found_quant",
                        "Quant case file missing. Run: python scripts/prepare_case_study_dataset.py",
                    ),
                    strand="quant",
                )
        with r2:
            if st.button(
                _label("mm_reload_qual", "Load qual into Analysis"),
                use_container_width=True,
                key="mm_reload_qual",
            ):
                _load_case_into_workspace(
                    SAMPLE_CASE_STUDY_QUAL_CSV,
                    source_label=_label(
                        "case_loaded_qual",
                        "Public qual case — Zenodo PubPeer open-ended answers (CC BY 4.0)",
                    ),
                    missing_hint=_label(
                        "case_not_found_qual",
                        "Qual case file missing. Run: python scripts/prepare_case_study_qual_dataset.py",
                    ),
                    strand="qual",
                )


def _load_case_into_workspace(
    csv_path: Path,
    source_label: str,
    missing_hint: str,
    *,
    strand: str | None = None,
) -> None:
    if not csv_path.exists():
        st.error(missing_hint)
        return
    try:
        raw_df = load_reviews_file(csv_path)
        encoded = run_analysis_pipeline(raw_df, source_label=source_label)
        st.session_state.pop("brief_overrides", None)
        _stash_strand_snapshot(encoded, source_label, strand=strand)
        st.session_state["nav_page"] = NAV_ANALYSIS
        st.rerun()
    except Exception as exc:
        st.error(t("error_sample_load", error=exc))


def render_case_study_page() -> None:
    """Dedicated Case Study site page — mixed-methods pair (quant + qual), public data only."""
    _render_main_back_button()
    st.markdown(
        f"""
        <div class="case-soft-banner">
            <div>
                <div class="eyebrow">{_label("case_section_eyebrow", "Portfolio case studies")}</div>
                <h2>{_label("case_mm_title", "Mixed methods: one quant strand, one qual strand")}</h2>
                <p class="body">{_label("case_mm_body", "Two public corpora, one research workflow. The quantitative case uses large-n rated patient reviews; the qualitative case uses open-ended survey verbatims with a published coding scheme. Both are citable (DOI + CC BY 4.0) — no synthetic or private data.")}</p>
            </div>
            <div class="case-soft-banner__art">{_svg_case_accent()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_q, col_ql = st.columns(2, gap="large")

    with col_q:
        st.markdown(
            f"""
            <div class="case-pair-card">
                {_glyph_html("quant")}
                <h3>{_label("case_quant_title", "Quantitative — Drugs.com / UCI")}</h3>
                <p class="case-pair-body">{_label("case_quant_body", "1,500 public patient medication reviews with ratings. Themes + urgency from volume and severity signals.")}</p>
                <p class="case-pair-source">{_label("case_quant_source", "DOI 10.24432/C5SK5S · CC BY 4.0 · Cite Kallumadi &amp; Grasser (2018)")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**{_label('brief_rq_title', 'Research questions')}**")
        st.markdown(
            "1. Where do negative medication experiences concentrate?\n\n"
            "2. Which themes are highest urgency for follow-up research?\n\n"
            "3. What verbatim evidence supports each claim — and what must we not over-claim?"
        )
        if st.button(
            _label("mode_case_quant_btn", "Open quant case"),
            type="primary",
            use_container_width=True,
            key="load_case_quant_btn",
        ):
            _load_case_into_workspace(
                SAMPLE_CASE_STUDY_CSV,
                source_label=_label(
                    "case_loaded_quant",
                    "Public quant case — UCI Drugs.com patient reviews (CC BY 4.0)",
                ),
                missing_hint=_label(
                    "case_not_found_quant",
                    "Quant case file missing. Run: python scripts/prepare_case_study_dataset.py",
                ),
                strand="quant",
            )

    with col_ql:
        st.markdown(
            f"""
            <div class="case-pair-card">
                {_glyph_html("qual")}
                <h3>{_label("case_qual_title", "Qualitative — PubPeer open answers / Zenodo")}</h3>
                <p class="case-pair-body">{_label("case_qual_body", "1,115 open-ended responses on PubPeer / post-publication review, with a published coding scheme. Text-only sentiment — no star ratings.")}</p>
                <p class="case-pair-source">{_label("case_qual_source", "DOI 10.5281/zenodo.20413424 · CC BY 4.0 · Cite Hepkema &amp; Bordignon (2026)")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**{_label('brief_rq_title', 'Research questions')}**")
        st.markdown(
            "1. What perception themes emerge from open-ended responses about PubPeer?\n\n"
            "2. Which themes merit deeper qualitative follow-up (interviews / diary)?\n\n"
            "3. What verbatim evidence supports each theme — and how does that differ from coded categories in the published scheme?"
        )
        if st.button(
            _label("mode_case_qual_btn", "Open qual case"),
            type="primary",
            use_container_width=True,
            key="load_case_qual_btn",
        ):
            _load_case_into_workspace(
                SAMPLE_CASE_STUDY_QUAL_CSV,
                source_label=_label(
                    "case_loaded_qual",
                    "Public qual case — Zenodo PubPeer open-ended answers (CC BY 4.0)",
                ),
                missing_hint=_label(
                    "case_not_found_qual",
                    "Qual case file missing. Run: python scripts/prepare_case_study_qual_dataset.py",
                ),
                strand="qual",
            )

    st.caption(
        _label(
            "case_section_hint_mm",
            "Both demos load into the same analysis workspace. Rebuild scripts live under scripts/prepare_case_study*.py",
        )
    )

    st.divider()
    render_mixed_methods_compare(show_reload=True)


def render_empty_state() -> None:
    """Home — Case Study entry + other data sources."""
    st.markdown(
        f"""
        <div class="soft-hero">
            <div class="soft-hero__copy">
                <h1>{_label("empty_title", "InsightOptima")}</h1>
                <p class="tagline">{_label("empty_tagline", "A research workspace: discover friction themes, draft P0/P1/P2, and ship an evidence-backed brief.")}</p>
                <p class="cta-line">{_label("home_soft_cta", "Start with a public Case Study demo, or connect your own feedback below.")}</p>
            </div>
            <div class="soft-hero__art">{_svg_research_scene()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="home-case-entry">
            {_glyph_html("quant")}
            <div class="home-case-entry__copy">
                <h3>{_label("home_case_entry_title", "Portfolio Case Study")}</h3>
                <p>{_label("home_case_entry_body", "Mixed methods on public data: Drugs.com (quant) + PubPeer open answers (qual). DOI + CC BY 4.0.")}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(
        _material_label("science", _label("home_case_entry_btn", "Open Case Study")),
        type="primary",
        use_container_width=False,
        key="home_open_case_btn",
    ):
        _go_to_page(NAV_CASE)

    st.markdown(f"### {_label('connect_other_title', 'Connect a data source')}")
    st.caption(
        _label(
            "connect_caption",
            "Amazon sample or upload your own file.",
        )
    )

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        st.markdown(
            f"""
            <div class="source-card">
                {_glyph_html("sample")}
                <h3>{_label("source_sample_title", "Amazon sample")}</h3>
                <p>{_label("source_sample_body", "2,000 real Amazon Fine Food reviews — general-purpose sample corpus.")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        load_amazon = st.button(
            _label("mode_b_btn", "Load Amazon Sample"),
            use_container_width=True,
            key="load_sample_btn",
        )
    with col_b:
        st.markdown(
            f"""
            <div class="source-card">
                {_glyph_html("upload", "glyph--stone")}
                <h3>{_label("source_upload_title", "Upload file")}</h3>
                <p>{_label("source_upload_body", "CSV, Excel, TSV, JSON, or ZIP with a comments column.")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploader_version = int(st.session_state.get("file_uploader_version") or 0)
        uploaded = st.file_uploader(
            _label("choose_file", "Choose a data file"),
            type=["csv", "tsv", "xlsx", "xls", "json", "zip"],
            key=f"file_uploader_{uploader_version}",
            label_visibility="collapsed",
        )

    if load_amazon:
        sample_path = SAMPLE_DATA_XLSX if SAMPLE_DATA_XLSX.exists() else SAMPLE_DATA_CSV
        if not sample_path.exists():
            st.error(
                _label(
                    "amazon_not_found",
                    "Amazon sample not found. Run: python scripts/prepare_sample_dataset.py",
                )
            )
        else:
            try:
                raw_df = load_reviews_file(sample_path)
                run_analysis_pipeline(
                    raw_df,
                    source_label=_label(
                        "sample_loaded",
                        "Amazon Fine Food Reviews sample (public corpus subset)",
                    ),
                )
                st.session_state["nav_page"] = NAV_ANALYSIS
                st.rerun()
            except Exception as exc:
                st.error(t("error_sample_load", error=exc))

    if uploaded is not None:
        handle_file_upload(uploaded)


def render_data_input_section() -> pd.DataFrame | None:
    """Legacy helper — intake is handled by render_empty_state()."""
    return st.session_state.get("review_data")


def main() -> None:
    """Site shell: Home / Case Study nav; single expandable analysis page."""
    st.set_page_config(
        page_title=f"{t('hero_title')} | {t('hero_tagline')}",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    if is_rtl():
        st.markdown(
            '<style>.stApp, .dashboard-hero, .soft-hero, .case-soft-banner, .section-header { direction: rtl; text-align: right; }</style>',
            unsafe_allow_html=True,
        )

    df = st.session_state.get("review_data")
    has_data = df is not None and not getattr(df, "empty", True)
    page = render_app_sidebar(has_data=has_data)

    if page == NAV_CASE:
        render_case_study_page()
    elif page == NAV_ANALYSIS and has_data:
        render_workspace(df)
    elif page == NAV_HOME:
        if has_data:
            st.info(
                _label(
                    "home_has_data",
                    "A dataset is already loaded. Open **Analysis** in the left nav, or connect a new source below.",
                )
            )
        render_empty_state()
    elif has_data:
        render_workspace(df)
    else:
        render_empty_state()



def render_ask_insight_panel(df: pd.DataFrame) -> None:
    """Viable-inspired natural language Q&A."""
    st.markdown(f'<p class="section-header">{t("qa_title")}</p>', unsafe_allow_html=True)
    st.caption(t("qa_caption"))

    examples = [
        "What are the top complaints?",
        "Which lifecycle stage is riskiest?",
        "How bad is sentiment and rage?",
        "What do users say about packaging?",
    ]
    q = st.text_input(t("qa_input"), placeholder=examples[0], key="qa_box")
    st.caption(" · ".join(examples))

    if st.button(t("qa_ask"), type="primary", key="qa_ask_btn") and q.strip():
        st.session_state["qa_result"] = ask_insight(df, q)

    result = st.session_state.get("qa_result")
    if result:
        # Streamlit markdown is safer than raw HTML for user-derived answers
        st.markdown(result["answer"])
        for quote in result.get("quotes") or []:
            st.markdown(
                f"> **{quote.get('topic', '')}** · rage {quote.get('rage', '')}  \n"
                f"> {quote.get('text', '')}"
            )


def render_prd_panel(df: pd.DataFrame, roadmap_df: pd.DataFrame) -> None:
    """Draft PRD + user stories from discovered pain points and evidence."""
    st.markdown(f'<p class="section-header">{t("prd_title")}</p>', unsafe_allow_html=True)
    st.caption(t("prd_caption"))

    prd_md = generate_prd_markdown(
        df,
        roadmap_df,
        product_name=str(st.session_state.get("data_source", "Product")),
    )
    with st.expander(t("prd_preview"), expanded=False):
        st.markdown(prd_md)
    st.download_button(
        label=t("prd_download"),
        data=prd_md.encode("utf-8"),
        file_name="InsightOptima_PRD.md",
        mime="text/markdown",
        use_container_width=True,
        key="prd_dl_btn",
    )


def render_export_section(df: pd.DataFrame, roadmap_df: pd.DataFrame) -> None:
    """Render download buttons for CSV / Excel / HTML (print-to-PDF) reports."""
    st.markdown(
        f'<p class="section-header">📤 {t("export_title")}</p>',
        unsafe_allow_html=True,
    )
    st.caption(t("export_caption"))

    source_label = str(st.session_state.get("data_source", t("source_unknown")))
    try:
        bundle = build_report_bundle(df, source_label=source_label, roadmap_df=roadmap_df)
        basename = default_export_basename(source_label)
        csv_bytes = export_roadmap_csv(bundle)
        xlsx_bytes = export_workbook_xlsx(bundle)
        html_bytes = export_html_report(bundle)
    except Exception as exc:
        st.error(t("error_analysis_failed", error=exc))
        return

    st.success(t("export_ready"))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label=f"📄 {t('export_csv')}",
            data=csv_bytes,
            file_name=f"{basename}_roadmap.csv",
            mime="text/csv",
            use_container_width=True,
            key="export_csv_btn",
        )
    with c2:
        st.download_button(
            label=f"📊 {t('export_xlsx')}",
            data=xlsx_bytes,
            file_name=f"{basename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="export_xlsx_btn",
        )
    with c3:
        st.download_button(
            label=f"🖨️ {t('export_html')}",
            data=html_bytes,
            file_name=f"{basename}.html",
            mime="text/html",
            use_container_width=True,
            key="export_html_btn",
        )



# Future API Integration Skeleton
# ---------------------------------------------------------------------------
# The functions below are placeholders for production LLM pipeline integration.
# Wire these to OpenAI / Anthropic APIs when moving from demo to live deployment.
# ---------------------------------------------------------------------------


def semantic_coding_with_openai(reviews: list[str], model: str = "gpt-4o") -> list[dict[str, Any]]:
    """
    Placeholder: Send raw review texts to OpenAI for semantic encoding.

    Intended output per review:
    - lifecycle_stage classification
    - sentiment score (-1.0 to 1.0)
    - rage/frustration index (0-100)
    - extracted pain point topic label
    - confidence score

    Parameters
    ----------
    reviews : list[str]
        Raw user review texts.
    model : str
        OpenAI model identifier.

    Returns
    -------
    list[dict[str, Any]]
        List of encoded review dictionaries.

    TODO
    ----
    - Implement batch processing with rate-limit handling
    - Define structured output JSON schema via response_format
    - Add retry logic with exponential backoff
    - Cache encoded results to avoid re-processing
    """
    raise NotImplementedError("Connect to OpenAI API — semantic coding pipeline not yet implemented.")


def semantic_coding_with_anthropic(reviews: list[str], model: str = "claude-sonnet-4-20250514") -> list[dict[str, Any]]:
    """
    Placeholder: Send raw review texts to Anthropic Claude for semantic encoding.

    Parameters
    ----------
    reviews : list[str]
        Raw user review texts.
    model : str
        Anthropic model identifier.

    Returns
    -------
    list[dict[str, Any]]
        List of encoded review dictionaries.

    TODO
    ----
    - Implement tool-use or structured output for consistent JSON
    - Add prompt template versioning for reproducible coding
    - Support multi-language review encoding
    """
    raise NotImplementedError("Connect to Anthropic API — semantic coding pipeline not yet implemented.")


def calculate_retention_metrics(encoded_df: pd.DataFrame) -> dict[str, Any]:
    """
    Placeholder: Compute advanced retention metrics from semantically encoded data.

    Planned metrics:
    - Stage-weighted churn probability score
    - Topic co-occurrence network centrality
    - Sentiment trend velocity (week-over-week)
    - Retention driver vs. drop-out driver classification

    Parameters
    ----------
    encoded_df : pd.DataFrame
        DataFrame with LLM-encoded semantic fields.

    Returns
    -------
    dict[str, Any]
        Dictionary of computed retention metrics.

    TODO
    ----
    - Implement statistical significance testing on sentiment shifts
    - Add cohort-based retention correlation
    - Export metrics to JSON for downstream BI tools
    """
    raise NotImplementedError("Advanced metrics calculation pipeline not yet implemented.")


def generate_ai_roadmap_recommendations(encoded_df: pd.DataFrame, llm_provider: str = "openai") -> pd.DataFrame:
    """
    Placeholder: Use LLM to generate contextual implementation recommendations.

    Unlike rule-based roadmap (build_implementation_roadmap), this function
    will synthesize cross-topic patterns and propose novel interventions.

    Parameters
    ----------
    encoded_df : pd.DataFrame
        Semantically encoded review dataframe.
    llm_provider : str
        'openai' or 'anthropic'.

    Returns
    -------
    pd.DataFrame
        AI-generated roadmap with priority, diagnosis, action, and retention estimate.

    TODO
    ----
    - Feed aggregated pain point clusters as context to LLM
    - Include product domain metadata in system prompt
    - Validate recommendations against historical A/B test data
    """
    raise NotImplementedError("AI-powered roadmap generation not yet implemented.")


if __name__ == "__main__":
    main()
