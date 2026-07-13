"""
Reliable module loader for InsightOptima.

Loads service modules directly from file paths (same folder as app.py).
This avoids stale package cache and OneDrive sync issues with the
services/ package namespace.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parent
SERVICES_DIR = PROJECT_ROOT / "services"

# Bump this when service files change — shown in UI for sync verification
LOADER_VERSION = "1.12.0"


def _load_module_from_file(module_name: str, filename: str) -> ModuleType:
    """Load a Python module from an absolute file path."""
    filepath = SERVICES_DIR / filename
    if not filepath.exists():
        raise ImportError(
            f"Module file not found: {filepath}\n"
            f"Ensure OneDrive has finished syncing the InsightOptima folder."
        )

    full_name = f"insightoptima.{module_name}"
    spec = importlib.util.spec_from_file_location(full_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for: {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def verify_service_files() -> list[str]:
    """Verify all required service files exist and contain expected symbols."""
    required: dict[str, list[str]] = {
        "data_loader.py": [
            "auto_resolve_columns",
            "read_raw_file",
            "normalize_raw_reviews",
            "normalize_rating_to_5",
        ],
        "preflight.py": ["PreflightReport", "run_preflight"],
        "semantic_coder.py": ["encode_reviews"],
        "topic_modeling.py": ["extract_topic_labels", "topic_label_needs_review"],
        "topic_merge.py": ["apply_topic_merge", "merge_topic_labels"],
        "credibility.py": ["compute_credibility_report"],
        "behavior_data.py": ["normalize_behavior_df", "enrich_roadmap_with_behavior"],
        "analysis_cache.py": ["corpus_fingerprint", "load_cached_encoding"],
        "analysis_history.py": ["save_analysis_snapshot", "compare_snapshots"],
        "multilingual_nlp.py": ["detect_language", "compute_sentiment_multilingual"],
        "i18n.py": ["t", "render_language_selector"],
        "roadmap_generator.py": ["build_implementation_roadmap", "apply_roadmap_overrides"],
        "evidence_chain.py": ["fetch_topic_evidence", "summarize_topic_evidence"],
        "report_export.py": ["build_report_bundle", "export_html_report"],
        "revenue_at_risk.py": ["compute_revenue_at_risk"],
        "prd_generator.py": ["generate_prd_markdown", "generate_user_stories"],
        "insight_qa.py": ["ask_insight"],
        "research_brief.py": [
            "build_research_brief",
            "research_brief_to_markdown",
            "apply_brief_overrides",
        ],
        "theme_edits.py": ["rename_topic", "merge_topics", "list_theme_stats"],
        "strand_compare.py": [
            "build_strand_snapshot",
            "detect_strand_key",
            "both_strands_ready",
        ],
        "project_store.py": ["list_projects", "save_project", "load_project"],
    }

    errors: list[str] = []
    for filename, symbols in required.items():
        filepath = SERVICES_DIR / filename
        if not filepath.exists():
            errors.append(f"Missing file: {filepath}")
            continue
        content = filepath.read_text(encoding="utf-8")
        for symbol in symbols:
            if f"def {symbol}" not in content and f"class {symbol}" not in content:
                errors.append(f"{filename} is outdated — missing '{symbol}'. Wait for OneDrive sync.")
    return errors


_verify_errors = verify_service_files()
if _verify_errors:
    raise ImportError(
        "InsightOptima service files are incomplete or outdated:\n"
        + "\n".join(f"  • {e}" for e in _verify_errors)
        + "\n\nFix: wait for OneDrive sync, then restart the app."
    )

_dl = _load_module_from_file("data_loader", "data_loader.py")
_pf = _load_module_from_file("preflight", "preflight.py")
_tm = _load_module_from_file("topic_modeling", "topic_modeling.py")
_ml = _load_module_from_file("multilingual_nlp", "multilingual_nlp.py")
_tmerge = _load_module_from_file("topic_merge", "topic_merge.py")
_cred = _load_module_from_file("credibility", "credibility.py")
_cache = _load_module_from_file("analysis_cache", "analysis_cache.py")
_beh = _load_module_from_file("behavior_data", "behavior_data.py")
_hist = _load_module_from_file("analysis_history", "analysis_history.py")

_services_pkg = ModuleType("services")
_services_pkg.__path__ = [str(SERVICES_DIR)]  # type: ignore[attr-defined]
sys.modules["services"] = _services_pkg
sys.modules["services.topic_modeling"] = _tm
sys.modules["services.multilingual_nlp"] = _ml
sys.modules["services.topic_merge"] = _tmerge
sys.modules["services.credibility"] = _cred
sys.modules["services.analysis_cache"] = _cache
sys.modules["services.behavior_data"] = _beh
sys.modules["services.analysis_history"] = _hist

_idata = _load_module_from_file("i18n_data", "i18n_data.py")
sys.modules["services.i18n_data"] = _idata
_i18n = _load_module_from_file("i18n", "i18n.py")
sys.modules["services.i18n"] = _i18n

_sc = _load_module_from_file("semantic_coder", "semantic_coder.py")
_rm = _load_module_from_file("roadmap_generator", "roadmap_generator.py")
_ec = _load_module_from_file("evidence_chain", "evidence_chain.py")
sys.modules["services.roadmap_generator"] = _rm
sys.modules["services.evidence_chain"] = _ec

_rx = _load_module_from_file("report_export", "report_export.py")
sys.modules["services.report_export"] = _rx

_rar = _load_module_from_file("revenue_at_risk", "revenue_at_risk.py")
_prd = _load_module_from_file("prd_generator", "prd_generator.py")
_qa = _load_module_from_file("insight_qa", "insight_qa.py")
_brief = _load_module_from_file("research_brief", "research_brief.py")
_tedits = _load_module_from_file("theme_edits", "theme_edits.py")
_strand = _load_module_from_file("strand_compare", "strand_compare.py")
_proj = _load_module_from_file("project_store", "project_store.py")
sys.modules["services.revenue_at_risk"] = _rar
sys.modules["services.prd_generator"] = _prd
sys.modules["services.insight_qa"] = _qa
sys.modules["services.research_brief"] = _brief
sys.modules["services.theme_edits"] = _tedits
sys.modules["services.strand_compare"] = _strand
sys.modules["services.project_store"] = _proj

# Re-export public API
auto_resolve_columns = _dl.auto_resolve_columns
read_raw_file = _dl.read_raw_file
normalize_raw_reviews = _dl.normalize_raw_reviews
normalize_rating_to_5 = _dl.normalize_rating_to_5
load_reviews_file = _dl.load_reviews_file

PreflightReport = _pf.PreflightReport
run_preflight = _pf.run_preflight

encode_reviews = _sc.encode_reviews

build_implementation_roadmap = _rm.build_implementation_roadmap
apply_roadmap_overrides = _rm.apply_roadmap_overrides

fetch_topic_evidence = _ec.fetch_topic_evidence
format_evidence_for_display = _ec.format_evidence_for_display
summarize_topic_evidence = _ec.summarize_topic_evidence

build_report_bundle = _rx.build_report_bundle
export_roadmap_csv = _rx.export_roadmap_csv
export_workbook_xlsx = _rx.export_workbook_xlsx
export_html_report = _rx.export_html_report
default_export_basename = _rx.default_export_basename

compute_credibility_report = _cred.compute_credibility_report
apply_topic_merge = _tmerge.apply_topic_merge
topic_label_needs_review = _tm.topic_label_needs_review
extract_topic_labels = _tm.extract_topic_labels

load_behavior_dataframe = _beh.load_behavior_dataframe
normalize_behavior_df = _beh.normalize_behavior_df
auto_map_behavior_columns = _beh.auto_map_behavior_columns
enrich_roadmap_with_behavior = _beh.enrich_roadmap_with_behavior

corpus_fingerprint = _cache.corpus_fingerprint
clear_analysis_cache = _cache.clear_analysis_cache

save_analysis_snapshot = _hist.save_analysis_snapshot
list_snapshots = _hist.list_snapshots
load_snapshot = _hist.load_snapshot
compare_snapshots = _hist.compare_snapshots
snapshots_to_dataframe = _hist.snapshots_to_dataframe

compute_revenue_at_risk = _rar.compute_revenue_at_risk
generate_prd_markdown = _prd.generate_prd_markdown
generate_user_stories = _prd.generate_user_stories
ask_insight = _qa.ask_insight
build_research_brief = _brief.build_research_brief
research_brief_to_markdown = _brief.research_brief_to_markdown
apply_brief_overrides = _brief.apply_brief_overrides

rename_topic = _tedits.rename_topic
merge_topics = _tedits.merge_topics
list_theme_stats = _tedits.list_theme_stats

build_strand_snapshot = _strand.build_strand_snapshot
detect_strand_key = _strand.detect_strand_key
both_strands_ready = _strand.both_strands_ready

list_projects = _proj.list_projects
save_project = _proj.save_project
load_project = _proj.load_project
create_project_id = _proj.create_project_id

t = _i18n.t
get_language = _i18n.get_language
set_language = _i18n.set_language
render_language_selector = _i18n.render_language_selector
translate_lifecycle_stage = _i18n.translate_lifecycle_stage
translate_risk = _i18n.translate_risk
translate_quadrant = _i18n.translate_quadrant
is_rtl = _i18n.is_rtl
SUPPORTED_LANGUAGES = _i18n.SUPPORTED_LANGUAGES
detect_corpus_languages = _ml.detect_corpus_languages
