"""
Internationalization (i18n) for InsightOptima UI.

Supports 22 major world languages with English fallback for missing keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from services.i18n_data import ENGLISH_STRINGS

LOCALES_DIR = Path(__file__).resolve().parent.parent / "locales"
DEFAULT_LANGUAGE = "en"

# ISO-style codes → native display names (language selector)
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh": "中文 (简体)",
    "zh-TW": "中文 (繁體)",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "ja": "日本語",
    "ko": "한국어",
    "pt": "Português",
    "ar": "العربية",
    "hi": "हिन्दी",
    "ru": "Русский",
    "it": "Italiano",
    "nl": "Nederlands",
    "tr": "Türkçe",
    "vi": "Tiếng Việt",
    "th": "ไทย",
    "id": "Bahasa Indonesia",
    "pl": "Polski",
    "ms": "Bahasa Melayu",
    "bn": "বাংলা",
    "uk": "Українська",
}

# RTL languages — apply right-to-left layout
RTL_LANGUAGES = {"ar"}

_messages_cache: dict[str, dict[str, str]] | None = None
_english_mtime: float | None = None
_english_live: dict[str, str] | None = None


def _live_english_strings() -> dict[str, str]:
    """
    Always read the on-disk English catalog (mtime-cached).

    Streamlit keeps imported modules warm across reruns; without this,
    newly added UI keys can render as raw key names.
    """
    global _english_mtime, _english_live
    path = Path(__file__).resolve().parent / "i18n_data.py"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return dict(ENGLISH_STRINGS)

    if _english_live is not None and _english_mtime == mtime:
        return _english_live

    import importlib.util

    spec = importlib.util.spec_from_file_location("_insightoptima_i18n_data_live", path)
    if spec is None or spec.loader is None:
        return dict(ENGLISH_STRINGS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _english_live = dict(getattr(module, "ENGLISH_STRINGS", ENGLISH_STRINGS))
    _english_mtime = mtime
    return _english_live


def _load_all_messages() -> dict[str, dict[str, str]]:
    """Load merged message catalog; always sync live English keys."""
    global _messages_cache
    english = _live_english_strings()

    if _messages_cache is None:
        merged: dict[str, dict[str, str]] = {"en": dict(english)}
        path = LOCALES_DIR / "messages.json"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                loaded = json.load(f)
            for lang, strings in loaded.items():
                if lang == "en":
                    continue
                merged[lang] = dict(english)
                merged[lang].update(strings)
        _messages_cache = merged
    else:
        _messages_cache["en"] = dict(english)
        for lang, strings in _messages_cache.items():
            if lang == "en":
                continue
            for key, value in english.items():
                strings.setdefault(key, value)

    return _messages_cache


def get_language() -> str:
    """Return active UI language code from session state."""
    return st.session_state.get("ui_language", DEFAULT_LANGUAGE)


def set_language(code: str) -> None:
    """Set active UI language."""
    if code in SUPPORTED_LANGUAGES:
        st.session_state["ui_language"] = code


def t(key: str, **kwargs: str | int | float) -> str:
    """
    Translate a message key to the active UI language.

    Falls back: selected lang → zh-TW→zh → live English → raw key.
    """
    catalog = _load_all_messages()
    english = catalog.get("en") or _live_english_strings()
    lang = get_language()

    chain = [lang]
    if lang == "zh-TW":
        chain.append("zh")
    chain.append("en")

    text = english.get(key, key)
    for code in chain:
        if code in catalog and key in catalog[code]:
            candidate = catalog[code][key]
            if candidate and candidate != key:
                text = candidate
                break

    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    return text


def is_rtl() -> bool:
    """Whether active language uses right-to-left text direction."""
    return get_language() in RTL_LANGUAGES


def render_language_selector(*, in_sidebar: bool = True) -> None:
    """Render global language selector (sidebar or nested panel)."""
    codes = list(SUPPORTED_LANGUAGES.keys())
    labels = [SUPPORTED_LANGUAGES[c] for c in codes]
    current = get_language()
    index = codes.index(current) if current in codes else 0

    def _body() -> None:
        selected_label = st.selectbox(
            t("ui_language"),
            options=labels,
            index=index,
            key="ui_language_select",
            help=t("ui_language"),
            label_visibility="collapsed",
        )
        selected_code = codes[labels.index(selected_label)]
        if selected_code != current:
            set_language(selected_code)
            st.rerun()

        if is_rtl():
            st.markdown(
                '<div dir="rtl" style="text-align:right;font-size:0.85rem;color:#aaa;">'
                f"{t('rtl_active_note')}</div>",
                unsafe_allow_html=True,
            )

    if in_sidebar:
        with st.sidebar:
            _body()
    else:
        _body()


def translate_lifecycle_stage(stage: str) -> str:
    """Map internal English lifecycle stage name to localized label."""
    mapping = {
        "Onboarding": "stage_onboarding",
        "Core Feature Activation": "stage_core",
        "Daily Retention": "stage_retention",
        "General feedback": "stage_general",
    }
    key = mapping.get(stage)
    return t(key) if key else stage


def translate_risk(level: str) -> str:
    """Map risk level label to localized string."""
    mapping = {"High": "risk_high", "Medium": "risk_medium", "Low": "risk_low"}
    key = mapping.get(level)
    return t(key) if key else level


def translate_quadrant(quadrant: str) -> str:
    """Map quadrant zone name to localized string."""
    mapping = {
        "Red Core Blast Zone": "quad_red",
        "Purple Hidden Sting Zone": "quad_purple",
        "Yellow Monitor Zone": "quad_yellow",
        "Green Low Priority Zone": "quad_green",
    }
    key = mapping.get(quadrant)
    return t(key) if key else quadrant
