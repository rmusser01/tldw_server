from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from tldw_Server_API.app.core.config import settings as _settings
from tldw_Server_API.app.core.custom_openai_providers import iter_custom_openai_provider_names

NO_SPACE_LANGS = {"zh", "zh-cn", "zh-tw", "ja", "ko", "th"}

_SCRIPT_LANGUAGE_HINTS: list[tuple[str, str]] = [
    (r"[\u4e00-\u9fff]", "zh"),
    (r"[\u3040-\u309f\u30a0-\u30ff]", "ja"),
    (r"[\u0e00-\u0e7f]", "th"),
    (r"[\u0900-\u097f]", "hi"),
    (r"[\u0400-\u04ff]", "ru"),
    (r"[\uac00-\ud7af]", "ko"),
    (r"[\u0600-\u06ff]", "ar"),
]

DEFAULT_NER_MODEL_MAP: dict[str, str] = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "pt": "pt_core_news_sm",
    "it": "it_core_news_sm",
    "nl": "nl_core_news_sm",
}

LLM_PROVIDER_MODES = {
    "openai",
    "anthropic",
    "cohere",
    "google",
    "groq",
    "huggingface",
    "openrouter",
    "deepseek",
    "mistral",
    "ollama",
    "kobold",
    "ooba",
    "tabbyapi",
    "vllm",
    "custom-openai-api",
    "custom-openai-api-2",
    "local-llm",
    "llama.cpp",
}
LLM_PROVIDER_MODES.update(iter_custom_openai_provider_names(start=3))


@dataclass(frozen=True)
class ClaimsExtractorCatalogItem:
    mode: str
    label: str
    description: str
    execution: str
    supports_languages: list[str] | None = None
    providers: list[str] | None = None
    auto_selectable: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "mode": self.mode,
            "label": self.label,
            "description": self.description,
            "execution": self.execution,
            "supports_languages": self.supports_languages,
            "providers": self.providers,
            "auto_selectable": self.auto_selectable,
        }
        return {k: v for k, v in payload.items() if v is not None}


def _load_custom_ner_model_map() -> dict[str, str]:
    raw = _settings.get("CLAIMS_LOCAL_NER_MODEL_MAP")
    if isinstance(raw, dict):
        return {str(k).lower(): str(v) for k, v in raw.items() if v}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(parsed, dict):
            return {str(k).lower(): str(v) for k, v in parsed.items() if v}
    return {}


def resolve_ner_model_name(language: str | None) -> str:
    lang = (language or "").strip().lower()
    custom_map = _load_custom_ner_model_map()
    if lang in custom_map:
        return custom_map[lang]
    default_model = str(_settings.get("CLAIMS_LOCAL_NER_MODEL", "") or "").strip()
    if default_model:
        return default_model
    return DEFAULT_NER_MODEL_MAP.get(lang, DEFAULT_NER_MODEL_MAP.get("en", "en_core_web_sm"))


@lru_cache(maxsize=8)
def get_spacy_pipeline(model_name: str, language: str) -> Any | None:
    try:
        import spacy  # type: ignore
    except (ImportError, ModuleNotFoundError):
        return None

    model_name = (model_name or "").strip()
    language = (language or "xx").strip() or "xx"

    if model_name:
        try:
            return spacy.load(model_name)
        except (OSError, RuntimeError, TypeError, ValueError):
            pass

    try:
        nlp = spacy.blank(language)
    except (AttributeError, KeyError, TypeError, ValueError):
        try:
            nlp = spacy.blank("xx")
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    if not nlp.has_pipe("sentencizer"):
        with contextlib.suppress(KeyError, RuntimeError, TypeError, ValueError):
            nlp.add_pipe("sentencizer")
    return nlp


def detect_claims_language(text: str | None, default: str | None = None) -> str:
    fallback = (default or str(_settings.get("CLAIMS_EXTRACTOR_LANGUAGE_DEFAULT", "en"))).strip() or "en"
    if not text:
        return fallback
    for pattern, lang in _SCRIPT_LANGUAGE_HINTS:
        try:
            if re.search(pattern, text):
                return lang
        except (re.error, TypeError, ValueError):
            continue
    return fallback


def split_claims_sentences(
    text: str,
    language: str | None,
    *,
    min_length: int | None = None,
    max_sentences: int | None = None,
) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    lang = (language or "").strip().lower()
    if lang in NO_SPACE_LANGS:
        pattern = r"(?<=[。！？!?])\s*"
        default_min = 8
    else:
        pattern = r"(?<=[\.!?…؟])\s+"
        default_min = 12

    try:
        parts = re.split(pattern, cleaned)
    except (re.error, TypeError, ValueError):
        parts = [cleaned]

    threshold = min_length if min_length is not None else default_min
    sentences = []
    for part in parts:
        sentence = (part or "").strip()
        if len(sentence) >= threshold:
            sentences.append(sentence)
        if max_sentences and len(sentences) >= max_sentences:
            break
    return sentences


def resolve_claims_extractor_mode(
    requested: str | None,
    text: str | None,
) -> tuple[str, str]:
    normalized = (requested or "heuristic").strip().lower()
    language = detect_claims_language(text)

    if normalized not in {"auto", "detect"}:
        return normalized, language

    if language in NO_SPACE_LANGS:
        return "heuristic", language

    model_name = resolve_ner_model_name(language)
    nlp = get_spacy_pipeline(model_name, language)
    if nlp is not None and nlp.has_pipe("ner"):
        return "ner", language

    return "heuristic", language


def get_claims_extractor_catalog() -> list[dict[str, Any]]:
    lang_support = sorted(set(DEFAULT_NER_MODEL_MAP.keys()))
    providers = sorted(LLM_PROVIDER_MODES)
    items = [
        ClaimsExtractorCatalogItem(
            mode="heuristic",
            label="Heuristic",
            description="Local sentence-based extraction with language-aware punctuation.",
            execution="local",
            supports_languages=["any"],
            auto_selectable=True,
        ),
        ClaimsExtractorCatalogItem(
            mode="ner",
            label="NER-assisted",
            description="Local NER-driven sentence selection with spaCy models.",
            execution="local",
            supports_languages=lang_support,
            auto_selectable=True,
        ),
        ClaimsExtractorCatalogItem(
            mode="aps",
            label="APS propositions",
            description="LLM-backed proposition extraction via APS prompt strategy.",
            execution="llm",
            supports_languages=["any"],
        ),
        ClaimsExtractorCatalogItem(
            mode="llm",
            label="LLM extractor",
            description="LLM-based extraction using the configured provider/model.",
            execution="llm",
            supports_languages=["any"],
            providers=providers,
        ),
        ClaimsExtractorCatalogItem(
            mode="auto",
            label="Auto",
            description="Auto-selects between heuristic and NER based on language hints.",
            execution="local",
            supports_languages=["any"],
        ),
    ]
    return [item.to_dict() for item in items]


__all__ = [
    "LLM_PROVIDER_MODES",
    "NO_SPACE_LANGS",
    "detect_claims_language",
    "resolve_claims_extractor_mode",
    "resolve_ner_model_name",
    "get_spacy_pipeline",
    "get_claims_extractor_catalog",
    "split_claims_sentences",
]
