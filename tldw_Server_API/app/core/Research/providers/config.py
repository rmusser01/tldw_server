"""Config resolution for deep research provider-backed execution."""

from __future__ import annotations

import configparser
from copy import deepcopy
from typing import Any

from tldw_Server_API.app.core.config import get_config_value

_VALID_LOCAL_SOURCES = ("media_db", "notes", "prompts", "kanban")
_VALID_ACADEMIC_PROVIDERS = ("arxiv", "pubmed", "crossref")
_VALID_WEB_ENGINES = (
    "duckduckgo",
    "brave",
    "google",
    "kagi",
    "serper",
    "tavily",
    "searx",
    "bing",
    "yandex",
)


def _coerce_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _coerce_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _dedupe_allowlisted(values: Any, *, allowlist: tuple[str, ...]) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for item in values:
        candidate = str(item).strip().lower()
        if candidate in allowlist and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _default_provider_config() -> dict[str, Any]:
    default_web_engine = str(
        get_config_value("Search-Agent", "deep_research_web_engine", default="duckduckgo") or "duckduckgo"
    ).strip().lower()
    if default_web_engine not in _VALID_WEB_ENGINES:
        default_web_engine = "duckduckgo"

    raw_vector = str(get_config_value("Search-Agent", "deep_research_local_use_vector", default="true")).strip().lower()
    try:
        use_vector = configparser.ConfigParser.BOOLEAN_STATES[raw_vector]
    except KeyError:
        raise ValueError("deep_research_local_use_vector must be a boolean") from None

    return {
        "local": {
            "use_vector": use_vector,
            "top_k": _coerce_int(
                get_config_value("Search-Agent", "deep_research_local_top_k", default="5"),
                default=5,
                minimum=1,
                maximum=20,
            ),
            "sources": ["media_db"],
        },
        "web": {
            "engine": default_web_engine,
            "result_count": _coerce_int(
                get_config_value("Search-Agent", "deep_research_web_result_count", default="5"),
                default=5,
                minimum=1,
                maximum=20,
            ),
        },
        "academic": {
            "providers": ["arxiv", "pubmed", "crossref"],
            "max_results": _coerce_int(
                get_config_value("Search-Agent", "deep_research_academic_max_results", default="5"),
                default=5,
                minimum=1,
                maximum=20,
            ),
        },
        "synthesis": {
            "provider": (get_config_value("Search-Agent", "deep_research_synthesis_provider", default=None) or None),
            "model": (get_config_value("Search-Agent", "deep_research_synthesis_model", default=None) or None),
            "temperature": _coerce_float(
                get_config_value("Search-Agent", "deep_research_synthesis_temperature", default="0.2"),
                default=0.2,
                minimum=0.0,
                maximum=1.0,
            ),
        },
    }


def resolve_provider_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Return bounded provider config for a deep research run."""
    resolved = deepcopy(_default_provider_config())
    raw = overrides if isinstance(overrides, dict) else {}

    local = raw.get("local")
    if isinstance(local, dict):
        if "top_k" in local:
            resolved["local"]["top_k"] = _coerce_int(local.get("top_k"), default=5, minimum=1, maximum=20)
        sources = _dedupe_allowlisted(local.get("sources"), allowlist=_VALID_LOCAL_SOURCES)
        if sources:
            resolved["local"]["sources"] = sources

    web = raw.get("web")
    if isinstance(web, dict):
        engine = str(web.get("engine") or "").strip().lower()
        if engine in _VALID_WEB_ENGINES:
            resolved["web"]["engine"] = engine
        if "result_count" in web:
            resolved["web"]["result_count"] = _coerce_int(web.get("result_count"), default=5, minimum=1, maximum=20)

    academic = raw.get("academic")
    if isinstance(academic, dict):
        providers = _dedupe_allowlisted(academic.get("providers"), allowlist=_VALID_ACADEMIC_PROVIDERS)
        if providers:
            resolved["academic"]["providers"] = providers
        if "max_results" in academic:
            resolved["academic"]["max_results"] = _coerce_int(
                academic.get("max_results"),
                default=5,
                minimum=1,
                maximum=20,
            )

    synthesis = raw.get("synthesis")
    if isinstance(synthesis, dict):
        provider = str(synthesis.get("provider") or "").strip() or None
        model = str(synthesis.get("model") or "").strip() or None
        if provider is not None:
            resolved["synthesis"]["provider"] = provider
        if model is not None:
            resolved["synthesis"]["model"] = model
        if "temperature" in synthesis:
            resolved["synthesis"]["temperature"] = _coerce_float(
                synthesis.get("temperature"),
                default=0.2,
                minimum=0.0,
                maximum=1.0,
            )

    return resolved


__all__ = ["resolve_provider_config"]
