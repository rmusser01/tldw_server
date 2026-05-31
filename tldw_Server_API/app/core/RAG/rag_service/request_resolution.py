"""Request canonicalization helpers for unified RAG handling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from tldw_Server_API.app.core.RAG.rag_service.profiles import get_profile_kwargs
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

_DEFAULT_PROFILE_ALIASES: dict[str, str] = {
    "fast": "fast",
    "balanced": "balanced",
    "accuracy": "accuracy",
    "speed": "fast",
    "quality": "accuracy",
}
_SEARCH_AGENT_BOOL_DEFAULTS: tuple[tuple[str, str, str], ...] = (
    ("enable_query_classification", "SEARCH_QUERY_CLASSIFICATION", "search_query_classification"),
    ("enable_query_reformulation", "SEARCH_QUERY_REFORMULATION", "search_query_reformulation"),
    ("enable_research_loop", "SEARCH_RESEARCH_LOOP", "search_research_loop"),
    ("enable_discussion_search", "SEARCH_DISCUSSIONS_ENABLED", "search_discussions_enabled"),
    ("enable_research_progress", "SEARCH_PROGRESS_STREAMING", "search_progress_streaming"),
    ("search_url_scraping", "SEARCH_URL_SCRAPING", "search_url_scraping"),
    ("enable_suggestions", "SEARCH_SUGGESTIONS", "search_suggestions"),
    ("enable_structured_response", "SEARCH_STRUCTURED_RESPONSE", "search_structured_response"),
    ("enable_image_search", "SEARCH_IMAGE_SEARCH", "search_image_search"),
    ("enable_video_search", "SEARCH_VIDEO_SEARCH", "search_video_search"),
)
_SEARCH_AGENT_INT_DEFAULTS: tuple[tuple[str, str, str], ...] = (
    ("research_max_iterations", "SEARCH_MAX_ITERATIONS", "search_max_iterations"),
    ("research_max_iterations_speed", "SEARCH_MAX_ITERATIONS_SPEED", "search_max_iterations_speed"),
    ("research_max_iterations_balanced", "SEARCH_MAX_ITERATIONS_BALANCED", "search_max_iterations_balanced"),
    ("research_max_iterations_quality", "SEARCH_MAX_ITERATIONS_QUALITY", "search_max_iterations_quality"),
)
_SEARCH_AGENT_MODE_VALUES = {"speed", "balanced", "quality"}


def _default_single_user_id_resolver() -> int:
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    return int(DatabasePaths.get_single_user_id())


def _is_truthy_value(raw_value: Any) -> bool:
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_or_json_list(raw_value: Any) -> Optional[list[str]]:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, list):
            values = [str(item).strip() for item in parsed if str(item).strip()]
            return values or None
    values = [item.strip() for item in text.split(",") if item.strip()]
    return values or None


def _parse_int_or_none(raw_value: Any) -> Optional[int]:
    if raw_value is None:
        return None
    try:
        return int(str(raw_value).strip())
    except (TypeError, ValueError):
        return None


def _request_fields_set(request: Any) -> set[str]:
    raw_fields = getattr(request, "model_fields_set", None)
    if raw_fields is None:
        raw_fields = getattr(request, "__fields_set__", None)
    if raw_fields is None:
        return set()
    try:
        return {str(name) for name in raw_fields}
    except (TypeError, ValueError):
        return set()


def _resolve_implicit_feedback_user_id(
    request_user_id: Optional[str],
    current_user: Optional[Any],
    *,
    single_user_id_resolver: Callable[[], Any],
) -> Optional[str]:
    raw_request = str(request_user_id).strip() if request_user_id is not None else ""
    if raw_request:
        if raw_request.lower() == "single_user":
            if current_user is not None:
                current_id = getattr(current_user, "id_int", None)
                if isinstance(current_id, int):
                    return str(current_id)
                fallback_id = getattr(current_user, "id", None)
                if fallback_id is not None:
                    fallback_raw = str(fallback_id).strip()
                    if fallback_raw:
                        return fallback_raw
            try:
                return str(single_user_id_resolver())
            except (RuntimeError, ValueError, OSError, TypeError):
                return None
        return raw_request

    if current_user is None:
        return None

    current_id = getattr(current_user, "id_int", None)
    if isinstance(current_id, int):
        return str(current_id)

    fallback_id = getattr(current_user, "id", None)
    if fallback_id is None:
        return None

    fallback_raw = str(fallback_id).strip()
    return fallback_raw or None


def apply_search_agent_defaults(
    request: Any,
    payload: dict[str, Any],
    *,
    search_agent_setting_fn: Callable[[str, str], Any],
    allowed_fields: Optional[set[str]] = None,
) -> None:
    """Apply Search-Agent defaults for fields omitted by the caller."""

    explicit_fields = _request_fields_set(request)

    for field_name, env_key, cfg_key in _SEARCH_AGENT_BOOL_DEFAULTS:
        if allowed_fields is not None and field_name not in allowed_fields:
            continue
        if field_name in explicit_fields:
            continue
        raw_value = search_agent_setting_fn(env_key, cfg_key)
        if raw_value is None:
            continue
        payload[field_name] = _is_truthy_value(raw_value)

    if (allowed_fields is None or "search_depth_mode" in allowed_fields) and "search_depth_mode" not in explicit_fields:
        raw_mode = search_agent_setting_fn("SEARCH_DEFAULT_MODE", "search_default_mode")
        if raw_mode is not None:
            mode = str(raw_mode).strip().lower()
            if mode in _SEARCH_AGENT_MODE_VALUES:
                payload["search_depth_mode"] = mode

    if (allowed_fields is None or "discussion_platforms" in allowed_fields) and "discussion_platforms" not in explicit_fields:
        raw_platforms = search_agent_setting_fn("SEARCH_DISCUSSION_PLATFORMS", "search_discussion_platforms")
        parsed_platforms = _parse_csv_or_json_list(raw_platforms)
        if parsed_platforms is not None:
            payload["discussion_platforms"] = parsed_platforms

    if (allowed_fields is None or "classifier_provider" in allowed_fields) and "classifier_provider" not in explicit_fields:
        raw_provider = search_agent_setting_fn("SEARCH_CLASSIFIER_PROVIDER", "search_classifier_provider")
        if raw_provider is not None and str(raw_provider).strip():
            payload["classifier_provider"] = str(raw_provider).strip()

    if (allowed_fields is None or "classifier_model" in allowed_fields) and "classifier_model" not in explicit_fields:
        raw_model = search_agent_setting_fn("SEARCH_CLASSIFIER_MODEL", "search_classifier_model")
        if raw_model is not None and str(raw_model).strip():
            payload["classifier_model"] = str(raw_model).strip()

    for field_name, env_key, cfg_key in _SEARCH_AGENT_INT_DEFAULTS:
        if allowed_fields is not None and field_name not in allowed_fields:
            continue
        if field_name in explicit_fields:
            continue
        parsed_value = _parse_int_or_none(search_agent_setting_fn(env_key, cfg_key))
        if parsed_value is None:
            continue
        if field_name == "research_max_iterations":
            if parsed_value > 0:
                payload[field_name] = parsed_value
            continue
        if parsed_value >= 0:
            payload[field_name] = parsed_value


@dataclass(slots=True)
class ResolvedRAGRequest:
    """Canonical request state used by RAG wrappers and pipeline callers."""

    query: str
    strategy: str
    payload: dict[str, Any]
    index_namespace: Optional[str]
    rag_profile: Optional[str]
    user_id: Optional[str]
    feedback_user_id: Optional[str]


def resolve_rag_request(
    request: Any,
    *,
    current_user: Optional[Any] = None,
    get_profile_kwargs_fn: Callable[[str], Mapping[str, Any]] = get_profile_kwargs,
    profile_aliases: Optional[Mapping[str, str]] = None,
    single_user_id_resolver: Callable[[], Any] = _default_single_user_id_resolver,
    search_agent_setting_fn: Optional[Callable[[str, str], Any]] = None,
    search_agent_allowed_fields: Optional[set[str]] = None,
) -> ResolvedRAGRequest:
    """
    Resolve a request into a canonical internal payload.

    Precedence for profile defaults is explicit request fields over profile values.
    User IDs are normalized to avoid legacy single-user alias path issues.
    """

    payload = model_dump_compat(request)
    explicit_fields = _request_fields_set(request)

    alias_map: dict[str, str] = dict(_DEFAULT_PROFILE_ALIASES)
    if profile_aliases:
        alias_map.update({str(k).strip().lower(): str(v).strip() for k, v in profile_aliases.items()})

    resolved_profile: Optional[str] = None
    raw_profile = payload.get("rag_profile")
    if raw_profile is not None:
        profile_key = str(raw_profile).strip().lower()
        if profile_key:
            resolved_profile = alias_map.get(profile_key, profile_key)
            payload["rag_profile"] = resolved_profile

    index_namespace = payload.get("index_namespace")
    if index_namespace is None:
        corpus = payload.get("corpus")
        if isinstance(corpus, str):
            stripped = corpus.strip()
            index_namespace = stripped or None
            if index_namespace is not None:
                payload["index_namespace"] = index_namespace

    if search_agent_setting_fn is not None:
        apply_search_agent_defaults(
            request,
            payload,
            search_agent_setting_fn=search_agent_setting_fn,
            allowed_fields=search_agent_allowed_fields,
        )

    if resolved_profile:
        try:
            profile_defaults = get_profile_kwargs_fn(resolved_profile)
        except ValueError:
            profile_defaults = {}
        for field_name, value in profile_defaults.items():
            if field_name in explicit_fields:
                continue
            payload[field_name] = value

    resolved_storage_user_id = _resolve_implicit_feedback_user_id(
        payload.get("user_id"),
        current_user,
        single_user_id_resolver=single_user_id_resolver,
    )
    if resolved_storage_user_id is None:
        resolved_storage_user_id = _resolve_implicit_feedback_user_id(
            None,
            current_user,
            single_user_id_resolver=single_user_id_resolver,
        )

    if payload.get("feedback_user_id") is not None:
        resolved_feedback_user_id = _resolve_implicit_feedback_user_id(
            payload.get("feedback_user_id"),
            current_user,
            single_user_id_resolver=single_user_id_resolver,
        )
    else:
        resolved_feedback_user_id = resolved_storage_user_id

    payload["user_id"] = resolved_storage_user_id
    payload["feedback_user_id"] = resolved_feedback_user_id
    payload["query"] = str(payload.get("query") or ((payload.get("queries") or [""])[0]))
    payload["strategy"] = str(payload.get("strategy") or "standard")

    return ResolvedRAGRequest(
        query=str(payload.get("query", "")),
        strategy=str(payload.get("strategy", "standard")),
        payload=payload,
        index_namespace=index_namespace if isinstance(index_namespace, str) else None,
        rag_profile=resolved_profile,
        user_id=resolved_storage_user_id,
        feedback_user_id=resolved_feedback_user_id,
    )


def resolve_legacy_standard_pipeline_request(
    *,
    query: str,
    search_mode: str,
    top_k: int,
    sources: Optional[list[str]] = None,
    min_score: float = 0.0,
    index_namespace: Optional[str] = None,
    rag_profile: Optional[str] = None,
    user_id: Optional[str] = None,
    feedback_user_id: Optional[str] = None,
    enable_generation: bool = True,
    include_sources: bool = True,
    include_metadata: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> ResolvedRAGRequest:
    """Resolve legacy ``unified_rag_pipeline`` kwargs into the standard contract."""

    payload = dict(metadata or {})
    payload.update(
        {
            "query": query,
            "strategy": "standard",
            "sources": list(sources or ["media_db"]),
            "search_mode": search_mode,
            "top_k": top_k,
            "min_score": min_score,
            "enable_generation": enable_generation,
            "include_sources": include_sources,
            "include_metadata": include_metadata,
        }
    )
    payload["user_id"] = user_id
    payload["feedback_user_id"] = feedback_user_id or user_id
    payload["index_namespace"] = index_namespace
    payload["rag_profile"] = rag_profile
    return ResolvedRAGRequest(
        query=query,
        strategy="standard",
        payload=payload,
        index_namespace=index_namespace,
        rag_profile=rag_profile,
        user_id=user_id,
        feedback_user_id=feedback_user_id or user_id,
    )
