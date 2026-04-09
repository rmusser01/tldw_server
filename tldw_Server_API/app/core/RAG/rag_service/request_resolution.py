"""Request canonicalization helpers for unified RAG handling."""

from __future__ import annotations

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


def _default_single_user_id_resolver() -> int:
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    return int(DatabasePaths.get_single_user_id())


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
        None,
        current_user,
        single_user_id_resolver=single_user_id_resolver,
    )
    if resolved_storage_user_id is None:
        resolved_storage_user_id = _resolve_implicit_feedback_user_id(
            payload.get("user_id"),
            current_user,
            single_user_id_resolver=single_user_id_resolver,
        )

    resolved_feedback_user_id = (
        _resolve_implicit_feedback_user_id(
            payload.get("feedback_user_id"),
            current_user,
            single_user_id_resolver=single_user_id_resolver,
        )
        or resolved_storage_user_id
    )

    payload["user_id"] = resolved_storage_user_id
    payload["feedback_user_id"] = resolved_feedback_user_id

    return ResolvedRAGRequest(
        query=str(payload.get("query", "")),
        strategy=str(payload.get("strategy", "standard")),
        payload=payload,
        index_namespace=index_namespace if isinstance(index_namespace, str) else None,
        rag_profile=resolved_profile,
        user_id=resolved_storage_user_id,
        feedback_user_id=resolved_feedback_user_id,
    )
