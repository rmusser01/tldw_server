"""Scope normalization for Recurring Question scheduled tasks."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.Scheduled_Tasks.recurring_question_models import (
    DEFAULT_SEARCHABLE_SOURCES,
    SUPPORTED_SCOPE_FIELDS,
)


def normalize_recurring_question_scope(
    scope: Any,
    *,
    available_sources: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize a Recurring Question scope without binding to source-specific UI."""
    readable_sources = list(dict.fromkeys(available_sources or DEFAULT_SEARCHABLE_SOURCES))
    raw_scope = scope if isinstance(scope, dict) else {}
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for field in raw_scope:
        if field not in SUPPORTED_SCOPE_FIELDS:
            errors.append(
                {
                    "field": f"config.scope.{field}",
                    "code": "unsupported",
                    "message": f"Unsupported scope field: {field}",
                }
            )

    mode = raw_scope.get("mode")
    if mode is not None and mode not in {"all_searchable_library", "sources"}:
        return {
            "mode": str(mode),
        }, [
            {
                "field": "config.scope.mode",
                "code": "unsupported",
                "message": f"Unsupported scope mode: {mode}",
            }
        ], warnings

    if mode == "all_searchable_library" or (mode is None and "sources" not in raw_scope):
        normalized = {
            "mode": "all_searchable_library",
            "resolved_sources": readable_sources,
        }
        if not readable_sources:
            errors.append(_scope_empty_error())
        return normalized, errors, warnings

    requested_sources = _string_list(raw_scope.get("sources"))
    resolved_sources: list[str] = []
    for source in requested_sources:
        if source in readable_sources:
            resolved_sources.append(source)
        else:
            warnings.append({"code": "source_unavailable", "source": source})

    normalized = {"mode": "sources", "sources": list(dict.fromkeys(resolved_sources))}
    for field in (
        "collection_ids",
        "tag_ids",
        "saved_search_ids",
        "source_types",
        "date_window",
        "workspace_id",
        "advanced_filters",
    ):
        if field in raw_scope:
            normalized[field] = raw_scope[field]

    if not normalized["sources"]:
        errors.append(_scope_empty_error())
    return normalized, errors, warnings


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _scope_empty_error() -> dict[str, str]:
    return {
        "field": "config.scope",
        "code": "scope_empty",
        "message": "Scope must include at least one readable searchable source.",
    }
