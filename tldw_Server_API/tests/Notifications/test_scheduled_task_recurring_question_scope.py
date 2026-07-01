from __future__ import annotations

from tldw_Server_API.app.core.Scheduled_Tasks.recurring_question_scope import (
    normalize_recurring_question_scope,
)


def test_all_searchable_library_resolves_capability_reported_sources():
    scope, errors, warnings = normalize_recurring_question_scope(
        {"mode": "all_searchable_library"},
        available_sources=["media_db", "notes"],
    )

    assert errors == []  # nosec B101
    assert warnings == []  # nosec B101
    assert scope == {  # nosec B101
        "mode": "all_searchable_library",
        "resolved_sources": ["media_db", "notes"],
    }


def test_explicit_sources_are_deduplicated_and_filtered_to_available_sources():
    scope, errors, warnings = normalize_recurring_question_scope(
        {"sources": ["media_db", "github", "media_db", "youtube"]},
        available_sources=["media_db", "youtube"],
    )

    assert errors == []  # nosec B101
    assert warnings == [{"code": "source_unavailable", "source": "github"}]  # nosec B101
    assert scope == {  # nosec B101
        "mode": "sources",
        "sources": ["media_db", "youtube"],
    }


def test_empty_scope_returns_scope_empty_error():
    scope, errors, warnings = normalize_recurring_question_scope(
        {"sources": []},
        available_sources=["media_db"],
    )

    assert scope == {"mode": "sources", "sources": []}  # nosec B101
    assert warnings == []  # nosec B101
    assert errors == [  # nosec B101
        {
            "field": "config.scope",
            "code": "scope_empty",
            "message": "Scope must include at least one readable searchable source.",
        }
    ]


def test_unsupported_scope_fields_return_field_errors():
    _scope, errors, _warnings = normalize_recurring_question_scope(
        {"sources": ["media_db"], "provider": "github"},
        available_sources=["media_db"],
    )

    assert errors == [  # nosec B101
        {
            "field": "config.scope.provider",
            "code": "unsupported",
            "message": "Unsupported scope field: provider",
        }
    ]
