"""Recipient shared-workspace API contract tests."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import sharing
from tldw_Server_API.app.api.v1.schemas.shared_workspace_recipient_schemas import (
    SharedWorkspaceChatRequest,
    SharedWorkspaceCitation,
    SharedWorkspaceErrorDetail,
    SharedWorkspacePartialError,
    SharedWorkspaceSource,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.chacha.shared_workspace_chat_store import (
    SharedWorkspaceChatStore,
    SharedWorkspaceStoredMessage,
)
from tldw_Server_API.app.core.DB_Management.chacha.shared_workspace_chat_store import (
    SharedWorkspaceMessagePage as StoredMessagePage,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
    SharedWorkspaceAccessContext,
    SharedWorkspaceNotFound,
    SharedWorkspaceUnavailable,
)

pytestmark = pytest.mark.integration


AUTHENTICATION_REQUIRED = {
    "detail": {
        "code": "authentication_required",
        "message": "Authentication is required.",
        "retryable": False,
    }
}
PERMISSION_REQUIRED = {
    "detail": {
        "code": "sharing_permission_required",
        "message": "The sharing.read permission is required.",
        "retryable": False,
    }
}
READ_RATE_LIMITED = {
    "detail": {
        "code": "shared_workspace_rate_limited",
        "message": "Shared workspace requests are temporarily rate limited.",
        "retryable": True,
        "recovery_action": "retry",
    }
}
INVALID_WORKSPACE_REQUEST = {
    "detail": {
        "code": "invalid_shared_workspace_request",
        "message": "The shared workspace request is invalid.",
        "retryable": False,
    }
}
INVALID_CHAT_REQUEST = {
    "detail": {
        "code": "invalid_shared_chat_request",
        "message": "The shared chat request is invalid.",
        "retryable": False,
    }
}
NOT_FOUND = {
    "detail": {
        "code": "shared_workspace_not_found",
        "message": "Shared workspace not found.",
        "retryable": False,
    }
}
UNAVAILABLE = {
    "detail": {
        "code": "shared_workspace_unavailable",
        "message": "Shared workspace is temporarily unavailable.",
        "retryable": True,
        "recovery_action": "retry",
    }
}


def _context(**overrides: Any) -> SharedWorkspaceAccessContext:
    values: dict[str, Any] = {
        "share_id": 42,
        "workspace_id": "workspace-alpha",
        "owner_user_id": 7,
        "recipient_user_id": 9,
        "share_scope_type": "team",
        "share_scope_id": 11,
        "access_level": "view_chat",
        "allow_clone": False,
        "owner_display_name": "Research owner",
        "shared_at": "2026-08-20T18:00:00+00:00",
        "workspace": {
            "id": "workspace-alpha",
            "name": "Evidence review",
            "description": "Review set",
        },
        "policy_actions": {
            "inspect_sources": {"allowed": True, "reason_code": None},
            "ask_grounded_questions": {"allowed": True, "reason_code": None},
            "add_sources": {
                "allowed": False,
                "reason_code": "shared_write_not_available",
            },
            "edit_workspace": {
                "allowed": False,
                "reason_code": "shared_write_not_available",
            },
            "clone_workspace": {"allowed": False, "reason_code": "clone_deferred"},
        },
    }
    values.update(overrides)
    return SharedWorkspaceAccessContext(**values)


def _source(index: int, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "id": f"source-{index:03d}",
        "workspace_id": "workspace-alpha",
        "media_id": 10_000 + index,
        "title": f"Evidence {index:03d}",
        "source_type": "pdf",
        "url": f"https://example.test/reports/{index}?secret=yes#fragment",
        "position": index,
        "added_at": "2026-08-20T18:00:00+00:00",
        "selected": True,
    }
    values.update(overrides)
    return values


def _status(source: dict[str, Any], *, state: str = "queryable") -> dict[str, Any]:
    queryable = state in {"queryable", "partially_queryable"}
    return {
        "id": source["id"],
        "state": state,
        "status_reason": "source_queryable" if queryable else f"source_{state}",
        "readiness": {
            "text_extracted": queryable,
            "fts_ready": queryable,
            "citation_ready": queryable,
            "tool_accessible": queryable,
        },
    }


class _AccessService:
    def __init__(
        self,
        *,
        context: SharedWorkspaceAccessContext | None = None,
        error: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.context = context or _context()
        self.error = error
        self.events = events if events is not None else []

    async def resolve(self, *, share_id: int, recipient_user_id: int):
        self.events.append(f"access:{share_id}:{recipient_user_id}")
        if self.error is not None:
            raise self.error
        return self.context


@pytest.fixture
def test_user() -> User:
    return User(
        id=9,
        username="recipient",
        email="recipient@example.test",
        password_hash="hash",
    )


@pytest.fixture
def principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=9,
        username="recipient",
        permissions=["sharing.read"],
    )


@pytest.fixture
def api_factory(monkeypatch, test_user, principal):
    def _build(
        *,
        service: _AccessService | None = None,
        auth_principal: AuthPrincipal | None = principal,
        principal_error: HTTPException | None = None,
        rate_error: HTTPException | None = None,
    ) -> tuple[TestClient, _AccessService]:
        resolved_service = service or _AccessService()

        async def _principal():
            if principal_error is not None:
                raise principal_error
            return auth_principal

        async def _user():
            return test_user

        async def _rate_limit(*_args, **_kwargs):
            if rate_error is not None:
                raise rate_error

        async def _sources(context):
            return [_source(1), _source(2)]

        async def _projection(context, sources):
            return {
                "sources": [_status(source) for source in sources],
                "summary": {
                    "total": len(sources),
                    "queryable": len(sources),
                    "processing": 0,
                    "failed": 0,
                },
                "partial_errors": [],
            }

        async def _history(context, *, before, limit):
            return StoredMessagePage(messages=(), next_before=None)

        async def _generation(context):
            return {
                "provider": None,
                "model": None,
                "ready": False,
                "reason_code": "no_provider_configured",
            }

        monkeypatch.setattr(auth_deps, "enforce_rbac_rate_limit", _rate_limit)
        defaults = {
            "_load_recipient_workspace_sources": _sources,
            "_project_recipient_source_status": _projection,
            "_load_recipient_chat_history": _history,
            "_resolve_recipient_generation_default": _generation,
        }
        for name, default in defaults.items():
            current = getattr(sharing, name)
            if getattr(current, "__module__", None) == sharing.__name__:
                monkeypatch.setattr(sharing, name, default)

        app = FastAPI()
        app.include_router(sharing.router, prefix="/api/v1")
        app.dependency_overrides[auth_deps.get_auth_principal] = _principal
        app.dependency_overrides[sharing.get_request_user] = _user
        app.dependency_overrides[sharing.get_shared_workspace_access_service] = (
            lambda: resolved_service
        )
        return TestClient(app, raise_server_exceptions=False), resolved_service

    return _build


def test_recipient_models_forbid_extras_and_enforce_projection_bounds() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SharedWorkspaceErrorDetail(
            code="x",
            message="safe",
            retryable=False,
            raw_error="/private/owner.db",
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SharedWorkspaceChatRequest.model_validate(
            {
                "request_id": "de305d54-75b4-431b-adb2-eb6b9e546014",
                "query": "Question",
                "source_scope": {"mode": "all", "source_ids": []},
                "system_message": "leak",
            }
        )
    with pytest.raises(ValidationError):
        SharedWorkspaceSource(
            source_id="source-a",
            title="x" * 513,
            source_type="pdf",
            state="queryable",
            citation_ready=True,
            retrieval_ready=True,
            position=0,
            added_at="2026-08-20T18:00:00+00:00",
        )
    with pytest.raises(ValidationError):
        SharedWorkspacePartialError(
            area="history",
            code="history_unavailable",
            message="x" * 321,
            retryable=True,
        )


def test_recipient_models_reject_invalid_timestamps_and_non_finite_scores() -> None:
    with pytest.raises(ValidationError):
        SharedWorkspaceSource(
            source_id="source-a",
            title="Evidence",
            source_type="pdf",
            state="queryable",
            citation_ready=True,
            retrieval_ready=True,
            position=0,
            added_at="not-a-timestamp",
        )
    for score in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValidationError):
            SharedWorkspaceCitation(
                citation_id="citation-a",
                source_id="source-a",
                source_title="Evidence",
                locator={"chunk": 1},
                quote="Bounded quote",
                score=score,
            )


def test_route_scoped_authentication_error_is_typed(api_factory) -> None:
    client, _service = api_factory(
        principal_error=HTTPException(status_code=401, detail="raw auth detail")
    )

    response = client.get("/api/v1/sharing/shared-with-me/42/workspace")

    assert response.status_code == 401
    assert response.json() == AUTHENTICATION_REQUIRED


def test_route_scoped_permission_error_is_typed(api_factory, principal) -> None:
    denied = principal.model_copy(update={"permissions": []})
    client, _service = api_factory(auth_principal=denied)

    response = client.get("/api/v1/sharing/shared-with-me/42/workspace")

    assert response.status_code == 403
    assert response.json() == PERMISSION_REQUIRED


def test_route_scoped_read_rate_limit_error_is_typed(api_factory) -> None:
    client, _service = api_factory(
        rate_error=HTTPException(status_code=429, detail="raw rate-limit detail")
    )

    response = client.get("/api/v1/sharing/shared-with-me/42/workspace")

    assert response.status_code == 429
    assert response.json() == READ_RATE_LIMITED


def test_route_scoped_validation_error_is_typed(api_factory) -> None:
    client, _service = api_factory()

    response = client.get("/api/v1/sharing/shared-with-me/not-an-int/sources")

    assert response.status_code == 422
    assert response.json() == INVALID_WORKSPACE_REQUEST


def test_access_failures_have_exact_neutral_or_safe_operational_errors(api_factory) -> None:
    bodies = []
    for error in (
        SharedWorkspaceNotFound(),
        SharedWorkspaceNotFound(),
        SharedWorkspaceNotFound(),
    ):
        client, _service = api_factory(service=_AccessService(error=error))
        response = client.get("/api/v1/sharing/shared-with-me/42/workspace")
        assert response.status_code == 404
        bodies.append(response.json())
    assert bodies == [NOT_FOUND, NOT_FOUND, NOT_FOUND]

    client, _service = api_factory(
        service=_AccessService(error=SharedWorkspaceUnavailable())
    )
    response = client.get("/api/v1/sharing/shared-with-me/42/workspace")
    assert response.status_code == 503
    assert response.json() == UNAVAILABLE


def test_bootstrap_is_bounded_and_disables_ask_without_generation(api_factory, monkeypatch) -> None:
    sources = [_source(index) for index in range(75)]
    messages = tuple(
        SharedWorkspaceStoredMessage(
            message_id=f"message-{index}",
            role="user" if index % 2 == 0 else "assistant",
            content=f"Message {index}",
            created_at=datetime(2026, 8, 20, 18, index % 60, tzinfo=timezone.utc),
            last_modified=datetime(2026, 8, 20, 18, index % 60, tzinfo=timezone.utc),
        )
        for index in range(30)
    )
    calls: list[tuple[str, int]] = []

    async def _sources(context):
        return sources

    async def _projection(context, projected_sources):
        calls.append(("projection", len(projected_sources)))
        return {
            "sources": [_status(source) for source in projected_sources],
            "summary": {
                "total": len(projected_sources),
                "queryable": len(projected_sources),
                "processing": 0,
                "failed": 0,
            },
            "partial_errors": [
                {
                    "area": f"optional-{index}",
                    "code": "optional_unavailable",
                    "message": "Optional data is unavailable.",
                    "retryable": True,
                }
                for index in range(12)
            ],
        }

    async def _history(context, *, before, limit):
        calls.append(("history", limit))
        return StoredMessagePage(messages=messages, next_before="older")

    monkeypatch.setattr(sharing, "_load_recipient_workspace_sources", _sources)
    monkeypatch.setattr(sharing, "_project_recipient_source_status", _projection)
    monkeypatch.setattr(sharing, "_load_recipient_chat_history", _history)
    client, _service = api_factory()

    response = client.get("/api/v1/sharing/shared-with-me/42/workspace")

    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]["items"]) == 50
    assert body["sources"]["pagination"] == {
        "offset": 0,
        "limit": 50,
        "total": 75,
        "has_more": True,
    }
    assert len(body["conversation"]["messages"]) == 30
    assert body["conversation"]["next_before"] == "older"
    assert len(body["partial_errors"]) == 8
    assert body["allowed_actions"]["inspect_sources"]["allowed"] is True
    assert body["allowed_actions"]["ask_grounded_questions"] == {
        "allowed": False,
        "reason_code": "no_provider_configured",
    }
    assert body["generation_default"] == {
        "provider": None,
        "model": None,
        "ready": False,
        "reason_code": "no_provider_configured",
    }
    assert calls == [("projection", 75), ("history", 30)]
    serialized = response.text.lower()
    assert "media_id" not in serialized
    assert "owner_user_id" not in serialized
    assert "share_scope" not in serialized


def test_empty_readable_workspace_keeps_inspection_and_disables_ask(api_factory, monkeypatch) -> None:
    async def _sources(context):
        return []

    async def _projection(context, sources):
        return {
            "sources": [],
            "summary": {"total": 0, "queryable": 0, "processing": 0, "failed": 0},
            "partial_errors": [],
        }

    monkeypatch.setattr(sharing, "_load_recipient_workspace_sources", _sources)
    monkeypatch.setattr(sharing, "_project_recipient_source_status", _projection)
    client, _service = api_factory()

    response = client.get("/api/v1/sharing/shared-with-me/42/workspace")

    actions = response.json()["allowed_actions"]
    assert actions["inspect_sources"] == {"allowed": True, "reason_code": None}
    assert actions["ask_grounded_questions"] == {
        "allowed": False,
        "reason_code": "no_queryable_sources",
    }


def test_source_page_filters_before_pagination_and_orders_deterministically(
    api_factory,
    monkeypatch,
) -> None:
    sources = [
        _source(4, title="Needle B", position=2),
        _source(3, title="Other", position=0),
        _source(2, title="Needle A", position=2),
        _source(1, title="Needle C", position=1),
    ]
    projected_ids: list[str] = []

    async def _sources(context):
        return sources

    async def _projection(context, projected_sources):
        projected_ids.extend(source["id"] for source in projected_sources)
        statuses = []
        for source in projected_sources:
            state = "failed" if source["id"] == "source-004" else "queryable"
            statuses.append(_status(source, state=state))
        return {
            "sources": statuses,
            "summary": {"total": 3, "queryable": 2, "processing": 0, "failed": 1},
            "partial_errors": [],
        }

    monkeypatch.setattr(sharing, "_load_recipient_workspace_sources", _sources)
    monkeypatch.setattr(sharing, "_project_recipient_source_status", _projection)
    client, _service = api_factory()

    response = client.get(
        "/api/v1/sharing/shared-with-me/42/sources",
        params={"q": "needle", "state": "queryable", "offset": 1, "limit": 1},
    )

    assert response.status_code == 200
    assert projected_ids == ["source-001", "source-002", "source-004"]
    body = response.json()
    assert [item["source_id"] for item in body["items"]] == ["source-002"]
    assert body["pagination"] == {
        "offset": 1,
        "limit": 1,
        "total": 2,
        "has_more": False,
    }
    assert body["summary"] == {
        "total": 3,
        "queryable": 2,
        "processing": 0,
        "failed": 1,
    }


@pytest.mark.parametrize(
    ("raw_url", "expected_origin", "expected_host"),
    [
        ("HTTPS://Example.COM:8443/private?q=secret#fragment", "https://example.com:8443", "example.com"),
        ("https://user:password@example.com/private", None, "example.com"),
        ("ftp://Example.COM/private/report.pdf", None, "example.com"),
        ("//Example.COM/private/report.pdf", None, "example.com"),
        ("file:///private/owner.db", None, None),
        ("/private/owner.db", None, None),
        ("javascript:alert(1)", None, None),
    ],
)
def test_source_origin_sanitization_never_exposes_paths_or_credentials(
    raw_url: str,
    expected_origin: str | None,
    expected_host: str | None,
) -> None:
    assert sharing._sanitize_recipient_source_origin(raw_url) == (
        expected_origin,
        expected_host,
    )


def test_source_page_serializes_only_recipient_safe_fields(api_factory, monkeypatch) -> None:
    unsafe = _source(
        1,
        title="Evidence",
        url="https://user:password@example.test/private/report.pdf?token=secret#raw",
        db_path="/private/owner.db",
        query="owner query",
        prompt="owner prompt",
        excerpt="owner excerpt",
    )

    async def _sources(context):
        return [unsafe]

    async def _projection(context, sources):
        return {
            "sources": [_status(unsafe)],
            "summary": {"total": 1, "queryable": 1, "processing": 0, "failed": 0},
            "partial_errors": [],
        }

    monkeypatch.setattr(sharing, "_load_recipient_workspace_sources", _sources)
    monkeypatch.setattr(sharing, "_project_recipient_source_status", _projection)
    client, _service = api_factory()

    response = client.get("/api/v1/sharing/shared-with-me/42/sources")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item == {
        "source_id": "source-001",
        "title": "Evidence",
        "source_type": "pdf",
        "origin_url": None,
        "origin_host": "example.test",
        "state": "queryable",
        "reason_code": "source_queryable",
        "citation_ready": True,
        "retrieval_ready": True,
        "position": 1,
        "added_at": "2026-08-20T18:00:00Z",
    }
    serialized = response.text.lower()
    for forbidden in (
        "media_id",
        "owner_user_id",
        "db_path",
        "/private/",
        "password",
        "token=",
        "owner query",
        "owner prompt",
        "owner excerpt",
    ):
        assert forbidden not in serialized


def test_source_search_ignores_hidden_raw_url_content(api_factory, monkeypatch) -> None:
    sources = [
        _source(
            1,
            title="Visible title",
            url=(
                "https://hidden-oracle:password@safe.example/hidden-oracle"
                "?hidden-oracle=yes#hidden-oracle"
            ),
        ),
        _source(2, title="File source", url="file:///private/hidden-oracle.db"),
        _source(3, title="Unsupported source", url="javascript:hidden-oracle"),
    ]
    projected: list[str] = []

    async def _sources(_context):
        return sources

    async def _projection(_context, selected_sources):
        projected.extend(str(source["id"]) for source in selected_sources)
        return {
            "sources": [_status(source) for source in selected_sources],
            "summary": {
                "total": len(selected_sources),
                "queryable": len(selected_sources),
                "processing": 0,
                "failed": 0,
            },
            "partial_errors": [],
        }

    monkeypatch.setattr(sharing, "_load_recipient_workspace_sources", _sources)
    monkeypatch.setattr(sharing, "_project_recipient_source_status", _projection)
    client, _service = api_factory()

    response = client.get(
        "/api/v1/sharing/shared-with-me/42/sources",
        params={"q": "hidden-oracle"},
    )

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["pagination"]["total"] == 0
    assert projected == []


def test_source_search_matches_sanitized_origin(api_factory, monkeypatch) -> None:
    source = _source(
        1,
        title="Visible title",
        url="https://user:password@Safe.Example/private?secret=yes#fragment",
    )

    async def _sources(_context):
        return [source]

    async def _projection(_context, selected_sources):
        return {
            "sources": [_status(item) for item in selected_sources],
            "summary": {
                "total": len(selected_sources),
                "queryable": len(selected_sources),
                "processing": 0,
                "failed": 0,
            },
            "partial_errors": [],
        }

    monkeypatch.setattr(sharing, "_load_recipient_workspace_sources", _sources)
    monkeypatch.setattr(sharing, "_project_recipient_source_status", _projection)
    client, _service = api_factory()

    response = client.get(
        "/api/v1/sharing/shared-with-me/42/sources",
        params={"q": "safe.example"},
    )

    assert response.status_code == 200
    assert response.json()["pagination"]["total"] == 1
    assert response.json()["items"][0]["origin_host"] == "safe.example"


def test_preview_resolves_canonical_source_membership_and_focus(api_factory, monkeypatch) -> None:
    events: list[str] = []
    service = _AccessService(events=events)

    async def _sources(context):
        events.append("sources")
        return [_source(1)]

    async def _preview(context, source, *, max_chars, chunk_limit, chunk_index):
        events.append(f"preview:{source['id']}:{chunk_index}:{max_chars}:{chunk_limit}")
        return {
            "source_id": source["id"],
            "title": source["title"],
            "source_type": source["source_type"],
            "state": "queryable",
            "reason_code": "source_queryable",
            "content_available": True,
            "preview_mode": "available",
            "unavailable_reason": None,
            "text_preview": "Bounded evidence",
            "text_total_chars": 16,
            "text_truncated": False,
            "snippets": [
                {
                    "kind": "chunk",
                    "text": "Bounded evidence",
                    "start_char": 0,
                    "end_char": 16,
                    "chunk_index": 7,
                }
            ],
            "generated_at": "2026-08-21T20:00:00+00:00",
        }

    monkeypatch.setattr(sharing, "_load_recipient_workspace_sources", _sources)
    monkeypatch.setattr(sharing, "_build_recipient_source_preview", _preview)
    client, _service = api_factory(service=service)

    response = client.get(
        "/api/v1/sharing/shared-with-me/42/sources/source-001/preview",
        params={"chunk_index": 7, "max_chars": 3000, "chunk_limit": 3},
    )

    assert response.status_code == 200
    assert events == ["access:42:9", "sources", "preview:source-001:7:3000:3"]
    serialized = response.text.lower()
    assert "media_id" not in serialized
    assert "chunk_uuid" not in serialized


@pytest.mark.parametrize(
    ("max_chars", "expected_preview_chars", "expected_focus_chars"),
    [(1, 0, 1), (12_000, 6_000, 6_000)],
)
def test_preview_text_uses_one_aggregate_budget_with_focus_first(
    max_chars: int,
    expected_preview_chars: int,
    expected_focus_chars: int,
) -> None:
    main_text = "M" * 8_000
    preview = {
        "text_preview": main_text,
        "text_truncated": True,
        "snippets": [
            {"kind": "content_excerpt", "text": main_text, "start_char": 0, "end_char": 8_000},
            {"kind": "chunk", "text": "L" * 4_000, "chunk_index": 6},
            {"kind": "chunk", "text": "F" * 6_000, "chunk_index": 7},
            {"kind": "chunk", "text": "R" * 4_000, "chunk_index": 8},
        ],
    }

    bounded = sharing._recipient_preview_text_projection(
        preview,
        max_chars=max_chars,
        focus_chunk_index=7,
    )

    emitted_texts = [
        bounded["text_preview"] or "",
        *(snippet["text"] for snippet in bounded["snippets"]),
    ]
    assert sum(len(text) for text in emitted_texts) == max_chars
    assert len(bounded["text_preview"] or "") == expected_preview_chars
    assert bounded["snippets"][0]["chunk_index"] == 7
    assert len(bounded["snippets"][0]["text"]) == expected_focus_chars
    assert all(snippet["kind"] == "chunk" for snippet in bounded["snippets"])
    assert len([text for text in emitted_texts if text]) == len(
        {text for text in emitted_texts if text}
    )
    assert bounded["text_truncated"] is True


def test_preview_missing_source_is_neutral_and_does_not_open_media(api_factory, monkeypatch) -> None:
    preview_called = False

    async def _sources(context):
        return [_source(1)]

    async def _preview(*args, **kwargs):
        nonlocal preview_called
        preview_called = True
        raise AssertionError("preview must not open media")

    monkeypatch.setattr(sharing, "_load_recipient_workspace_sources", _sources)
    monkeypatch.setattr(sharing, "_build_recipient_source_preview", _preview)
    client, _service = api_factory()

    response = client.get(
        "/api/v1/sharing/shared-with-me/42/sources/unknown/preview"
    )

    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    assert preview_called is False


def test_empty_history_does_not_create_a_thread(api_factory, monkeypatch) -> None:
    calls: list[tuple[str, Any]] = []

    async def _history(context, *, before, limit):
        calls.append(("list", (context.share_id, before, limit)))
        return StoredMessagePage(messages=(), next_before=None)

    monkeypatch.setattr(sharing, "_load_recipient_chat_history", _history)
    client, _service = api_factory()

    response = client.get("/api/v1/sharing/shared-with-me/42/chat/messages")

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": None,
        "messages": [],
        "next_before": None,
    }
    assert calls == [("list", (42, None, 30))]


def test_history_cursor_page_is_bounded_and_chronological(api_factory, monkeypatch) -> None:
    messages = (
        SharedWorkspaceStoredMessage(
            message_id="message-1",
            role="user",
            content="Question",
            created_at=datetime(2026, 8, 21, 20, 0, tzinfo=timezone.utc),
            last_modified=datetime(2026, 8, 21, 20, 0, tzinfo=timezone.utc),
        ),
        SharedWorkspaceStoredMessage(
            message_id="message-2",
            role="assistant",
            content="Answer",
            created_at=datetime(2026, 8, 21, 20, 1, tzinfo=timezone.utc),
            last_modified=datetime(2026, 8, 21, 20, 1, tzinfo=timezone.utc),
            citations=(
                {
                    "citation_id": "citation-1",
                    "source_id": "source-001",
                    "source_title": "Evidence",
                    "locator": {"chunk": 7},
                    "quote": "Bounded quote",
                    "score": 0.87,
                },
            ),
        ),
    )
    captured: dict[str, Any] = {}

    async def _history(context, *, before, limit):
        captured.update(before=before, limit=limit)
        return StoredMessagePage(messages=messages, next_before="next-opaque")

    monkeypatch.setattr(sharing, "_load_recipient_chat_history", _history)
    client, _service = api_factory()

    response = client.get(
        "/api/v1/sharing/shared-with-me/42/chat/messages",
        params={"before": "opaque", "limit": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert captured == {"before": "opaque", "limit": 2}
    assert [message["message_id"] for message in body["messages"]] == [
        "message-1",
        "message-2",
    ]
    assert body["next_before"] == "next-opaque"
    assert body["messages"][1]["citations"][0]["source_id"] == "source-001"


def test_history_rejects_cursor_from_canonical_store_decoder(
    api_factory,
    monkeypatch,
) -> None:
    store = SharedWorkspaceChatStore(SimpleNamespace(client_id="9"))

    async def _history(context, *, before, limit):
        return store.list_messages(
            share_id=context.share_id,
            before=before,
            limit=limit,
        )

    monkeypatch.setattr(sharing, "_load_recipient_chat_history", _history)
    client, _service = api_factory()

    response = client.get(
        "/api/v1/sharing/shared-with-me/42/chat/messages",
        params={"before": "not-a-canonical-cursor"},
    )

    assert response.status_code == 422
    assert response.json() == INVALID_WORKSPACE_REQUEST


def test_history_store_failure_remains_unavailable(api_factory, monkeypatch) -> None:
    async def _history(_context, *, before, limit):
        raise RuntimeError("database unavailable at /private/recipient.db")

    monkeypatch.setattr(sharing, "_load_recipient_chat_history", _history)
    client, _service = api_factory()

    response = client.get("/api/v1/sharing/shared-with-me/42/chat/messages")

    assert response.status_code == 503
    assert response.json() == UNAVAILABLE


@pytest.mark.parametrize(
    ("body", "raw"),
    [
        ({"query": "missing typed fields"}, None),
        (
            {
                "request_id": "de305d54-75b4-431b-adb2-eb6b9e546014",
                "query": "Question",
                "source_scope": {"mode": "all", "source_ids": []},
                "owner_media_id": 99,
            },
            None,
        ),
        (None, "{not-json"),
    ],
)
def test_interim_chat_validation_is_typed_and_fail_closed(
    api_factory,
    body: dict[str, Any] | None,
    raw: str | None,
) -> None:
    client, _service = api_factory()
    kwargs = {"json": body} if raw is None else {"content": raw, "headers": {"content-type": "application/json"}}

    response = client.post(
        "/api/v1/sharing/shared-with-me/42/chat",
        **kwargs,
    )

    assert response.status_code == 422
    assert response.json() == INVALID_CHAT_REQUEST


def test_valid_interim_chat_request_authorizes_then_remains_unavailable(api_factory) -> None:
    events: list[str] = []
    service = _AccessService(events=events)
    client, _service = api_factory(service=service)

    response = client.post(
        "/api/v1/sharing/shared-with-me/42/chat",
        json={
            "request_id": "de305d54-75b4-431b-adb2-eb6b9e546014",
            "query": "What evidence supports the conclusion?",
            "source_scope": {"mode": "all", "source_ids": []},
        },
    )

    assert response.status_code == 503
    assert response.json() == UNAVAILABLE
    assert events == ["access:42:9"]


def test_recipient_openapi_keeps_typed_models_and_does_not_change_clone_route(api_factory) -> None:
    client, _service = api_factory()

    schema = client.get("/openapi.json").json()

    chat = schema["paths"]["/api/v1/sharing/shared-with-me/{share_id}/chat"]["post"]
    assert chat["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/SharedWorkspaceChatRequest"
    )
    workspace = schema["paths"]["/api/v1/sharing/shared-with-me/{share_id}/workspace"]["get"]
    success_schema = workspace["responses"]["200"]["content"]["application/json"]["schema"]
    assert success_schema["$ref"].endswith("/SharedWorkspaceBootstrapResponse")
    clone = schema["paths"]["/api/v1/sharing/shared-with-me/{share_id}/clone"]["post"]
    assert clone["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/CloneWorkspaceResponse"
    )


def test_recipient_openapi_declares_only_typed_route_scoped_errors(api_factory) -> None:
    client, _service = api_factory()

    schema = client.get("/openapi.json").json()
    operations = (
        ("/api/v1/sharing/shared-with-me/{share_id}/workspace", "get"),
        ("/api/v1/sharing/shared-with-me/{share_id}/sources", "get"),
        (
            "/api/v1/sharing/shared-with-me/{share_id}/sources/{source_id}/preview",
            "get",
        ),
        ("/api/v1/sharing/shared-with-me/{share_id}/chat/messages", "get"),
        ("/api/v1/sharing/shared-with-me/{share_id}/chat", "post"),
    )
    error_statuses = {"401", "403", "404", "422", "429", "503"}

    for path, method in operations:
        operation = schema["paths"][path][method]
        assert error_statuses <= operation["responses"].keys()
        assert "HTTPValidationError" not in json.dumps(operation)
        for status_code in error_statuses:
            response_schema = operation["responses"][status_code]["content"][
                "application/json"
            ]["schema"]
            assert response_schema["$ref"].endswith("/SharedWorkspaceErrorResponse")

    wrapper = schema["components"]["schemas"]["SharedWorkspaceErrorResponse"]
    assert wrapper["additionalProperties"] is False
    assert wrapper["required"] == ["detail"]
    assert wrapper["properties"]["detail"]["$ref"].endswith(
        "/SharedWorkspaceErrorDetail"
    )

    chat = schema["paths"]["/api/v1/sharing/shared-with-me/{share_id}/chat"]["post"]
    assert "200" not in chat["responses"]
    assert chat["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/SharedWorkspaceChatRequest"
    )
    assert chat["responses"]["503"]["description"] == (
        "Shared workspace generation is not available."
    )


@pytest.mark.asyncio
async def test_optional_permission_detail_changes_only_supplied_denial(principal) -> None:
    custom = {
        "code": "sharing_permission_required",
        "message": "The sharing.read permission is required.",
        "retryable": False,
    }
    denied = principal.model_copy(update={"permissions": []})

    with pytest.raises(HTTPException) as typed_error:
        await auth_deps.require_permissions("sharing.read", detail=custom)(denied)
    with pytest.raises(HTTPException) as default_error:
        await auth_deps.require_permissions("sharing.read")(denied)

    assert typed_error.value.status_code == 403
    assert typed_error.value.detail == custom
    assert default_error.value.status_code == 403
    assert default_error.value.detail == "Permission denied: missing sharing.read"


@pytest.mark.asyncio
async def test_optional_rate_detail_preserves_metadata_and_default_behavior(monkeypatch) -> None:
    custom = {
        "code": "shared_workspace_rate_limited",
        "message": "Shared workspace requests are temporarily rate limited.",
        "retryable": True,
        "recovery_action": "retry",
    }

    async def _limited(*_args, **_kwargs):
        raise HTTPException(status_code=429, detail="legacy detail")

    monkeypatch.setattr(auth_deps, "enforce_rbac_rate_limit", _limited)
    typed = auth_deps.rbac_rate_limit("sharing.read", detail=custom)
    default = auth_deps.rbac_rate_limit("sharing.read")
    request = SimpleNamespace()

    with pytest.raises(HTTPException) as typed_error:
        await typed(request, None)
    with pytest.raises(HTTPException) as default_error:
        await default(request, None)

    assert typed._tldw_rate_limit_resource == "sharing.read"
    assert default._tldw_rate_limit_resource == "sharing.read"
    assert typed_error.value.detail == custom
    assert default_error.value.detail == "legacy detail"


def test_recipient_routes_have_isolated_permission_and_rate_dependencies() -> None:
    recipient_paths = {
        "/sharing/shared-with-me/{share_id}/workspace": "sharing.read",
        "/sharing/shared-with-me/{share_id}/sources": "sharing.read",
        "/sharing/shared-with-me/{share_id}/sources/{source_id}/preview": "sharing.read",
        "/sharing/shared-with-me/{share_id}/chat/messages": "sharing.read",
        "/sharing/shared-with-me/{share_id}/chat": "sharing.read",
    }
    routes = {route.path: route for route in sharing.router.routes if hasattr(route, "dependant")}

    for path, expected_resource in recipient_paths.items():
        route = routes[path]
        dependencies = [dependency.call for dependency in route.dependant.dependencies]
        resources = [
            dependency._tldw_rate_limit_resource
            for dependency in dependencies
            if hasattr(dependency, "_tldw_rate_limit_resource")
        ]
        assert resources == [expected_resource]
        assert any(
            any(
                child.call is auth_deps.get_auth_principal
                for child in dependency.dependencies
            )
            for dependency in route.dependant.dependencies
        )
    clone = routes["/sharing/shared-with-me/{share_id}/clone"]
    assert not isinstance(clone, sharing.SharedWorkspaceRecipientRoute)
