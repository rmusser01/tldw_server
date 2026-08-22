"""Authorization-order tests for recipient shared workspace reads."""
from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
    SharedWorkspaceAccessService,
    SharedWorkspaceNotFound,
    SharedWorkspaceUnavailable,
)

pytestmark = pytest.mark.unit


def _share(**overrides: Any) -> dict[str, Any]:
    share = {
        "id": 42,
        "workspace_id": "workspace-alpha",
        "owner_user_id": 7,
        "share_scope_type": "team",
        "share_scope_id": 11,
        "access_level": "full_edit",
        "allow_clone": True,
        "created_at": "2026-08-20T18:00:00+00:00",
    }
    share.update(overrides)
    return share


class _ShareRepo:
    def __init__(
        self,
        events: list[str],
        *,
        result: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._events = events
        self._result = result
        self._error = error

    async def get_active_share_for_user(
        self,
        share_id: int,
        user_id: int,
    ) -> dict[str, Any] | None:
        self._events.append(f"share:{share_id}:{user_id}")
        if self._error is not None:
            raise self._error
        return dict(self._result) if self._result is not None else None


class _UsersRepo:
    def __init__(
        self,
        events: list[str],
        *,
        result: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._events = events
        self._result = result
        self._error = error

    async def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        self._events.append(f"user:{user_id}")
        if self._error is not None:
            raise self._error
        return dict(self._result) if self._result is not None else None


class _OwnerDB:
    def __init__(
        self,
        events: list[str],
        *,
        workspace: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._events = events
        self._workspace = workspace
        self._error = error

    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        self._events.append(f"workspace:{workspace_id}")
        if self._error is not None:
            raise self._error
        return dict(self._workspace) if self._workspace is not None else None


def _service(
    *,
    share: dict[str, Any] | None = None,
    share_error: Exception | None = None,
    user: dict[str, Any] | None = None,
    user_error: Exception | None = None,
    workspace: dict[str, Any] | None = None,
    owner_db_error: Exception | None = None,
    loader_error: Exception | None = None,
) -> tuple[SharedWorkspaceAccessService, list[str]]:
    events: list[str] = []
    share_repo = _ShareRepo(events, result=share, error=share_error)
    users_repo = _UsersRepo(events, result=user, error=user_error)
    owner_db = _OwnerDB(
        events,
        workspace=workspace,
        error=owner_db_error,
    )

    async def load_owner(owner_user_id: int) -> _OwnerDB:
        events.append(f"loader:{owner_user_id}")
        if loader_error is not None:
            raise loader_error
        return owner_db

    return SharedWorkspaceAccessService(share_repo, users_repo, load_owner), events


@pytest.mark.asyncio
async def test_resolve_authorizes_before_loading_owner_data_and_projects_recipient_policy() -> None:
    service, events = _service(
        share=_share(),
        user={"id": 7, "username": "Research owner"},
        workspace={
            "id": "workspace-alpha",
            "name": "Evidence review",
            "description": "Review set",
            "archived": False,
        },
    )

    context = await service.resolve(share_id=42, recipient_user_id=9)

    assert events == [
        "share:42:9",
        "user:7",
        "loader:7",
        "workspace:workspace-alpha",
    ]
    assert context.share_id == 42
    assert context.workspace_id == "workspace-alpha"
    assert context.owner_user_id == 7
    assert context.recipient_user_id == 9
    assert context.share_scope_type == "team"
    assert context.share_scope_id == 11
    assert context.access_level == "full_edit"
    assert context.allow_clone is True
    assert context.owner_display_name == "Research owner"
    assert context.shared_at == "2026-08-20T18:00:00+00:00"
    assert context.workspace["name"] == "Evidence review"
    assert context.policy_actions == {
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
    }


@pytest.mark.asyncio
async def test_owner_uses_same_deny_by_default_recipient_projection() -> None:
    service, _events = _service(
        share=_share(),
        user={"id": 7, "username": "owner"},
        workspace={"id": "workspace-alpha", "archived": False},
    )

    context = await service.resolve(share_id=42, recipient_user_id=7)

    assert context.recipient_user_id == context.owner_user_id
    assert context.policy_actions["edit_workspace"] == {
        "allowed": False,
        "reason_code": "shared_write_not_available",
    }
    assert context.policy_actions["clone_workspace"] == {
        "allowed": False,
        "reason_code": "clone_deferred",
    }


@pytest.mark.asyncio
async def test_denied_share_does_not_lookup_owner_or_open_owner_database() -> None:
    service, events = _service(
        share=None,
        user={"id": 7, "username": "must-not-load"},
        workspace={"id": "workspace-alpha"},
    )

    with pytest.raises(SharedWorkspaceNotFound):
        await service.resolve(share_id=42, recipient_user_id=9)

    assert events == ["share:42:9"]


@pytest.mark.asyncio
async def test_authoritative_repository_failure_is_unavailable_without_owner_disclosure() -> None:
    service, events = _service(
        share_error=RuntimeError("sensitive authnz backend detail for share 42"),
        user={"id": 7, "username": "must-not-load"},
        workspace={"id": "workspace-alpha"},
    )

    with pytest.raises(SharedWorkspaceUnavailable) as exc_info:
        await service.resolve(share_id=42, recipient_user_id=9)

    assert "sensitive" not in str(exc_info.value)
    assert "42" not in str(exc_info.value)
    assert events == ["share:42:9"]


@pytest.mark.asyncio
async def test_owner_user_lookup_failure_does_not_open_owner_database() -> None:
    service, events = _service(
        share=_share(),
        user_error=RuntimeError("sensitive owner lookup failure"),
        workspace={"id": "workspace-alpha"},
    )

    with pytest.raises(SharedWorkspaceUnavailable) as exc_info:
        await service.resolve(share_id=42, recipient_user_id=9)

    assert "sensitive" not in str(exc_info.value)
    assert events == ["share:42:9", "user:7"]


@pytest.mark.asyncio
async def test_owner_database_failure_is_unavailable_after_authorization() -> None:
    service, events = _service(
        share=_share(),
        user={"id": 7, "username": "owner"},
        loader_error=RuntimeError("/private/owner/database/path"),
    )

    with pytest.raises(SharedWorkspaceUnavailable) as exc_info:
        await service.resolve(share_id=42, recipient_user_id=9)

    assert "/private/owner/database/path" not in str(exc_info.value)
    assert events == ["share:42:9", "user:7", "loader:7"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workspace",
    [None, {"id": "workspace-alpha", "archived": True}],
    ids=["deleted", "archived"],
)
async def test_missing_or_inactive_workspace_is_neutral_not_found(
    workspace: dict[str, Any] | None,
) -> None:
    service, events = _service(
        share=_share(),
        user={"id": 7, "username": "owner"},
        workspace=workspace,
    )

    with pytest.raises(SharedWorkspaceNotFound) as exc_info:
        await service.resolve(share_id=42, recipient_user_id=9)

    assert "workspace-alpha" not in str(exc_info.value)
    assert events[-1] == "workspace:workspace-alpha"


@pytest.mark.asyncio
async def test_owner_display_name_is_sanitized_bounded_and_has_safe_fallback() -> None:
    unsafe_username = "  Research\n\tOwner\u200b  " + ("x" * 200)
    service, _events = _service(
        share=_share(),
        user={"id": 7, "username": unsafe_username},
        workspace={"id": "workspace-alpha", "archived": False},
    )

    context = await service.resolve(share_id=42, recipient_user_id=9)

    assert context.owner_display_name.startswith("Research Owner")
    assert "\n" not in context.owner_display_name
    assert "\t" not in context.owner_display_name
    assert "\u200b" not in context.owner_display_name
    assert len(context.owner_display_name) == 128

    fallback_service, _fallback_events = _service(
        share=_share(),
        user={"id": 7, "username": "\n\t\u200b"},
        workspace={"id": "workspace-alpha", "archived": False},
    )
    fallback = await fallback_service.resolve(share_id=42, recipient_user_id=9)
    assert fallback.owner_display_name == "Workspace owner"
