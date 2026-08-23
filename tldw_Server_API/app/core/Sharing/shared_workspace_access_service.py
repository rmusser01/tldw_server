"""Authoritative authorization and owner-workspace resolution for shared reads."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import (
    SharedWorkspaceRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.exceptions import (
    SharedWorkspaceAccessError as SharedWorkspaceAccessError,
)
from tldw_Server_API.app.core.exceptions import (
    SharedWorkspaceNotFound,
    SharedWorkspaceUnavailable,
)

_OWNER_DISPLAY_NAME_MAX_CHARS = 128
_OWNER_DISPLAY_NAME_FALLBACK = "Workspace owner"


@dataclass(frozen=True)
class SharedWorkspaceAccessContext:
    """Internal authorized context for recipient shared-workspace operations."""

    share_id: int
    workspace_id: str
    owner_user_id: int
    recipient_user_id: int
    share_scope_type: Literal["team", "org"]
    share_scope_id: int
    access_level: str
    allow_clone: bool
    owner_display_name: str
    shared_at: str | None
    workspace: dict[str, Any]
    policy_actions: dict[str, dict[str, Any]]


def _recipient_policy_actions() -> dict[str, dict[str, Any]]:
    """Return the current fail-closed recipient capability policy."""
    return {
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


def _sanitize_owner_display_name(value: Any) -> str:
    raw = str(value or "")
    printable = "".join(character if character.isprintable() else " " for character in raw)
    normalized = " ".join(printable.split()).strip()
    bounded = normalized[:_OWNER_DISPLAY_NAME_MAX_CHARS].strip()
    return bounded or _OWNER_DISPLAY_NAME_FALLBACK


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class SharedWorkspaceAccessService:
    """Resolve an authorized share before opening any owner content database."""

    def __init__(
        self,
        share_repo: SharedWorkspaceRepo,
        users_repo: AuthnzUsersRepo,
        owner_chacha_loader: Callable[[int], Awaitable[Any]],
    ) -> None:
        self._share_repo = share_repo
        self._users_repo = users_repo
        self._owner_chacha_loader = owner_chacha_loader

    async def resolve(
        self,
        *,
        share_id: int,
        recipient_user_id: int,
    ) -> SharedWorkspaceAccessContext:
        """Authorize current membership, then resolve the active owner workspace."""
        try:
            share = await self._share_repo.get_active_share_for_user(
                int(share_id),
                int(recipient_user_id),
            )
        except Exception as exc:
            raise SharedWorkspaceUnavailable() from exc
        if share is None:
            raise SharedWorkspaceNotFound()

        try:
            owner_user_id = int(share["owner_user_id"])
            workspace_id = str(share["workspace_id"])
            scope_type = str(share["share_scope_type"])
            scope_id = int(share["share_scope_id"])
            resolved_share_id = int(share["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SharedWorkspaceUnavailable() from exc
        if scope_type not in {"team", "org"} or not workspace_id:
            raise SharedWorkspaceUnavailable()

        try:
            owner = await self._users_repo.get_user_by_id(owner_user_id)
        except Exception as exc:
            raise SharedWorkspaceUnavailable() from exc
        owner_display_name = _sanitize_owner_display_name(
            owner.get("username") if owner else None
        )

        try:
            owner_db = await self._owner_chacha_loader(owner_user_id)
            if owner_db is None:
                raise SharedWorkspaceUnavailable()
            workspace = owner_db.get_workspace(workspace_id)
        except SharedWorkspaceUnavailable:
            raise
        except Exception as exc:
            raise SharedWorkspaceUnavailable() from exc
        if workspace is None or _is_truthy(workspace.get("deleted")) or _is_truthy(
            workspace.get("archived")
        ):
            raise SharedWorkspaceNotFound()

        shared_at_raw = share.get("created_at")
        shared_at = str(shared_at_raw) if shared_at_raw is not None else None
        return SharedWorkspaceAccessContext(
            share_id=resolved_share_id,
            workspace_id=workspace_id,
            owner_user_id=owner_user_id,
            recipient_user_id=int(recipient_user_id),
            share_scope_type=scope_type,
            share_scope_id=scope_id,
            access_level=str(share.get("access_level") or ""),
            allow_clone=bool(share.get("allow_clone")),
            owner_display_name=owner_display_name,
            shared_at=shared_at,
            workspace=dict(workspace),
            policy_actions=_recipient_policy_actions(),
        )
