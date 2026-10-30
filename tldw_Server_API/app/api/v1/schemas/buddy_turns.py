"""Accepted-turn input and metadata-only status responses."""

from typing import Literal

from tldw_Server_API.app.api.v1.schemas.buddies import BuddyModel

# Core-owned since TASK-13950.15; re-exported for the API layer.
from tldw_Server_API.app.core.Buddy.buddy_schemas import (  # noqa: F401 - re-exported (TASK-13950.15)
    BuddyTurnCreate,
)


class BuddyTurn(BuddyModel):
    """Return metadata-only lifecycle status for one principal-owned accepted turn.

    No message body or credentials are returned. A completed result references
    its exact persisted message; stopped/failed turns may carry a safe error code."""

    id: str
    client_slot: str
    client_request_id: str
    conversation_id: str
    conversation_title: str
    workspace_id: str | None
    attachment_version: int
    status: Literal["queued", "running", "completed", "failed", "stopped"]
    result_message_id: str | None
    error_code: str | None
    created_at: str
    updated_at: str


class BuddyTurnList(BuddyModel):
    """Return a bounded turn ledger page with the applied limit and offset."""

    turns: list[BuddyTurn]
    limit: int
    offset: int
