"""Metadata-only admission ledger and transaction fences for Buddy turns."""

from __future__ import annotations

import time
from typing import Any

from tldw_Server_API.app.core.DB_Management.Buddy_DB import BuddyConflictError, BuddyNotFoundError

BUDDY_TURNS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS buddy_turn_owners (
    user_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS buddy_turns (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    client_slot TEXT NOT NULL,
    client_request_id TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    conversation_title TEXT NOT NULL,
    conversation_version INTEGER NOT NULL,
    workspace_id TEXT,
    attachment_version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'completed', 'failed', 'stopped')),
    result_message_id TEXT,
    error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(user_id, client_request_id)
);
CREATE INDEX IF NOT EXISTS idx_buddy_turns_owner_slot
    ON buddy_turns(user_id, client_slot, created_at, id);
"""

BUDDY_TURNS_TABLES = ("buddy_turn_owners", "buddy_turns")


class BuddyPublicationRevokedError(RuntimeError):
    """The accepted turn no longer has permission to publish messages."""


class BuddyRuntimeBusyError(RuntimeError):
    """Another process currently owns this principal's in-memory queue."""


class BuddyTurnRepository:
    """Serialize cancellation, ownership transfer and message publication in SQL."""

    def __init__(self, db: Any, user_id: str) -> None:
        self.db = db
        self.user_id = user_id

    def get(self, turn_id: str) -> dict[str, Any]:
        row = self.db.execute_query(
            "SELECT * FROM buddy_turns WHERE user_id = ? AND id = ?", (self.user_id, turn_id)
        ).fetchone()
        if row is None:
            raise BuddyNotFoundError("Turn not found")
        return dict(row)

    def by_key(self, client_request_id: str) -> dict[str, Any] | None:
        row = self.db.execute_query(
            "SELECT * FROM buddy_turns WHERE user_id = ? AND client_request_id = ?",
            (self.user_id, client_request_id),
        ).fetchone()
        return dict(row) if row is not None else None

    def list_turns(
        self, client_slot: str, limit: int, offset: int, *, active_only: bool = False
    ) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.execute_query(
                "SELECT * FROM buddy_turns WHERE user_id = ? AND client_slot = ? AND (? = 0 OR status IN ('queued', 'running')) ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (self.user_id, client_slot, int(active_only), limit, offset),
            ).fetchall()
        ]

    def claim(self, owner_id: str, *, lease_seconds: float = 30) -> None:
        """A live process keeps its queue; expired owners fail terminal, never retry."""
        now = time.time()
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO buddy_turn_owners(user_id, owner_id, expires_at) VALUES (?, ?, ?) ON CONFLICT(user_id) DO NOTHING",
                (self.user_id, owner_id, now + lease_seconds),
            )
            # Serialize with publication and sample expiry only after acquiring
            # the owner lock. Even this same process can have lost its lease.
            conn.execute("UPDATE buddy_turn_owners SET owner_id = owner_id WHERE user_id = ?", (self.user_id,))
            now = time.time()
            self._expire_interrupted(conn, now)
            claimed = conn.execute(
                "UPDATE buddy_turn_owners SET owner_id = ?, expires_at = ? WHERE user_id = ? AND (owner_id = ? OR expires_at < ?)",
                (owner_id, now + lease_seconds, self.user_id, owner_id, now),
            )
            if claimed.rowcount != 1:
                raise BuddyRuntimeBusyError("Buddy work is owned by another server process")

    def expire_interrupted(self) -> None:
        """Status reads expose expired interrupted work even before a new send."""
        with self.db.transaction() as conn:
            conn.execute("UPDATE buddy_turn_owners SET owner_id = owner_id WHERE user_id = ?", (self.user_id,))
            self._expire_interrupted(conn, time.time())

    def _expire_interrupted(self, conn: Any, now: float) -> None:
        """Retire old work, preserving replies already committed by Chat."""
        conn.execute(
            "UPDATE buddy_turns SET status = CASE WHEN result_message_id IS NULL THEN 'failed' ELSE 'completed' END, error_code = CASE WHEN result_message_id IS NULL THEN 'interrupted_unknown' ELSE NULL END, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND status IN ('queued', 'running') AND NOT EXISTS (SELECT 1 FROM buddy_turn_owners o WHERE o.user_id = buddy_turns.user_id AND o.owner_id = buddy_turns.owner_id AND o.expires_at >= ?)",
            (self.user_id, now),
        )

    def create(self, value: dict[str, Any], *, check_attachment: bool = False) -> dict[str, Any]:
        with self.db.transaction() as conn:
            if check_attachment:
                attachment = conn.execute(
                    "UPDATE buddy_attachments SET version = version WHERE user_id = ? AND client_slot = ? AND version = ? AND buddy_id IS NOT NULL",
                    (self.user_id, value["client_slot"], value["attachment_version"]),
                )
                if attachment.rowcount != 1:
                    raise BuddyConflictError("Attachment version changed before acceptance")
            result = conn.execute(
                "INSERT INTO buddy_turns(id, user_id, owner_id, client_slot, client_request_id, request_digest, conversation_id, conversation_title, conversation_version, workspace_id, attachment_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?) ON CONFLICT(user_id, client_request_id) DO NOTHING",
                (
                    value["id"],
                    self.user_id,
                    value["owner_id"],
                    value["client_slot"],
                    value["client_request_id"],
                    value["request_digest"],
                    value["conversation_id"],
                    value["conversation_title"],
                    value["conversation_version"],
                    value["workspace_id"],
                    value["attachment_version"],
                    value["created_at"],
                    value["created_at"],
                ),
            )
            if result.rowcount != 1:
                raise BuddyConflictError("This request was already accepted")
        return self.get(value["id"])

    def transition(self, turn_id: str, status: str, *, error_code: str | None = None) -> dict[str, Any]:
        """A runtime failure cannot replace an atomically committed reply."""
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE buddy_turns SET status = CASE WHEN ? = 'failed' AND result_message_id IS NOT NULL THEN 'completed' ELSE ? END, error_code = CASE WHEN ? = 'failed' AND result_message_id IS NOT NULL THEN NULL ELSE ? END, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ? AND status IN ('queued', 'running')",
                (status, status, status, error_code, turn_id, self.user_id),
            )
        return self.get(turn_id)

    def stop(self, turn_id: str) -> dict[str, Any]:
        """A committed reply wins; otherwise revoke all subsequent publication."""
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE buddy_turns SET status = CASE WHEN result_message_id IS NULL THEN 'stopped' ELSE 'completed' END, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ? AND status IN ('queued', 'running')",
                (turn_id, self.user_id),
            )
        return self.get(turn_id)

    def assert_publication(self, conn: Any, turn: dict[str, Any], conversation_id: str) -> None:
        """Lock authority and target in the same transaction as the Chat write."""
        if conversation_id != turn["conversation_id"] or str(self.db.client_id) != self.user_id:
            raise BuddyPublicationRevokedError("Target changed")
        owner = conn.execute(
            "UPDATE buddy_turn_owners SET owner_id = owner_id WHERE user_id = ? AND owner_id = ? AND expires_at >= ?",
            (self.user_id, turn["owner_id"], time.time()),
        )
        active = conn.execute(
            "UPDATE buddy_turns SET status = status WHERE user_id = ? AND id = ? AND owner_id = ? AND status = 'running'",
            (self.user_id, turn["id"], turn["owner_id"]),
        )
        if owner.rowcount != 1 or active.rowcount != 1:
            raise BuddyPublicationRevokedError("Turn was stopped or its owner expired")
        workspace_id = turn["workspace_id"]
        if workspace_id:
            workspace = conn.execute("UPDATE workspaces SET id = id WHERE id = ? AND deleted = FALSE", (workspace_id,))
            if workspace.rowcount != 1:
                raise BuddyPublicationRevokedError("Workspace unavailable")
        target = conn.execute(
            "UPDATE conversations SET id = id WHERE id = ? AND client_id = ? AND deleted = FALSE AND version = ? AND ((scope_type = 'workspace' AND workspace_id = ?) OR (scope_type = 'global' AND CAST(? AS TEXT) IS NULL))",
            (conversation_id, self.user_id, turn["conversation_version"], workspace_id, workspace_id),
        )
        if target.rowcount != 1:
            raise BuddyPublicationRevokedError("Conversation changed or became unavailable")

    def record_message(self, conn: Any, turn_id: str, role: str, message_id: str | None) -> None:
        if role == "assistant" and message_id:
            conn.execute(
                "UPDATE buddy_turns SET result_message_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ? AND status = 'running'",
                (message_id, turn_id, self.user_id),
            )
