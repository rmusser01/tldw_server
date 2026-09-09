"""Principal-owned Buddy snapshots and versioned client attachment slots."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


from tldw_Server_API.app.core.exceptions import BuddyConflictError, BuddyNotFoundError

BUDDY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS buddy_profiles (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    optional_persona_id TEXT,
    display_mode TEXT NOT NULL CHECK(display_mode IN ('dynamic', 'static')),
    manifest_json TEXT NOT NULL,
    attribution_json TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT FALSE CHECK(deleted IN (FALSE, TRUE)),
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_modified TEXT NOT NULL,
    UNIQUE(id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_buddy_profiles_owner ON buddy_profiles(user_id, deleted, id);
CREATE TABLE IF NOT EXISTS buddy_assets (
    id TEXT PRIMARY KEY,
    buddy_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    byte_size INTEGER NOT NULL CHECK(byte_size > 0),
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    checksum_sha256 TEXT NOT NULL,
    FOREIGN KEY(buddy_id, user_id) REFERENCES buddy_profiles(id, user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_buddy_assets_owner ON buddy_assets(user_id, buddy_id);
CREATE TABLE IF NOT EXISTS buddy_attachments (
    user_id TEXT NOT NULL,
    client_slot TEXT NOT NULL,
    buddy_id TEXT,
    scope_type TEXT CHECK(scope_type IN ('conversation', 'workspace')),
    scope_id TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY(user_id, client_slot),
    FOREIGN KEY(buddy_id, user_id) REFERENCES buddy_profiles(id, user_id),
    CHECK((buddy_id IS NULL AND scope_type IS NULL AND scope_id IS NULL)
       OR (buddy_id IS NOT NULL AND scope_type IS NOT NULL AND scope_id IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS buddy_result_acknowledgements (
    user_id TEXT NOT NULL,
    client_slot TEXT NOT NULL,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    result_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    acknowledged_at TEXT NOT NULL,
    PRIMARY KEY(user_id, client_slot, conversation_id, result_message_id)
);
"""


class BuddyRepository:
    """Keep immutable art separate from editable identity and attachment state."""

    def __init__(self, db: CharactersRAGDB, user_id: str) -> None:
        self.db = db
        self.user_id = user_id

    @staticmethod
    def _profile(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        item["manifest"] = json.loads(item.pop("manifest_json"))
        item["attribution"] = json.loads(item.pop("attribution_json"))
        return item

    def get(self, buddy_id: str) -> dict[str, Any] | None:
        return self._profile(
            self.db.execute_query(
                "SELECT * FROM buddy_profiles WHERE id = ? AND user_id = ? AND deleted = 0",
                (buddy_id, self.user_id),
            ).fetchone()
        )

    def list_profiles(self, *, limit: int, offset: int) -> list[dict[str, Any]]:
        rows = self.db.execute_query(
            "SELECT * FROM buddy_profiles WHERE user_id = ? AND deleted = 0 ORDER BY created_at, id LIMIT ? OFFSET ?",
            (self.user_id, limit, offset),
        ).fetchall()
        return [self._profile(row) for row in rows]

    def assets(self, buddy_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.execute_query(
                "SELECT * FROM buddy_assets WHERE buddy_id = ? AND user_id = ? ORDER BY id",
                (buddy_id, self.user_id),
            ).fetchall()
        ]

    def create(self, profile: dict[str, Any], assets: list[dict[str, Any]]) -> None:
        """Publish the complete validated snapshot in one transaction."""
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO buddy_profiles (id, user_id, name, optional_persona_id, display_mode, manifest_json, attribution_json, created_at, last_modified) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    profile["id"],
                    self.user_id,
                    profile["name"],
                    profile["optional_persona_id"],
                    profile["display_mode"],
                    json.dumps(profile["manifest"]),
                    json.dumps(profile["attribution"]),
                    profile["created_at"],
                    profile["created_at"],
                ),
            )
            for asset in assets:
                conn.execute(
                    "INSERT INTO buddy_assets (id, buddy_id, user_id, mime_type, byte_size, width, height, checksum_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        asset["id"],
                        profile["id"],
                        self.user_id,
                        asset["mime_type"],
                        asset["byte_size"],
                        asset["width"],
                        asset["height"],
                        asset["checksum_sha256"],
                    ),
                )

    def update(self, buddy_id: str, *, expected_version: int, changes: dict[str, Any], timestamp: str) -> None:
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM buddy_profiles WHERE id = ? AND user_id = ? AND deleted = 0", (buddy_id, self.user_id)
            ).fetchone()
            if row is None:
                raise BuddyNotFoundError("Buddy not found")
            current = dict(row)
            current.update(changes)
            result = conn.execute(
                "UPDATE buddy_profiles SET name = ?, optional_persona_id = ?, display_mode = ?, deleted = ?, last_modified = ?, version = version + 1 WHERE id = ? AND user_id = ? AND version = ? AND deleted = 0",
                (
                    current["name"],
                    current["optional_persona_id"],
                    current["display_mode"],
                    bool(current["deleted"]),
                    timestamp,
                    buddy_id,
                    self.user_id,
                    expected_version,
                ),
            )
            if result.rowcount != 1:
                raise BuddyConflictError("Buddy version changed")

    def attachment(self, client_slot: str) -> dict[str, Any]:
        row = self.db.execute_query(
            "SELECT client_slot, buddy_id, scope_type, scope_id, version FROM buddy_attachments WHERE user_id = ? AND client_slot = ?",
            (self.user_id, client_slot),
        ).fetchone()
        return (
            dict(row)
            if row is not None
            else {"client_slot": client_slot, "buddy_id": None, "scope_type": None, "scope_id": None, "version": 0}
        )

    def set_attachment(self, client_slot: str, *, expected_version: int, attachment: dict[str, Any] | None) -> None:
        """CAS a slot; retain its revision after detaching to fence stale clients."""
        value = attachment or {}
        with self.db.transaction() as conn:
            if expected_version == 0:
                result = conn.execute(
                    "INSERT INTO buddy_attachments (user_id, client_slot, buddy_id, scope_type, scope_id) VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id, client_slot) DO NOTHING",
                    (self.user_id, client_slot, value.get("buddy_id"), value.get("scope_type"), value.get("scope_id")),
                )
            else:
                result = conn.execute(
                    "UPDATE buddy_attachments SET buddy_id = ?, scope_type = ?, scope_id = ?, version = version + 1 WHERE user_id = ? AND client_slot = ? AND version = ?",
                    (
                        value.get("buddy_id"),
                        value.get("scope_type"),
                        value.get("scope_id"),
                        self.user_id,
                        client_slot,
                        expected_version,
                    ),
                )
            if result.rowcount != 1:
                raise BuddyConflictError("Attachment version changed")

    def latest_results(self, client_slot: str, conversation_ids: list[str]) -> list[dict[str, Any]]:
        """Batch latest stored assistant results using the conversation's identity."""
        if not conversation_ids:
            return []
        placeholders = ",".join("?" for _ in conversation_ids)
        # Only placeholder count is interpolated; all identities remain parameters.
        query = f"""
            WITH ranked AS (
                SELECT m.id, m.conversation_id, m.timestamp AS created_at,
                       substr(COALESCE(m.content, ''), 1, 2000) AS content,
                       ROW_NUMBER() OVER (
                           PARTITION BY m.conversation_id
                           ORDER BY m.timestamp DESC, m.last_modified DESC, m.id DESC
                       ) AS result_rank
                  FROM messages m
                  JOIN conversations c ON c.id = m.conversation_id
                  JOIN buddy_attachments binding
                    ON binding.user_id = c.client_id AND binding.client_slot = ?
                   AND ((binding.scope_type = 'conversation' AND binding.scope_id = c.id)
                        OR (binding.scope_type = 'workspace' AND c.scope_type = 'workspace' AND binding.scope_id = c.workspace_id))
                  JOIN buddy_profiles buddy ON buddy.id = binding.buddy_id AND buddy.user_id = binding.user_id AND buddy.deleted = 0
                  LEFT JOIN workspaces workspace ON workspace.id = c.workspace_id
                  LEFT JOIN character_cards cc ON cc.id = c.character_id
                  LEFT JOIN persona_profiles p
                    ON c.assistant_kind = 'persona' AND p.id = c.assistant_id AND p.user_id = c.client_id
                 WHERE c.client_id = ? AND c.deleted = 0 AND m.deleted = 0
                   AND (c.scope_type = 'global' OR (workspace.id IS NOT NULL AND workspace.deleted = 0))
                   AND c.id IN ({placeholders})
                   AND lower(trim(m.sender)) NOT IN ('user', 'human', 'system', 'tool')
                   AND (lower(trim(m.sender)) IN ('assistant', 'bot', 'ai', 'character')
                        OR lower(trim(m.sender)) = lower(trim(cc.name))
                        OR lower(trim(m.sender)) = lower(trim(p.name)))
            )
            SELECT r.*, EXISTS (
                SELECT 1 FROM buddy_result_acknowledgements a
                 WHERE a.user_id = ? AND a.client_slot = ?
                   AND a.conversation_id = r.conversation_id AND a.result_message_id = r.id
            ) AS acknowledged
              FROM ranked r WHERE r.result_rank = 1
        """  # nosec B608 - generated placeholders only; values are bound.
        return [
            dict(row)
            for row in self.db.execute_query(
                query, (client_slot, self.user_id, *conversation_ids, self.user_id, client_slot)
            ).fetchall()
        ]

    def acknowledge(self, client_slot: str, *, conversation_id: str, message_id: str, timestamp: str) -> None:
        """Acknowledge only an exact owned persisted assistant result."""
        with self.db.transaction() as conn:
            result = conn.execute(
                """
                INSERT INTO buddy_result_acknowledgements
                    (user_id, client_slot, conversation_id, result_message_id, acknowledged_at)
                SELECT ?, ?, c.id, m.id, ?
                  FROM messages m JOIN conversations c ON c.id = m.conversation_id
                  JOIN buddy_attachments binding
                    ON binding.user_id = c.client_id AND binding.client_slot = ?
                   AND ((binding.scope_type = 'conversation' AND binding.scope_id = c.id)
                        OR (binding.scope_type = 'workspace' AND c.scope_type = 'workspace' AND binding.scope_id = c.workspace_id))
                  JOIN buddy_profiles buddy ON buddy.id = binding.buddy_id AND buddy.user_id = binding.user_id AND buddy.deleted = 0
                  LEFT JOIN workspaces workspace ON workspace.id = c.workspace_id
                  LEFT JOIN character_cards cc ON cc.id = c.character_id
                  LEFT JOIN persona_profiles p
                    ON c.assistant_kind = 'persona' AND p.id = c.assistant_id AND p.user_id = c.client_id
                 WHERE c.client_id = ? AND c.id = ? AND m.id = ? AND c.deleted = 0 AND m.deleted = 0
                   AND (c.scope_type = 'global' OR (workspace.id IS NOT NULL AND workspace.deleted = 0))
                   AND lower(trim(m.sender)) NOT IN ('user', 'human', 'system', 'tool')
                   AND (lower(trim(m.sender)) IN ('assistant', 'bot', 'ai', 'character')
                        OR lower(trim(m.sender)) = lower(trim(cc.name))
                        OR lower(trim(m.sender)) = lower(trim(p.name)))
                ON CONFLICT(user_id, client_slot, conversation_id, result_message_id)
                DO UPDATE SET acknowledged_at = excluded.acknowledged_at
                """,
                (self.user_id, client_slot, timestamp, client_slot, self.user_id, conversation_id, message_id),
            )
            if result.rowcount != 1:
                raise BuddyNotFoundError("Result not found")
