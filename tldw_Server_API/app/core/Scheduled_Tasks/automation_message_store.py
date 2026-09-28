"""Encrypted, owner-scoped store for automation agent messages (ADR-184).

Decision 1A: ``agent_task`` definitions keep the ``metadata_only`` redaction
in the scheduled-tasks DB (``message_redacted``/``message_ref``/
``message_preview`` metadata replaces the raw message); the raw message
lives ONLY here — a separate per-owner database file, encrypted with the
server's secret-material envelope, TTL-bounded. Backups and exports of the
scheduled-tasks DBs therefore never carry raw prompts.

Read path discipline: only the agent-task executor resolves a ref, in
memory, at dispatch. Logs, API projections, audit events, and run rows
keep the ref, never the payload. An unresolvable ref (purged TTL, missing
store, decrypt failure) is an honest failed run, never a silent skip.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Security.crypto import (
    decrypt_json_blob_with_key,
    encrypt_json_blob_with_key,
)

_STORE_DB_NAME = "automation_message_store.db"

#: v1 retention: a fixed TTL, refreshed on each successful resolve
#: (last-access semantics). Reference-liveness reclaim — delete only when no
#: non-archived definition cites the ref — is a follow-up; the fixed window
#: bounds orphaned preview writes in the meantime.
DEFAULT_MESSAGE_TTL_DAYS = 30


def _message_store_keys() -> tuple[str | None, str | None]:
    """Return the message-store envelope keys: dedicated, else BYOK fallback."""
    settings = get_settings()
    primary = (
        getattr(settings, "AUTOMATION_MESSAGE_ENCRYPTION_KEY", None)
        or getattr(settings, "BYOK_ENCRYPTION_KEY", None)
    )
    secondary = getattr(settings, "BYOK_SECONDARY_ENCRYPTION_KEY", None)
    return primary, secondary


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AutomationMessageStore:
    """Per-owner encrypted message store keyed by ``message_ref``."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the store at ``db_path`` (schema created lazily)."""
        self.db_path = Path(db_path)
        self._schema_ready = False

    @classmethod
    def for_user(cls, user_id: int) -> AutomationMessageStore:
        """Return the per-owner store, sibling directory of the scheduled-tasks DB."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        return cls(user_dir / _STORE_DB_NAME)

    # --- schema ---------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create the store table if missing (idempotent)."""
        if self._schema_ready:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS automation_messages (
                    message_ref TEXT PRIMARY KEY,
                    owner_id INTEGER NOT NULL,
                    encrypted_blob TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_automation_messages_owner"
                " ON automation_messages (owner_id)"
            )
        self._schema_ready = True

    # --- write / read ----------------------------------------------------

    def store_message(
        self,
        owner_id: int,
        message_ref: str,
        raw_message: str,
        *,
        ttl_days: int = DEFAULT_MESSAGE_TTL_DAYS,
    ) -> None:
        """Persist one raw message encrypted under its ref.

        Args:
            owner_id: Owning account; resolution is scoped to it.
            message_ref: The ref the redacted definition row carries.
            raw_message: The raw prompt; never persisted anywhere else.
            ttl_days: Retention window from now (refreshed on resolve).

        Raises:
            RuntimeError: If encryption keys are not configured or the
                write fails — the caller must refuse authoring rather than
                persist a definition whose ref can never resolve.
        """
        self.ensure_schema()
        primary, _secondary = _message_store_keys()
        if not primary:
            raise RuntimeError("automation message encryption key is not configured")
        envelope = encrypt_json_blob_with_key({"message": raw_message}, primary)
        if not envelope:
            raise RuntimeError("automation message encryption failed")
        import json

        blob = json.dumps(envelope)
        now = _utcnow()
        expires = now + timedelta(days=ttl_days)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO automation_messages
                    (message_ref, owner_id, encrypted_blob, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(message_ref) DO UPDATE SET
                    encrypted_blob = excluded.encrypted_blob,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    message_ref,
                    int(owner_id),
                    blob,
                    now.isoformat(),
                    expires.isoformat(),
                ),
            )

    def resolve_message(self, owner_id: int, message_ref: str) -> str | None:
        """Return the raw message for one ref, or None when unresolvable.

        None covers every unresolvable case deliberately — missing ref,
        wrong owner, expired TTL, corrupt or undecryptable blob — because
        the executor's contract is a single honest failure, not a taxonomy
        the run row would then leak parts of. A successful resolve refreshes
        the TTL (last-access retention).
        """
        self.ensure_schema()
        now = _utcnow()
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT encrypted_blob, expires_at FROM automation_messages"
                " WHERE message_ref = ? AND owner_id = ?",
                (message_ref, int(owner_id)),
            ).fetchone()
        if row is None:
            return None
        try:
            if datetime.fromisoformat(str(row["expires_at"])) < now:
                return None
            import json

            envelope = json.loads(str(row["encrypted_blob"]))
        except (ValueError, TypeError):
            return None
        primary, secondary = _message_store_keys()
        payload: dict[str, Any] | None = None
        if primary:
            payload = decrypt_json_blob_with_key(envelope, primary)
        if payload is None and secondary:
            payload = decrypt_json_blob_with_key(envelope, secondary)
        if payload is None:
            return None
        message = payload.get("message")
        if not isinstance(message, str) or not message:
            return None
        # Last-access retention: the ref is demonstrably still in use.
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "UPDATE automation_messages SET expires_at = ?"
                    " WHERE message_ref = ? AND owner_id = ?",
                    (
                        (now + timedelta(days=DEFAULT_MESSAGE_TTL_DAYS)).isoformat(),
                        message_ref,
                        int(owner_id),
                    ),
                )
        except sqlite3.Error:
            pass  # Retention refresh is best-effort; the resolve stands.
        return message

    # --- retention --------------------------------------------------------

    def purge_expired(self, now: datetime | None = None) -> int:
        """Delete expired entries; return the number removed."""
        self.ensure_schema()
        moment = (now or _utcnow()).isoformat()
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM automation_messages WHERE expires_at < ?", (moment,)
            )
            return int(cursor.rowcount or 0)
