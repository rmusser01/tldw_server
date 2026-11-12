"""Transaction-local owner-bound receipts for strict Workspace chat startup."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.Chat.assistant_startup import AssistantStartup, decode_assistant_startup
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@dataclass(frozen=True)
class WorkspaceStartupResult:
    """Committed conversation result, never a serialized receipt or request."""

    conversation: Mapping[str, Any]
    replayed: bool


class WorkspaceStartupError(InputError):
    """Content-free domain error for the strict startup protocol."""

    def __init__(self, code: str, status_code: int, reason: str | None = None) -> None:
        """Carry only internal bounded codes, never database/request messages."""
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.reason = reason


def startup_binding_digest(db: CharactersRAGDB, row: Mapping[str, Any]) -> str:
    """Hash existing normalized identity, scope and validated server-owned origin."""
    try:
        kind, identity, character, memory = db.conversation_store._normalize_conversation_assistant_identity(
            character_id=row.get("character_id"),
            assistant_kind=row.get("assistant_kind"),
            assistant_id=row.get("assistant_id"),
            persona_memory_mode=row.get("persona_memory_mode"),
        )
        scope, workspace = db.conversation_store._normalize_scope(row.get("scope_type"), row.get("workspace_id"))
        raw = row.get("assistant_startup_json")
        # Projection intentionally tolerates corruption; replay authority must not.
        if raw is not None:
            AssistantStartup.model_validate_json(raw)
        origin = decode_assistant_startup(raw).model_dump(mode="json")
        encoded = json.dumps(
            {
                "schema_version": 1,
                "assistant_kind": kind,
                "assistant_id": identity,
                "character_id": character,
                "persona_memory_mode": memory,
                "assistant_startup": origin,
                "scope_type": scope,
                "workspace_id": workspace,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    except (ValueError, TypeError, AttributeError, RecursionError):
        raise WorkspaceStartupError("workspace_chat_startup_changed", 409) from None
    return hashlib.sha256(encoded).hexdigest()


class WorkspaceChatStartupStore:
    """Explicit-connection primitives; the orchestrator owns acceptance transactions."""

    def __init__(self, db: CharactersRAGDB) -> None:
        """Retain the scoped database, not a separate connection or cached owner."""
        self._db = db

    def _owner(self, owner_id: str) -> None:
        """Reject namespace substitution before any query, including on SQLite."""
        if owner_id != self._db.owner_user_id:
            raise WorkspaceStartupError("workspace_chat_startup_owner_mismatch", 404)

    def _transaction(self, conn: Any) -> None:
        """Never allow locking/count/write helpers to open an implicit transaction."""
        active = bool(getattr(conn, "in_transaction", False))
        backend = self._db._get_pinned_backend() or self._db._backend
        if backend.backend_type == BackendType.POSTGRESQL:
            raw = getattr(conn, "_connection", None)
            state = self._db._connection_state()
            status = getattr(getattr(raw, "info", None), "transaction_status", None)
            # PostgreSQL's compatibility in_transaction flag is always true.
            # A managed transaction may be lazy (IDLE) before its first SQL.
            active = bool(
                getattr(conn, "_db", None) is self._db
                and raw is getattr(state, "conn", None)
                and getattr(state, "tx_depth", 0) > 0
                and not getattr(raw, "autocommit", True)
                and getattr(status, "name", None) in ("IDLE", "INTRANS")
            )
        if not active:
            raise WorkspaceStartupError("workspace_chat_startup_transaction_required", 409)

    def require_outermost(self) -> None:
        """Reject managed or driver-open caller work without settling it or issuing SQL."""
        db = self._db
        state = db._connection_state()
        connection = getattr(state, "conn", None)
        # The public backend accessor can refresh and bootstrap a shared target.
        backend = db._get_pinned_backend() or db._backend
        if backend.backend_type == BackendType.SQLITE:
            nested = bool(getattr(connection, "in_transaction", False))
        else:
            nested = bool(getattr(state, "tx_depth", 0))
            if connection is not None:
                status = getattr(getattr(connection, "info", None), "transaction_status", None)
                nested = bool(nested or backend._tx_depth(connection) or getattr(status, "name", None) != "IDLE")
        if nested:
            raise WorkspaceStartupError("workspace_chat_startup_transaction_required", 409)

    def lock_owner(self, owner_id: str, *, conn: Any) -> None:
        """Serialize lifetime capacity across all owner Workspaces with a stable key."""
        self._owner(owner_id)
        self._transaction(conn)
        if self._db.backend_type == BackendType.POSTGRESQL:
            digest = hashlib.sha256(f"workspace_chat_startup_v1\0{owner_id}".encode()).digest()
            keys = (int.from_bytes(digest[:4], "big", signed=True), int.from_bytes(digest[4:8], "big", signed=True))
            conn.execute("SELECT pg_advisory_xact_lock(?, ?)", keys).fetchone()

    def get_receipt(
        self, owner_id: str, key_digest: str, *, conn: Any, for_update: bool = False
    ) -> dict[str, Any] | None:
        """Read only the authenticated owner's key; optional lock comes after chat lock."""
        self._owner(owner_id)
        self._transaction(conn)
        query = "SELECT * FROM workspace_chat_startup_receipts WHERE owner_user_id = ? AND key_digest = ?"
        if for_update and self._db.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE"
        row = conn.execute(query, (owner_id, key_digest)).fetchone()
        return dict(row) if row else None

    def lock_conversation(self, conversation_id: str, *, conn: Any) -> dict[str, Any] | None:
        """Lock the current chat, including soft-deleted state, inside caller scope."""
        self._transaction(conn)
        query = "SELECT * FROM conversations WHERE id = ?"
        if self._db.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE"
        row = conn.execute(query, (conversation_id,)).fetchone()
        return dict(row) if row else None

    def lock_receipt_workspace(
        self, workspace_id: str, *, conn: Any, for_create: bool = False
    ) -> dict[str, Any] | None:
        """Include lifecycle fields; replay's weaker lock remains compatible with FK reads."""
        self._transaction(conn)
        query = "SELECT * FROM workspaces WHERE id = ? AND client_id = ?"
        if self._db.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE" if for_create else " FOR NO KEY UPDATE"
        row = conn.execute(query, (workspace_id, self._db.owner_user_id)).fetchone()
        return self._db._workspace_row_to_dict(row) if row else None

    def has_receipt(self, conversation_id: str, *, conn: Any) -> bool:
        """Keep association visible after invalidation without taking a receipt lock."""
        self._transaction(conn)
        return (
            conn.execute(
                "SELECT 1 FROM workspace_chat_startup_receipts WHERE owner_user_id = ? AND conversation_id = ? LIMIT 1",
                (self._db.owner_user_id, conversation_id),
            ).fetchone()
            is not None
        )

    def count_receipts(self, owner_id: str, *, conn: Any) -> int:
        """Count all lifetime keys, including invalidated and hard-deleted tombstones."""
        self._owner(owner_id)
        self._transaction(conn)
        return int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM workspace_chat_startup_receipts WHERE owner_user_id = ?", (owner_id,)
            ).fetchone()["n"]
        )

    def count_live_chats(self, owner_id: str, workspace_id: str, *, conn: Any) -> int:
        """Count new-chat quota inside the same owner admission transaction."""
        self._owner(owner_id)
        self._transaction(conn)
        return int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM conversations WHERE scope_type = 'workspace' AND workspace_id = ? AND deleted = ?",
                (workspace_id, False),
            ).fetchone()["n"]
        )

    def insert_receipt(
        self,
        owner_id: str,
        key_digest: str,
        request_fingerprint: str,
        binding_digest: str,
        workspace_id: str,
        conversation_id: str,
        *,
        conn: Any,
    ) -> None:
        """Insert references and hashes atomically with the already-inserted chat."""
        self._owner(owner_id)
        self._transaction(conn)
        conn.execute(
            "INSERT INTO workspace_chat_startup_receipts (owner_user_id, key_digest, request_fingerprint, binding_digest, workspace_id, conversation_id) VALUES (?, ?, ?, ?, ?, ?)",
            (owner_id, key_digest, request_fingerprint, binding_digest, workspace_id, conversation_id),
        )

    def invalidate_changed_binding(
        self, conversation_id: str, before: Mapping[str, Any], after: Mapping[str, Any], *, conn: Any
    ) -> None:
        """Permanently invalidate actual binding changes in their mutation transaction."""
        self._transaction(conn)
        try:
            unchanged = startup_binding_digest(self._db, before) == startup_binding_digest(self._db, after)
        except WorkspaceStartupError:
            unchanged = False
        if not unchanged:
            conn.execute(
                "UPDATE workspace_chat_startup_receipts SET invalidated_at = COALESCE(invalidated_at, ?) WHERE owner_user_id = ? AND conversation_id = ?",
                (self._db._get_current_utc_timestamp_iso(), self._db.owner_user_id, conversation_id),
            )

    def mark_hard_deleted(self, conversation_id: str, *, conn: Any) -> None:
        """Retain a permanently unbound receipt before the chat id can be reused."""
        self._transaction(conn)
        conn.execute(
            "UPDATE workspace_chat_startup_receipts SET conversation_id = NULL WHERE owner_user_id = ? AND conversation_id = ?",
            (self._db.owner_user_id, conversation_id),
        )
