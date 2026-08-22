"""Recipient-owned shared-workspace chat persistence with fenced leases."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
    InputError,
    logger,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


ClaimDisposition = Literal["claimed", "replay", "in_progress", "request_id_conflict"]

_MIN_LEASE_SECONDS = 5 * 60
_MAX_LEASE_SECONDS = 30 * 60
_MAX_RETRY_AFTER_MS = _MAX_LEASE_SECONDS * 1000
_MAX_SOURCE_IDS = 500
_MAX_SOURCE_ID_BYTES = 512
_MAX_SOURCE_IDS_JSON_BYTES = 64 * 1024
_MAX_CITATIONS = 20
_MAX_CITATION_QUOTE_CHARS = 1_000
_MAX_CITATION_QUOTES_CHARS = 16_000
_MAX_CITATION_JSON_BYTES = 64 * 1024
_MAX_CURSOR_BYTES = 2_048
_MAX_ERROR_CODE_BYTES = 128
_MAX_PROVIDER_BYTES = 128
_MAX_MODEL_BYTES = 512
_MAX_SNAPSHOT_HASH_BYTES = 512
_MAX_MESSAGE_CONTENT_BYTES = 512 * 1024
_CURSOR_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_CITATION_KEYS = frozenset(
    {"citation_id", "source_id", "source_title", "locator", "quote", "score"}
)
_LOCATOR_KEYS = frozenset({"chunk", "start_char", "end_char"})


class StaleSharedWorkspaceChatClaim(ConflictError):
    """Raised when a claimant no longer owns the receipt fence."""

    def __init__(self) -> None:
        super().__init__("Shared workspace chat claim is stale.")


@dataclass(frozen=True)
class SharedWorkspaceChatThread:
    share_id: int
    conversation_id: str
    owner_user_id: str
    workspace_id: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class SharedWorkspaceStoredMessage:
    message_id: str
    role: str
    content: str
    created_at: datetime
    last_modified: datetime
    citations: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class StoredSharedWorkspaceTurn:
    share_id: int
    request_id: UUID
    conversation_id: str
    user_message: SharedWorkspaceStoredMessage
    assistant_message: SharedWorkspaceStoredMessage
    citations: tuple[dict[str, Any], ...]
    provider: str
    model: str
    source_mode: str
    effective_source_count: int


@dataclass(frozen=True)
class SharedWorkspaceChatClaim:
    disposition: ClaimDisposition
    share_id: int
    request_id: UUID
    request_fingerprint: str
    conversation_id: str
    lease_epoch: int
    lease_token: str | None
    lease_expires_at: datetime | None
    retry_after_ms: int | None = None
    source_mode: str | None = None
    source_ids: tuple[str, ...] = ()
    source_snapshot_hash: str | None = None
    provider: str | None = None
    model: str | None = None
    completed_turn: StoredSharedWorkspaceTurn | None = None


@dataclass(frozen=True)
class SharedWorkspaceMessagePage:
    messages: tuple[SharedWorkspaceStoredMessage, ...]
    next_before: str | None


class SharedWorkspaceChatStore:
    """Own all SQL for recipient shared-workspace chat lifecycle state."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db
        raw_recipient = getattr(db, "client_id", None)
        if raw_recipient is None:
            raise InputError("A recipient client ID is required for shared workspace chat.")
        self._recipient_user_id = str(raw_recipient).strip()
        if not self._recipient_user_id:
            raise InputError("A non-blank recipient client ID is required for shared workspace chat.")

    def get_or_create_thread(
        self,
        *,
        share_id: int,
        owner_user_id: str,
        workspace_id: str,
        workspace_name: str,
    ) -> SharedWorkspaceChatThread:
        """Return the one recipient-owned conversation mapped to a share."""
        normalized_share_id = self._validate_share_id(share_id)
        owner = self._bounded_text(owner_user_id, "owner_user_id", 512)
        workspace = self._bounded_text(workspace_id, "workspace_id", 512)
        title = self._bounded_text(workspace_name, "workspace_name", 1_000)

        existing = self.get_thread(share_id=normalized_share_id)
        if existing is not None:
            return existing

        try:
            with self._db.transaction() as conn:
                existing = self._get_thread_with_conn(conn, normalized_share_id)
                if existing is not None:
                    return existing
                conversation_id = self._db.add_conversation(
                    {
                        "title": title,
                        "source": "shared_workspace",
                        "external_ref": f"share:{normalized_share_id}",
                        "scope_type": "global",
                        "workspace_id": None,
                        "client_id": self._recipient_user_id,
                    }
                )
                if not conversation_id:
                    raise CharactersRAGDBError("Shared workspace conversation creation failed.")
                conn.execute(
                    """
                    INSERT INTO shared_workspace_chat_threads(
                        recipient_user_id, share_id, conversation_id,
                        owner_user_id, workspace_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        self._recipient_user_id,
                        normalized_share_id,
                        conversation_id,
                        owner,
                        workspace,
                    ),
                )
                created = self._get_thread_with_conn(conn, normalized_share_id)
                if created is None:
                    raise CharactersRAGDBError("Shared workspace thread mapping was not persisted.")
                return created
        except (sqlite3.IntegrityError, BackendDatabaseError, ConflictError) as conflict_exc:
            # The failed transaction exits and rolls back before the winning row is reloaded.
            try:
                with self._db.transaction() as conn:
                    winner = self._get_thread_with_conn(conn, normalized_share_id)
            except (sqlite3.Error, BackendDatabaseError) as reload_exc:
                raise CharactersRAGDBError(
                    f"Shared workspace thread race reload failed: {reload_exc}"
                ) from reload_exc
            if winner is not None:
                return winner
            raise CharactersRAGDBError(
                "Shared workspace thread creation conflicted without a winner."
            ) from conflict_exc
        except CharactersRAGDBError:
            raise
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Shared workspace thread persistence failed: {exc}") from exc

    def get_thread(self, *, share_id: int) -> SharedWorkspaceChatThread | None:
        """Return a recipient-visible thread without creating one."""
        normalized_share_id = self._validate_share_id(share_id)
        try:
            with self._db.transaction() as conn:
                return self._get_thread_with_conn(conn, normalized_share_id)
        except CharactersRAGDBError:
            raise
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Shared workspace thread lookup failed: {exc}") from exc

    def claim_request(
        self,
        *,
        share_id: int,
        request_id: UUID,
        request_fingerprint: str,
        conversation_id: str,
        lease_seconds: int,
        now: datetime,
    ) -> SharedWorkspaceChatClaim:
        """Insert a new receipt or reclaim a matching retryable/expired receipt."""
        normalized_share_id = self._validate_share_id(share_id)
        normalized_request_id = self._validate_request_id(request_id)
        fingerprint = self._bounded_text(request_fingerprint, "request_fingerprint", 4_096)
        conversation = self._bounded_text(conversation_id, "conversation_id", 512)
        current_time = self._aware_utc(now, field="now", reject_naive=True)
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int):
            raise InputError("lease_seconds must be an integer.")
        lease_duration = max(_MIN_LEASE_SECONDS, min(_MAX_LEASE_SECONDS, lease_seconds))

        try:
            self.purge_expired_conflicts(now=current_time, limit=100)
        except CharactersRAGDBError as exc:
            logger.warning("Shared chat conflict cleanup failed before claim: {}", exc)

        token = secrets.token_urlsafe(32)
        expires_at = current_time + timedelta(seconds=lease_duration)
        try:
            with self._db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO shared_workspace_chat_requests(
                        recipient_user_id, share_id, request_id, request_fingerprint,
                        conversation_id, status, lease_epoch, lease_token,
                        lease_expires_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'in_progress', 1, ?, ?, ?, ?)
                    """,
                    (
                        self._recipient_user_id,
                        normalized_share_id,
                        str(normalized_request_id),
                        fingerprint,
                        conversation,
                        token,
                        self._db_datetime(expires_at),
                        self._db_datetime(current_time),
                        self._db_datetime(current_time),
                    ),
                )
            return SharedWorkspaceChatClaim(
                disposition="claimed",
                share_id=normalized_share_id,
                request_id=normalized_request_id,
                request_fingerprint=fingerprint,
                conversation_id=conversation,
                lease_epoch=1,
                lease_token=token,
                lease_expires_at=expires_at,
            )
        except (sqlite3.IntegrityError, BackendDatabaseError, ConflictError):
            pass
        except CharactersRAGDBError:
            raise
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Shared workspace request claim failed: {exc}") from exc

        try:
            return self._resolve_existing_claim(
                share_id=normalized_share_id,
                request_id=normalized_request_id,
                fingerprint=fingerprint,
                conversation_id=conversation,
                token=token,
                expires_at=expires_at,
                now=current_time,
            )
        except CharactersRAGDBError:
            raise
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(
                f"Shared workspace existing claim resolution failed: {exc}"
            ) from exc

    def freeze_sources(
        self,
        *,
        claim: SharedWorkspaceChatClaim,
        source_mode: str,
        source_ids: tuple[str, ...],
        snapshot_hash: str,
        provider: str,
        model: str,
    ) -> bool:
        """Freeze the canonical source set once under the active claim fence."""
        self._validate_claim(claim)
        mode = self._validate_source_mode(source_mode)
        normalized_ids, source_json = self._validate_source_ids(source_ids)
        snapshot = self._bounded_text(
            snapshot_hash, "snapshot_hash", _MAX_SNAPSHOT_HASH_BYTES
        )
        normalized_provider = self._bounded_text(provider, "provider", _MAX_PROVIDER_BYTES)
        normalized_model = self._bounded_text(model, "model", _MAX_MODEL_BYTES)
        del normalized_ids
        try:
            with self._db.transaction() as conn:
                current = conn.execute(
                    """
                    SELECT source_mode, source_ids_json, source_snapshot_hash, provider, model
                      FROM shared_workspace_chat_requests
                     WHERE recipient_user_id = ? AND share_id = ? AND request_id = ?
                       AND conversation_id = ? AND lease_epoch = ? AND lease_token = ?
                       AND status = 'in_progress'
                    """,
                    (
                        self._recipient_user_id,
                        claim.share_id,
                        str(claim.request_id),
                        claim.conversation_id,
                        claim.lease_epoch,
                        claim.lease_token,
                    ),
                ).fetchone()
                if current is None:
                    return False
                frozen = self._row_dict(current)
                if frozen.get("source_ids_json") is not None:
                    return frozen == {
                        "source_mode": mode,
                        "source_ids_json": source_json,
                        "source_snapshot_hash": snapshot,
                        "provider": normalized_provider,
                        "model": normalized_model,
                    }
                cursor = conn.execute(
                    """
                    UPDATE shared_workspace_chat_requests
                       SET source_mode = ?, source_ids_json = ?, source_snapshot_hash = ?,
                           provider = ?, model = ?, updated_at = ?
                     WHERE recipient_user_id = ? AND share_id = ? AND request_id = ?
                       AND conversation_id = ? AND lease_epoch = ? AND lease_token = ?
                       AND status = 'in_progress'
                       AND source_ids_json IS NULL
                    """,
                    (
                        mode,
                        source_json,
                        snapshot,
                        normalized_provider,
                        normalized_model,
                        self._db_datetime(datetime.now(timezone.utc)),
                        self._recipient_user_id,
                        claim.share_id,
                        str(claim.request_id),
                        claim.conversation_id,
                        claim.lease_epoch,
                        claim.lease_token,
                    ),
                )
                return cursor.rowcount == 1
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Shared workspace source freeze failed: {exc}") from exc

    def mark_retryable(self, *, claim: SharedWorkspaceChatClaim, error_code: str) -> bool:
        """Release a current claim for a matching retry."""
        return self._mark_failed(claim=claim, status="retryable", error_code=error_code)

    def mark_conflicted(self, *, claim: SharedWorkspaceChatClaim, error_code: str) -> bool:
        """Fence a frozen source mismatch until bounded cleanup."""
        return self._mark_failed(claim=claim, status="conflicted", error_code=error_code)

    def complete_turn(
        self,
        *,
        claim: SharedWorkspaceChatClaim,
        query: str,
        answer: str,
        citations: list[dict[str, Any]],
        provider: str,
        model: str,
        source_mode: str,
        effective_source_count: int,
    ) -> StoredSharedWorkspaceTurn:
        """Atomically persist both messages, strict metadata, and receipt completion."""
        self._validate_claim(claim)
        normalized_query = self._bounded_text(query, "query", _MAX_MESSAGE_CONTENT_BYTES)
        normalized_answer = self._bounded_text(answer, "answer", _MAX_MESSAGE_CONTENT_BYTES)
        normalized_citations = self._validate_citations(citations)
        normalized_provider = self._bounded_text(provider, "provider", _MAX_PROVIDER_BYTES)
        normalized_model = self._bounded_text(model, "model", _MAX_MODEL_BYTES)
        mode = self._validate_source_mode(source_mode)
        if isinstance(effective_source_count, bool) or not isinstance(effective_source_count, int):
            raise InputError("effective_source_count must be an integer.")
        if not 1 <= effective_source_count <= _MAX_SOURCE_IDS:
            raise InputError("effective_source_count must be between 1 and 500.")

        user_message_id = self._db._generate_uuid()
        assistant_message_id = self._db._generate_uuid()
        try:
            with self._db.transaction() as conn:
                receipt = self._assert_current_claim(conn, claim)
                source_ids = self._source_ids_from_row(receipt)
                if any(
                    citation["source_id"] not in source_ids
                    for citation in normalized_citations
                ):
                    raise InputError(
                        "Citation source_id must belong to the frozen source scope."
                    )
                if (
                    receipt.get("source_mode") != mode
                    or receipt.get("provider") != normalized_provider
                    or receipt.get("model") != normalized_model
                    or len(source_ids) != effective_source_count
                ):
                    raise StaleSharedWorkspaceChatClaim
                created_user = self._db.add_message(
                    {
                        "id": user_message_id,
                        "conversation_id": claim.conversation_id,
                        "sender": "user",
                        "content": normalized_query,
                        "client_id": self._recipient_user_id,
                    }
                )
                created_assistant = self._db.add_message(
                    {
                        "id": assistant_message_id,
                        "conversation_id": claim.conversation_id,
                        "sender": "assistant",
                        "content": normalized_answer,
                        "client_id": self._recipient_user_id,
                    }
                )
                if created_user != user_message_id or created_assistant != assistant_message_id:
                    raise CharactersRAGDBError("Shared workspace message IDs were not persisted exactly.")
                self._write_message_metadata_strict(
                    conn,
                    assistant_message_id,
                    normalized_citations,
                    normalized_provider,
                    normalized_model,
                    mode,
                    effective_source_count,
                )
                completed_at = datetime.now(timezone.utc)
                updated = conn.execute(
                    """
                    UPDATE shared_workspace_chat_requests
                       SET status = 'completed', user_message_id = ?, assistant_message_id = ?,
                           provider = ?, model = ?, error_code = NULL,
                           lease_token = NULL, lease_expires_at = NULL,
                           completed_at = ?, updated_at = ?
                     WHERE recipient_user_id = ? AND share_id = ? AND request_id = ?
                       AND conversation_id = ? AND lease_epoch = ? AND lease_token = ?
                       AND status = 'in_progress'
                    """,
                    (
                        user_message_id,
                        assistant_message_id,
                        normalized_provider,
                        normalized_model,
                        self._db_datetime(completed_at),
                        self._db_datetime(completed_at),
                        self._recipient_user_id,
                        claim.share_id,
                        str(claim.request_id),
                        claim.conversation_id,
                        claim.lease_epoch,
                        claim.lease_token,
                    ),
                )
                if updated.rowcount != 1:
                    raise StaleSharedWorkspaceChatClaim
            stored = self.load_completed_turn(
                share_id=claim.share_id,
                request_id=claim.request_id,
            )
            if stored is None:
                raise CharactersRAGDBError("Completed shared workspace turn could not be reloaded.")
            return stored
        except (InputError, StaleSharedWorkspaceChatClaim, CharactersRAGDBError):
            raise
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Shared workspace turn completion failed: {exc}") from exc

    def load_completed_turn(
        self,
        *,
        share_id: int,
        request_id: UUID,
    ) -> StoredSharedWorkspaceTurn | None:
        """Load a completed recipient turn from durable message references."""
        normalized_share_id = self._validate_share_id(share_id)
        normalized_request_id = self._validate_request_id(request_id)
        try:
            with self._db.transaction() as conn:
                row = self._fetch_receipt_with_conn(
                    conn, normalized_share_id, normalized_request_id
                )
                if row is None or row.get("status") != "completed":
                    return None
                return self._turn_from_receipt(conn, row)
        except CharactersRAGDBError:
            raise
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Shared workspace replay load failed: {exc}") from exc

    def list_messages(
        self,
        *,
        share_id: int,
        before: str | None,
        limit: int,
    ) -> SharedWorkspaceMessagePage:
        """List a stable recipient thread page, newest selection returned chronologically."""
        normalized_share_id = self._validate_share_id(share_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise InputError("Shared workspace message limit must be between 1 and 100.")
        cursor = self._decode_cursor(before) if before is not None else None
        try:
            with self._db.transaction() as conn:
                thread = self._get_thread_with_conn(conn, normalized_share_id)
                if thread is None:
                    return SharedWorkspaceMessagePage(messages=(), next_before=None)
                params: list[Any] = [
                    self._recipient_user_id,
                    normalized_share_id,
                    thread.conversation_id,
                    self._recipient_user_id,
                ]
                cursor_clause = ""
                if cursor is not None:
                    timestamp, last_modified, message_id = cursor
                    cursor_clause = """
                      AND (
                        message.timestamp < ?
                        OR (message.timestamp = ? AND message.last_modified < ?)
                        OR (
                          message.timestamp = ? AND message.last_modified = ?
                          AND message.id < ?
                        )
                      )
                    """
                    params.extend(
                        [timestamp, timestamp, last_modified, timestamp, last_modified, message_id]
                    )
                params.append(limit + 1)
                rows = conn.execute(
                    f"""
                    SELECT message.id, message.sender, message.content, message.timestamp,
                           message.last_modified, metadata.extra_json
                      FROM shared_workspace_chat_threads AS thread
                      JOIN messages AS message ON message.conversation_id = thread.conversation_id
                 LEFT JOIN message_metadata AS metadata ON metadata.message_id = message.id
                     WHERE thread.recipient_user_id = ? AND thread.share_id = ?
                       AND thread.conversation_id = ? AND message.client_id = ?
                       AND message.deleted = FALSE
                       {cursor_clause}
                  ORDER BY message.timestamp DESC, message.last_modified DESC, message.id DESC
                     LIMIT ?
                    """,  # nosec B608 - the optional clause is fixed store SQL.
                    tuple(params),
                ).fetchall()
                has_more = len(rows) > limit
                selected = rows[:limit]
                next_before = None
                if has_more and selected:
                    oldest = self._row_dict(selected[-1])
                    next_before = self._encode_cursor(
                        oldest["timestamp"], oldest["last_modified"], str(oldest["id"])
                    )
                messages = tuple(
                    self._message_from_row(self._row_dict(row)) for row in reversed(selected)
                )
                return SharedWorkspaceMessagePage(
                    messages=messages,
                    next_before=next_before,
                )
        except (InputError, CharactersRAGDBError):
            raise
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Shared workspace history load failed: {exc}") from exc

    def purge_expired_conflicts(self, *, now: datetime, limit: int = 100) -> int:
        """Delete at most 100 recipient conflicted receipts older than 24 hours."""
        current_time = self._aware_utc(now, field="now", reject_naive=True)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise InputError("Conflict cleanup limit must be a positive integer.")
        bounded_limit = min(limit, 100)
        cutoff = current_time - timedelta(hours=24)
        if self._db.backend_type == BackendType.SQLITE:
            select_sql = """
                SELECT share_id, request_id
                  FROM shared_workspace_chat_requests
                 WHERE recipient_user_id = ? AND status = 'conflicted'
                   AND julianday(updated_at) < julianday(?)
              ORDER BY julianday(updated_at) ASC, share_id ASC, request_id ASC
                 LIMIT ?
            """
            delete_sql = """
                DELETE FROM shared_workspace_chat_requests
                 WHERE recipient_user_id = ? AND share_id = ? AND request_id = ?
                   AND status = 'conflicted'
                   AND julianday(updated_at) < julianday(?)
            """
        else:
            select_sql = """
                SELECT share_id, request_id
                  FROM shared_workspace_chat_requests
                 WHERE recipient_user_id = ? AND status = 'conflicted'
                   AND updated_at < ?
              ORDER BY updated_at ASC, share_id ASC, request_id ASC
                 LIMIT ?
            """
            delete_sql = """
                DELETE FROM shared_workspace_chat_requests
                 WHERE recipient_user_id = ? AND share_id = ? AND request_id = ?
                   AND status = 'conflicted' AND updated_at < ?
            """
        try:
            with self._db.transaction() as conn:
                rows = conn.execute(
                    select_sql,
                    (
                        self._recipient_user_id,
                        self._db_datetime(cutoff),
                        bounded_limit,
                    ),
                ).fetchall()
                deleted = 0
                for row in rows:
                    record = self._row_dict(row)
                    cursor = conn.execute(
                        delete_sql,
                        (
                            self._recipient_user_id,
                            record["share_id"],
                            record["request_id"],
                            self._db_datetime(cutoff),
                        ),
                    )
                    deleted += max(0, cursor.rowcount)
                return deleted
        except (InputError, CharactersRAGDBError):
            raise
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Shared workspace conflict cleanup failed: {exc}") from exc

    def _resolve_existing_claim(
        self,
        *,
        share_id: int,
        request_id: UUID,
        fingerprint: str,
        conversation_id: str,
        token: str,
        expires_at: datetime,
        now: datetime,
    ) -> SharedWorkspaceChatClaim:
        try:
            with self._db.transaction() as conn:
                receipt = self._fetch_receipt_with_conn(conn, share_id, request_id)
            if receipt is None:
                raise CharactersRAGDBError(
                    "Conflicting shared workspace receipt was not reloadable."
                )

            for attempt in range(2):
                classified = self._classify_existing_claim_state(
                    receipt,
                    share_id=share_id,
                    request_id=request_id,
                    fingerprint=fingerprint,
                    conversation_id=conversation_id,
                    now=now,
                )
                if classified is not None:
                    return classified
                attempt_token = token if attempt == 0 else secrets.token_urlsafe(32)
                reclaimed = self._attempt_reclaim(
                    receipt,
                    share_id=share_id,
                    request_id=request_id,
                    fingerprint=fingerprint,
                    conversation_id=conversation_id,
                    token=attempt_token,
                    expires_at=expires_at,
                    now=now,
                )
                if reclaimed is not None:
                    return reclaimed
                with self._db.transaction() as conn:
                    receipt = self._fetch_receipt_with_conn(conn, share_id, request_id)
                if receipt is None:
                    raise CharactersRAGDBError("Reclaim race winner could not be loaded.")

            classified = self._classify_existing_claim_state(
                receipt,
                share_id=share_id,
                request_id=request_id,
                fingerprint=fingerprint,
                conversation_id=conversation_id,
                now=now,
            )
            if classified is not None:
                return classified
            raise CharactersRAGDBError(
                "Shared workspace request remained reclaimable after bounded contention."
            )
        except CharactersRAGDBError:
            raise
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Shared workspace request reclaim failed: {exc}") from exc

    def _classify_existing_claim_state(
        self,
        receipt: dict[str, Any],
        *,
        share_id: int,
        request_id: UUID,
        fingerprint: str,
        conversation_id: str,
        now: datetime,
    ) -> SharedWorkspaceChatClaim | None:
        if (
            receipt.get("request_fingerprint") != fingerprint
            or receipt.get("conversation_id") != conversation_id
        ):
            return self._claim_from_row(receipt, disposition="request_id_conflict")
        status = str(receipt.get("status"))
        if status == "completed":
            completed = self.load_completed_turn(share_id=share_id, request_id=request_id)
            if completed is None:
                raise CharactersRAGDBError("Completed receipt is missing its stored turn.")
            return self._claim_from_row(
                receipt,
                disposition="replay",
                completed_turn=completed,
            )
        prior_expiry = self._optional_datetime(receipt.get("lease_expires_at"))
        if status == "in_progress" and prior_expiry is not None and prior_expiry > now:
            retry_ms = min(
                _MAX_RETRY_AFTER_MS,
                max(0, math.ceil((prior_expiry - now).total_seconds() * 1000)),
            )
            return self._claim_from_row(
                receipt,
                disposition="in_progress",
                retry_after_ms=retry_ms,
            )
        if status not in {"retryable", "in_progress"}:
            return self._claim_from_row(receipt, disposition="request_id_conflict")
        return None

    def _attempt_reclaim(
        self,
        receipt: dict[str, Any],
        *,
        share_id: int,
        request_id: UUID,
        fingerprint: str,
        conversation_id: str,
        token: str,
        expires_at: datetime,
        now: datetime,
    ) -> SharedWorkspaceChatClaim | None:
        prior_epoch = int(receipt["lease_epoch"])
        status = str(receipt["status"])
        prior_expiry_db = receipt.get("lease_expires_at")
        with self._db.transaction() as conn:
            updated = conn.execute(
                """
                UPDATE shared_workspace_chat_requests
                   SET status = 'in_progress', lease_epoch = ?, lease_token = ?,
                       lease_expires_at = ?, error_code = NULL, updated_at = ?
                 WHERE recipient_user_id = ? AND share_id = ? AND request_id = ?
                   AND request_fingerprint = ? AND conversation_id = ?
                   AND lease_epoch = ? AND status = ?
                   AND (
                     lease_expires_at = ?
                     OR (lease_expires_at IS NULL AND ? IS NULL)
                   )
                """,
                (
                    prior_epoch + 1,
                    token,
                    self._db_datetime(expires_at),
                    self._db_datetime(now),
                    self._recipient_user_id,
                    share_id,
                    str(request_id),
                    fingerprint,
                    conversation_id,
                    prior_epoch,
                    status,
                    prior_expiry_db,
                    prior_expiry_db,
                ),
            )
            if updated.rowcount != 1:
                return None
            refreshed = self._fetch_receipt_with_conn(conn, share_id, request_id)
            if refreshed is None:
                raise CharactersRAGDBError("Reclaimed receipt could not be loaded.")
            return self._claim_from_row(refreshed, disposition="claimed")

    def _mark_failed(
        self,
        *,
        claim: SharedWorkspaceChatClaim,
        status: Literal["retryable", "conflicted"],
        error_code: str,
    ) -> bool:
        self._validate_claim(claim)
        code = self._bounded_text(error_code, "error_code", _MAX_ERROR_CODE_BYTES)
        failed_at = self._db_datetime(datetime.now(timezone.utc))
        try:
            with self._db.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE shared_workspace_chat_requests
                       SET status = ?, error_code = ?, lease_token = NULL,
                           lease_expires_at = NULL, updated_at = ?
                     WHERE recipient_user_id = ? AND share_id = ? AND request_id = ?
                       AND conversation_id = ? AND lease_epoch = ? AND lease_token = ?
                       AND status = 'in_progress'
                       AND (? = 'retryable' OR source_ids_json IS NOT NULL)
                    """,
                    (
                        status,
                        code,
                        failed_at,
                        self._recipient_user_id,
                        claim.share_id,
                        str(claim.request_id),
                        claim.conversation_id,
                        claim.lease_epoch,
                        claim.lease_token,
                        status,
                    ),
                )
                return cursor.rowcount == 1
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Shared workspace failure transition failed: {exc}") from exc

    def _get_thread_with_conn(
        self,
        conn: Any,
        share_id: int,
    ) -> SharedWorkspaceChatThread | None:
        row = conn.execute(
            """
            SELECT thread.share_id, thread.conversation_id, thread.owner_user_id,
                   thread.workspace_id, thread.created_at, thread.updated_at
              FROM shared_workspace_chat_threads AS thread
              JOIN conversations AS conversation ON conversation.id = thread.conversation_id
             WHERE thread.recipient_user_id = ? AND thread.share_id = ?
               AND conversation.client_id = ? AND conversation.deleted = FALSE
            """,
            (self._recipient_user_id, share_id, self._recipient_user_id),
        ).fetchone()
        if row is None:
            return None
        record = self._row_dict(row)
        return SharedWorkspaceChatThread(
            share_id=int(record["share_id"]),
            conversation_id=str(record["conversation_id"]),
            owner_user_id=str(record["owner_user_id"]),
            workspace_id=str(record["workspace_id"]),
            created_at=self._parse_datetime(record["created_at"]),
            updated_at=self._parse_datetime(record["updated_at"]),
        )

    def _fetch_receipt_with_conn(
        self,
        conn: Any,
        share_id: int,
        request_id: UUID,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT recipient_user_id, share_id, request_id, request_fingerprint,
                   conversation_id, status, lease_epoch, lease_token, lease_expires_at,
                   source_mode, source_ids_json, source_snapshot_hash, provider, model,
                   user_message_id, assistant_message_id, error_code,
                   created_at, updated_at, completed_at
              FROM shared_workspace_chat_requests
             WHERE recipient_user_id = ? AND share_id = ? AND request_id = ?
            """,
            (self._recipient_user_id, share_id, str(request_id)),
        ).fetchone()
        return None if row is None else self._row_dict(row)

    def _assert_current_claim(self, conn: Any, claim: SharedWorkspaceChatClaim) -> dict[str, Any]:
        row = conn.execute(
            """
            SELECT request_fingerprint, conversation_id, status, lease_epoch, lease_token,
                   source_mode, source_ids_json, source_snapshot_hash, provider, model
              FROM shared_workspace_chat_requests
             WHERE recipient_user_id = ? AND share_id = ? AND request_id = ?
               AND conversation_id = ? AND lease_epoch = ? AND lease_token = ?
               AND status = 'in_progress'
            """,
            (
                self._recipient_user_id,
                claim.share_id,
                str(claim.request_id),
                claim.conversation_id,
                claim.lease_epoch,
                claim.lease_token,
            ),
        ).fetchone()
        if row is None:
            raise StaleSharedWorkspaceChatClaim
        return self._row_dict(row)

    def _turn_from_receipt(self, conn: Any, receipt: dict[str, Any]) -> StoredSharedWorkspaceTurn:
        user_message_id = receipt.get("user_message_id")
        assistant_message_id = receipt.get("assistant_message_id")
        if not user_message_id or not assistant_message_id:
            raise CharactersRAGDBError("Completed receipt has missing message references.")
        rows = conn.execute(
            """
            SELECT message.id, message.sender, message.content, message.timestamp,
                   message.last_modified, metadata.extra_json
              FROM messages AS message
         LEFT JOIN message_metadata AS metadata ON metadata.message_id = message.id
             WHERE message.conversation_id = ? AND message.client_id = ?
               AND message.deleted = FALSE AND message.id IN (?, ?)
            """,
            (
                receipt["conversation_id"],
                self._recipient_user_id,
                user_message_id,
                assistant_message_id,
            ),
        ).fetchall()
        messages = {str(self._row_dict(row)["id"]): self._row_dict(row) for row in rows}
        if set(messages) != {str(user_message_id), str(assistant_message_id)}:
            raise CharactersRAGDBError("Completed receipt messages are missing or out of scope.")
        user_message = self._message_from_row(messages[str(user_message_id)])
        assistant_message = self._message_from_row(messages[str(assistant_message_id)])
        if user_message.role != "user" or assistant_message.role != "assistant":
            raise CharactersRAGDBError("Completed receipt message roles are invalid.")
        source_ids = self._source_ids_from_row(receipt)
        provider = receipt.get("provider")
        model = receipt.get("model")
        source_mode = receipt.get("source_mode")
        if not provider or not model or source_mode not in {"all", "include"} or not source_ids:
            raise CharactersRAGDBError("Completed receipt generation scope is incomplete.")
        return StoredSharedWorkspaceTurn(
            share_id=int(receipt["share_id"]),
            request_id=UUID(str(receipt["request_id"])),
            conversation_id=str(receipt["conversation_id"]),
            user_message=user_message,
            assistant_message=assistant_message,
            citations=assistant_message.citations,
            provider=str(provider),
            model=str(model),
            source_mode=str(source_mode),
            effective_source_count=len(source_ids),
        )

    def _write_message_metadata_strict(
        self,
        conn: Any,
        message_id: str,
        citations: tuple[dict[str, Any], ...],
        provider: str,
        model: str,
        source_mode: str,
        effective_source_count: int,
    ) -> None:
        rag_context = {
            "retrieved_documents": [
                {"source_id": citation["source_id"], "quote": citation["quote"]}
                for citation in citations
            ],
            "citations": list(citations),
            "generation": {"provider": provider, "model": model},
            "source_scope": {
                "mode": source_mode,
                "effective_source_count": effective_source_count,
            },
        }
        payload = json.dumps(
            {"rag_context": rag_context},
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(payload.encode("utf-8")) > _MAX_CITATION_JSON_BYTES:
            raise InputError("Shared workspace citation metadata is too large.")
        cursor = conn.execute(
            """
            INSERT INTO message_metadata(message_id, tool_calls_json, extra_json, last_modified)
            VALUES (?, NULL, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(message_id) DO UPDATE SET
              tool_calls_json = NULL,
              extra_json = excluded.extra_json,
              last_modified = CURRENT_TIMESTAMP
            """,
            (message_id, payload),
        )
        if cursor.rowcount != 1:
            raise CharactersRAGDBError("Strict shared workspace metadata write failed.")

    def _message_from_row(self, row: dict[str, Any]) -> SharedWorkspaceStoredMessage:
        citations: tuple[dict[str, Any], ...] = ()
        extra_json = row.get("extra_json")
        if extra_json:
            try:
                extra = json.loads(str(extra_json))
                raw_citations = extra.get("rag_context", {}).get("citations", [])
                citations = self._validate_citations(raw_citations)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise CharactersRAGDBError("Stored shared workspace citation metadata is invalid.") from exc
        return SharedWorkspaceStoredMessage(
            message_id=str(row["id"]),
            role=str(row["sender"]),
            content=str(row.get("content") or ""),
            created_at=self._parse_datetime(row["timestamp"]),
            last_modified=self._parse_datetime(row["last_modified"]),
            citations=citations,
        )

    def _claim_from_row(
        self,
        row: dict[str, Any],
        *,
        disposition: ClaimDisposition,
        retry_after_ms: int | None = None,
        completed_turn: StoredSharedWorkspaceTurn | None = None,
    ) -> SharedWorkspaceChatClaim:
        claimant = disposition == "claimed"
        return SharedWorkspaceChatClaim(
            disposition=disposition,
            share_id=int(row["share_id"]),
            request_id=UUID(str(row["request_id"])),
            request_fingerprint=str(row["request_fingerprint"]),
            conversation_id=str(row["conversation_id"]),
            lease_epoch=int(row["lease_epoch"]),
            lease_token=str(row["lease_token"]) if claimant and row.get("lease_token") else None,
            lease_expires_at=self._optional_datetime(row.get("lease_expires_at")),
            retry_after_ms=retry_after_ms,
            source_mode=str(row["source_mode"]) if row.get("source_mode") else None,
            source_ids=self._source_ids_from_row(row),
            source_snapshot_hash=(
                str(row["source_snapshot_hash"]) if row.get("source_snapshot_hash") else None
            ),
            provider=str(row["provider"]) if row.get("provider") else None,
            model=str(row["model"]) if row.get("model") else None,
            completed_turn=completed_turn,
        )

    def _source_ids_from_row(self, row: dict[str, Any]) -> tuple[str, ...]:
        raw = row.get("source_ids_json")
        if raw is None:
            return ()
        try:
            decoded = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise CharactersRAGDBError("Stored shared workspace source scope is invalid.") from exc
        if not isinstance(decoded, list):
            raise CharactersRAGDBError("Stored shared workspace source scope is invalid.")
        try:
            normalized, _ = self._validate_source_ids(tuple(decoded))
        except InputError as exc:
            raise CharactersRAGDBError("Stored shared workspace source scope is invalid.") from exc
        return normalized

    def _validate_source_ids(self, source_ids: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
        if not isinstance(source_ids, tuple) or not 1 <= len(source_ids) <= _MAX_SOURCE_IDS:
            raise InputError("source_ids must contain between 1 and 500 canonical IDs.")
        normalized: list[str] = []
        for source_id in source_ids:
            if not isinstance(source_id, str) or not source_id or source_id.strip() != source_id:
                raise InputError("source_ids must contain non-blank canonical strings.")
            if len(source_id.encode("utf-8")) > _MAX_SOURCE_ID_BYTES:
                raise InputError("A canonical source ID is too large.")
            normalized.append(source_id)
        if normalized != sorted(set(normalized)):
            raise InputError("source_ids must be sorted and unique.")
        payload = json.dumps(normalized, ensure_ascii=True, separators=(",", ":"))
        if len(payload.encode("utf-8")) > _MAX_SOURCE_IDS_JSON_BYTES:
            raise InputError("The serialized source scope is too large.")
        return tuple(normalized), payload

    def _validate_citations(
        self,
        citations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> tuple[dict[str, Any], ...]:
        if not isinstance(citations, (list, tuple)) or not 1 <= len(citations) <= _MAX_CITATIONS:
            raise InputError("Shared workspace turns require between 1 and 20 citations.")
        normalized: list[dict[str, Any]] = []
        total_quote_chars = 0
        for citation in citations:
            if not isinstance(citation, dict) or set(citation) != _CITATION_KEYS:
                raise InputError("Shared workspace citation fields are invalid.")
            citation_id = self._bounded_text(citation["citation_id"], "citation_id", 128)
            source_id = self._bounded_text(citation["source_id"], "source_id", _MAX_SOURCE_ID_BYTES)
            source_title = self._bounded_text(citation["source_title"], "source_title", 1_000)
            quote = self._bounded_text(citation["quote"], "quote", 4_000)
            if len(quote) > _MAX_CITATION_QUOTE_CHARS:
                raise InputError("A shared workspace citation quote exceeds 1,000 characters.")
            total_quote_chars += len(quote)
            if total_quote_chars > _MAX_CITATION_QUOTES_CHARS:
                raise InputError("Shared workspace citation quotes exceed 16,000 characters.")
            locator = citation["locator"]
            if not isinstance(locator, dict) or not set(locator).issubset(_LOCATOR_KEYS):
                raise InputError("Shared workspace citation locator is invalid.")
            normalized_locator: dict[str, int] = {}
            for key, value in locator.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise InputError("Shared workspace citation locator values must be non-negative integers.")
                normalized_locator[str(key)] = value
            score = citation["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise InputError("Shared workspace citation score must be numeric.")
            normalized_score = float(score)
            if not math.isfinite(normalized_score):
                raise InputError("Shared workspace citation score must be finite.")
            normalized.append(
                {
                    "citation_id": citation_id,
                    "source_id": source_id,
                    "source_title": source_title,
                    "locator": normalized_locator,
                    "quote": quote,
                    "score": normalized_score,
                }
            )
        try:
            payload = json.dumps(normalized, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise InputError("Shared workspace citations are not serializable.") from exc
        if len(payload.encode("utf-8")) > _MAX_CITATION_JSON_BYTES:
            raise InputError("Shared workspace citations are too large.")
        return tuple(normalized)

    def _encode_cursor(self, timestamp: Any, last_modified: Any, message_id: str) -> str:
        payload = json.dumps(
            [
                self._cursor_timestamp_text(timestamp, field="cursor timestamp"),
                self._cursor_timestamp_text(
                    last_modified,
                    field="cursor last_modified",
                ),
                message_id,
            ],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    def _decode_cursor(self, cursor: str) -> tuple[str, str, str]:
        if (
            not isinstance(cursor, str)
            or not cursor
            or len(cursor.encode("utf-8")) > _MAX_CURSOR_BYTES
            or not _CURSOR_RE.fullmatch(cursor)
        ):
            raise InputError("Invalid shared workspace message cursor.")
        try:
            padding = "=" * (-len(cursor) % 4)
            raw = base64.b64decode(
                (cursor + padding).encode("ascii"),
                altchars=b"-_",
                validate=True,
            )
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError, json.JSONDecodeError, binascii.Error) as exc:
            raise InputError("Invalid shared workspace message cursor.") from exc
        if base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") != cursor:
            raise InputError("Invalid shared workspace message cursor.")
        if (
            not isinstance(decoded, list)
            or len(decoded) != 3
            or not all(isinstance(value, str) and value for value in decoded)
            or len(decoded[2].encode("utf-8")) > 512
        ):
            raise InputError("Invalid shared workspace message cursor.")
        self._aware_utc_string(decoded[0], field="cursor timestamp")
        self._aware_utc_string(decoded[1], field="cursor last_modified")
        return decoded[0], decoded[1], decoded[2]

    @classmethod
    def _cursor_timestamp_text(cls, value: Any, *, field: str) -> str:
        if isinstance(value, datetime):
            return cls._format_datetime(cls._aware_utc(value, field=field, reject_naive=True))
        if isinstance(value, str):
            cls._aware_utc_string(value, field=field)
            return value
        raise CharactersRAGDBError("Stored shared workspace cursor timestamp is invalid.")

    def _db_datetime(self, value: datetime) -> datetime | str:
        aware = self._aware_utc(value, field="timestamp", reject_naive=False)
        if self._db.backend_type == BackendType.POSTGRESQL:
            return aware
        return aware.isoformat(timespec="microseconds")

    @classmethod
    def _parse_datetime(cls, value: Any) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                raise CharactersRAGDBError("Stored shared workspace timestamp is blank.")
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as exc:
                raise CharactersRAGDBError("Stored shared workspace timestamp is invalid.") from exc
        else:
            raise CharactersRAGDBError("Stored shared workspace timestamp is invalid.")
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @classmethod
    def _optional_datetime(cls, value: Any) -> datetime | None:
        return None if value is None else cls._parse_datetime(value)

    @staticmethod
    def _aware_utc(value: datetime, *, field: str, reject_naive: bool) -> datetime:
        if not isinstance(value, datetime):
            raise InputError(f"{field} must be an aware UTC datetime.")
        if value.tzinfo is None or value.utcoffset() is None:
            if reject_naive:
                raise InputError(f"{field} must be an aware UTC datetime.")
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def _aware_utc_string(cls, value: str, *, field: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise InputError(f"Invalid {field}.") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise InputError(f"Invalid {field}.")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _format_datetime(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )

    @staticmethod
    def _row_dict(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return dict(row)
        mapping = getattr(row, "_mapping", None)
        if mapping is not None:
            return dict(mapping)
        try:
            return dict(row)
        except (TypeError, ValueError) as exc:
            raise CharactersRAGDBError("Database row did not expose named columns.") from exc

    @staticmethod
    def _validate_share_id(share_id: int) -> int:
        if isinstance(share_id, bool) or not isinstance(share_id, int) or share_id < 1:
            raise InputError("share_id must be a positive integer.")
        return share_id

    @staticmethod
    def _validate_request_id(request_id: UUID) -> UUID:
        if not isinstance(request_id, UUID):
            raise InputError("request_id must be a UUID.")
        return request_id

    @staticmethod
    def _validate_source_mode(source_mode: str) -> str:
        if source_mode not in {"all", "include"}:
            raise InputError("source_mode must be 'all' or 'include'.")
        return source_mode

    @staticmethod
    def _bounded_text(value: Any, field: str, max_bytes: int) -> str:
        if not isinstance(value, str):
            raise InputError(f"{field} must be a string.")
        normalized = value.strip()
        if not normalized:
            raise InputError(f"{field} cannot be blank.")
        if len(normalized.encode("utf-8")) > max_bytes:
            raise InputError(f"{field} is too large.")
        return normalized

    def _validate_claim(self, claim: SharedWorkspaceChatClaim) -> None:
        if not isinstance(claim, SharedWorkspaceChatClaim):
            raise InputError("A shared workspace chat claim is required.")
        self._validate_share_id(claim.share_id)
        self._validate_request_id(claim.request_id)
        if claim.disposition != "claimed" or claim.lease_epoch < 1 or not claim.lease_token:
            raise StaleSharedWorkspaceChatClaim


__all__ = [
    "SharedWorkspaceChatClaim",
    "SharedWorkspaceChatStore",
    "SharedWorkspaceChatThread",
    "SharedWorkspaceMessagePage",
    "SharedWorkspaceStoredMessage",
    "StaleSharedWorkspaceChatClaim",
    "StoredSharedWorkspaceTurn",
]
