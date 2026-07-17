"""Shared persistence for short-lived chat document upload drafts."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from tldw_Server_API.app.core.Ingestion_Media_Processing.document_upload_preflight import (
    DEFAULT_MAX_CHAT_UPLOAD_BYTES,
)
from tldw_Server_API.app.core.Utils.Utils import get_project_root

DRAFT_TTL_SECONDS = 15 * 60
MAX_DRAFT_PAYLOAD_BYTES = DEFAULT_MAX_CHAT_UPLOAD_BYTES
MAX_DRAFTS_TOTAL = 256
MAX_DRAFTS_PER_OWNER = 32


class DocumentUploadDraftError(Exception):
    """Base error for document upload draft persistence."""


class DocumentUploadDraftPayloadTooLargeError(DocumentUploadDraftError):
    """Raised when a draft payload exceeds the configured byte limit."""


class DocumentUploadDraftQuotaError(DocumentUploadDraftError):
    """Raised when a draft count quota has been reached."""


@dataclass(frozen=True, slots=True)
class DocumentUploadDraft:
    """A persisted document upload handoff draft."""

    draft_id: str
    owner: str
    created_at: datetime
    expires_at: datetime
    payload: dict[str, Any]


class DocumentUploadDraftStore:
    """Persist upload drafts in SQLite for cross-worker visibility."""

    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        ttl_seconds: int = DRAFT_TTL_SECONDS,
        max_payload_bytes: int = MAX_DRAFT_PAYLOAD_BYTES,
        max_drafts_total: int = MAX_DRAFTS_TOTAL,
        max_drafts_per_owner: int = MAX_DRAFTS_PER_OWNER,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the store and its schema."""
        self.db_path = Path(db_path) if db_path is not None else self._default_db_path()
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_payload_bytes = max(1, int(max_payload_bytes))
        self.max_drafts_total = max(1, int(max_drafts_total))
        self.max_drafts_per_owner = max(1, int(max_drafts_per_owner))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @staticmethod
    def _default_db_path() -> Path:
        """Return the shared repository database path for upload drafts."""
        return Path(get_project_root()) / "Databases" / "document_upload_drafts.db"

    def _connect(self) -> sqlite3.Connection:
        """Open a configured SQLite connection for one operation."""
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize_schema(self) -> None:
        """Create the draft table and lookup indexes when absent."""
        with contextlib.closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_upload_drafts (
                    draft_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_upload_drafts_owner " "ON document_upload_drafts(owner)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_upload_drafts_expires " "ON document_upload_drafts(expires_at)"
            )

    def _now(self) -> datetime:
        """Return an aware UTC timestamp from the configured clock."""
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _cleanup_expired(connection: sqlite3.Connection, now_timestamp: float) -> None:
        """Delete expired rows inside the caller's transaction."""
        connection.execute(
            "DELETE FROM document_upload_drafts WHERE expires_at <= ?",
            (now_timestamp,),
        )

    def create(self, *, owner: str, payload: dict[str, Any]) -> DocumentUploadDraft:
        """Persist a draft after atomically applying TTL and quota rules."""
        payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        if len(payload_json.encode("utf-8")) > self.max_payload_bytes:
            raise DocumentUploadDraftPayloadTooLargeError("Draft payload exceeds upload limit")

        now = self._now()
        expires_at = now + timedelta(seconds=self.ttl_seconds)
        draft_id = uuid4().hex
        with contextlib.closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            self._cleanup_expired(connection, now.timestamp())
            owner_count = connection.execute(
                "SELECT COUNT(*) FROM document_upload_drafts WHERE owner = ?",
                (owner,),
            ).fetchone()[0]
            if owner_count >= self.max_drafts_per_owner:
                raise DocumentUploadDraftQuotaError("Too many active document upload drafts for this user")
            total_count = connection.execute("SELECT COUNT(*) FROM document_upload_drafts").fetchone()[0]
            if total_count >= self.max_drafts_total:
                raise DocumentUploadDraftQuotaError("Too many active document upload drafts")
            connection.execute(
                """
                INSERT INTO document_upload_drafts (
                    draft_id, owner, created_at, expires_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    draft_id,
                    owner,
                    now.timestamp(),
                    expires_at.timestamp(),
                    payload_json,
                ),
            )
        return DocumentUploadDraft(
            draft_id=draft_id,
            owner=owner,
            created_at=now,
            expires_at=expires_at,
            payload=payload,
        )

    def get(self, *, owner: str, draft_id: str) -> DocumentUploadDraft | None:
        """Return an unexpired draft owned by the caller."""
        now = self._now()
        with contextlib.closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT draft_id, owner, created_at, expires_at, payload_json
                FROM document_upload_drafts
                WHERE draft_id = ? AND owner = ? AND expires_at > ?
                """,
                (draft_id, owner, now.timestamp()),
            ).fetchone()
        if row is None:
            return None
        return DocumentUploadDraft(
            draft_id=str(row["draft_id"]),
            owner=str(row["owner"]),
            created_at=datetime.fromtimestamp(float(row["created_at"]), timezone.utc),
            expires_at=datetime.fromtimestamp(float(row["expires_at"]), timezone.utc),
            payload=json.loads(str(row["payload_json"])),
        )

    def delete(self, *, owner: str, draft_id: str) -> bool:
        """Delete an owned, unexpired draft and report whether it existed."""
        with contextlib.closing(self._connect()) as connection, connection:
            self._cleanup_expired(connection, self._now().timestamp())
            cursor = connection.execute(
                "DELETE FROM document_upload_drafts WHERE draft_id = ? AND owner = ?",
                (draft_id, owner),
            )
        return cursor.rowcount > 0


@lru_cache(maxsize=1)
def get_document_upload_draft_store() -> DocumentUploadDraftStore:
    """Return the process-local handle to the shared SQLite draft database."""
    return DocumentUploadDraftStore()
