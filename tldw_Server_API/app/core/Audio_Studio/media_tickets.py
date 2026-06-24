"""Central storage for short-lived Audio Studio artifact media tickets."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseBackend

MEDIA_TICKET_PURPOSES = {"playback", "download"}


@dataclass(frozen=True)
class AudioStudioMediaTicketRow:
    id: int
    token_hash: str
    user_id: int
    project_id: str
    artifact_id: str
    purpose: str
    expires_at: str
    consumed_at: str | None
    revoked_at: str | None
    created_at: str
    created_by_auth_mode: str | None
    last_redeemed_at: str | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_db_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_output_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return to_db_timestamp(value)
    return str(value)


def hash_media_ticket_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _row_to_ticket(row: dict[str, Any] | None) -> AudioStudioMediaTicketRow | None:
    if not row:
        return None
    return AudioStudioMediaTicketRow(
        id=int(row["id"]),
        token_hash=str(row["token_hash"]),
        user_id=int(row["user_id"]),
        project_id=str(row["project_id"]),
        artifact_id=str(row["artifact_id"]),
        purpose=str(row["purpose"]),
        expires_at=normalize_output_timestamp(row["expires_at"]) or "",
        consumed_at=normalize_output_timestamp(row.get("consumed_at")),
        revoked_at=normalize_output_timestamp(row.get("revoked_at")),
        created_at=normalize_output_timestamp(row["created_at"]) or "",
        created_by_auth_mode=row.get("created_by_auth_mode"),
        last_redeemed_at=normalize_output_timestamp(row.get("last_redeemed_at")),
    )


class AudioStudioMediaTicketStore:
    """DB-backed scoped bearer tickets for Audio Studio artifact media."""

    def __init__(self, backend: DatabaseBackend):
        self.backend = backend
        self.ensure_schema()

    def ensure_schema(self) -> None:
        if self.backend.backend_type == BackendType.POSTGRESQL:
            statements = [
                """
                CREATE TABLE IF NOT EXISTS audio_studio_media_tickets (
                    id BIGSERIAL PRIMARY KEY,
                    token_hash TEXT UNIQUE NOT NULL,
                    user_id BIGINT NOT NULL,
                    project_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    purpose TEXT NOT NULL CHECK (purpose IN ('playback', 'download')),
                    expires_at TIMESTAMPTZ NOT NULL,
                    consumed_at TIMESTAMPTZ,
                    revoked_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_by_auth_mode TEXT,
                    last_redeemed_at TIMESTAMPTZ
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_hash ON audio_studio_media_tickets(token_hash)",
                "CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_expiry ON audio_studio_media_tickets(expires_at)",
                "CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_artifact ON audio_studio_media_tickets(user_id, project_id, artifact_id)",
            ]
        else:
            statements = [
                """
                CREATE TABLE IF NOT EXISTS audio_studio_media_tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_hash TEXT UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    project_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    purpose TEXT NOT NULL CHECK (purpose IN ('playback', 'download')),
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT,
                    revoked_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_by_auth_mode TEXT,
                    last_redeemed_at TEXT
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_hash ON audio_studio_media_tickets(token_hash)",
                "CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_expiry ON audio_studio_media_tickets(expires_at)",
                "CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_artifact ON audio_studio_media_tickets(user_id, project_id, artifact_id)",
            ]
        with self.backend.transaction() as conn:
            for statement in statements:
                self.backend.execute(statement, connection=conn)

    def create_ticket(
        self,
        *,
        user_id: int,
        project_id: str,
        artifact_id: str,
        purpose: str,
        expires_at: datetime,
        created_by_auth_mode: str | None,
    ) -> tuple[str, AudioStudioMediaTicketRow]:
        if purpose not in MEDIA_TICKET_PURPOSES:
            raise ValueError("invalid_audio_studio_media_ticket_purpose")
        raw_token = secrets.token_urlsafe(32)
        token_hash = hash_media_ticket_token(raw_token)
        now = to_db_timestamp(utc_now())
        expiry = to_db_timestamp(expires_at)
        if self.backend.backend_type == BackendType.POSTGRESQL:
            sql = (
                "INSERT INTO audio_studio_media_tickets "
                "(token_hash, user_id, project_id, artifact_id, purpose, expires_at, created_at, created_by_auth_mode) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id"
            )
        else:
            sql = (
                "INSERT INTO audio_studio_media_tickets "
                "(token_hash, user_id, project_id, artifact_id, purpose, expires_at, created_at, created_by_auth_mode) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            )
        self.backend.execute(
            sql,
            (token_hash, user_id, project_id, artifact_id, purpose, expiry, now, created_by_auth_mode),
        )
        row = self.get_by_hash(token_hash)
        if row is None:
            raise RuntimeError("audio_studio_media_ticket_insert_failed")
        return raw_token, row

    def get_by_hash(self, token_hash: str) -> AudioStudioMediaTicketRow | None:
        if self.backend.backend_type == BackendType.POSTGRESQL:
            sql = "SELECT * FROM audio_studio_media_tickets WHERE token_hash = %s"
        else:
            sql = "SELECT * FROM audio_studio_media_tickets WHERE token_hash = ?"
        result = self.backend.execute(sql, (token_hash,))
        return _row_to_ticket(result.first)

    def get_by_raw_token(self, raw_token: str) -> AudioStudioMediaTicketRow | None:
        return self.get_by_hash(hash_media_ticket_token(raw_token))

    def touch_redeemed(self, token_hash: str) -> AudioStudioMediaTicketRow | None:
        now = to_db_timestamp(utc_now())
        if self.backend.backend_type == BackendType.POSTGRESQL:
            sql = "UPDATE audio_studio_media_tickets SET last_redeemed_at = %s WHERE token_hash = %s"
        else:
            sql = "UPDATE audio_studio_media_tickets SET last_redeemed_at = ? WHERE token_hash = ?"
        self.backend.execute(sql, (now, token_hash))
        return self.get_by_hash(token_hash)

    def consume_download_ticket(self, token_hash: str) -> AudioStudioMediaTicketRow | None:
        now = to_db_timestamp(utc_now())
        if self.backend.backend_type == BackendType.POSTGRESQL:
            sql = (
                "UPDATE audio_studio_media_tickets "
                "SET consumed_at = %s, last_redeemed_at = %s "
                "WHERE token_hash = %s AND purpose = %s "
                "AND consumed_at IS NULL AND revoked_at IS NULL "
                "AND expires_at > %s"
            )
        else:
            sql = (
                "UPDATE audio_studio_media_tickets "
                "SET consumed_at = ?, last_redeemed_at = ? "
                "WHERE token_hash = ? AND purpose = ? "
                "AND consumed_at IS NULL AND revoked_at IS NULL "
                "AND expires_at > ?"
            )
        result = self.backend.execute(sql, (now, now, token_hash, "download", now))
        if result.rowcount != 1:
            return None
        return self.get_by_hash(token_hash)

    def revoke_ticket(self, token_hash: str) -> AudioStudioMediaTicketRow | None:
        now = to_db_timestamp(utc_now())
        if self.backend.backend_type == BackendType.POSTGRESQL:
            sql = "UPDATE audio_studio_media_tickets SET revoked_at = %s WHERE token_hash = %s"
        else:
            sql = "UPDATE audio_studio_media_tickets SET revoked_at = ? WHERE token_hash = ?"
        self.backend.execute(sql, (now, token_hash))
        return self.get_by_hash(token_hash)

    def cleanup(self, *, retention: timedelta) -> int:
        cutoff = to_db_timestamp(utc_now() - retention)
        if self.backend.backend_type == BackendType.POSTGRESQL:
            sql = (
                "DELETE FROM audio_studio_media_tickets "
                "WHERE expires_at < %s OR consumed_at < %s OR revoked_at < %s"
            )
        else:
            sql = "DELETE FROM audio_studio_media_tickets WHERE expires_at < ? OR consumed_at < ? OR revoked_at < ?"
        result = self.backend.execute(sql, (cutoff, cutoff, cutoff))
        return max(int(result.rowcount or 0), 0)
