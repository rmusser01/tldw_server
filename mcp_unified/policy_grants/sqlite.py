"""SQLite-backed TTL-bound policy grant store for the standalone MCP gateway."""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import (
    URL,
    Column,
    Engine,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    select,
)

from .models import PolicyGrant, validate_grant_request


class SQLitePolicyGrantStore:
    """SQLite-backed TTL-bound policy grant store for cooperating local processes."""

    def __init__(
        self,
        path: str | Path,
        *,
        timeout_seconds: float = 30.0,
        token_bytes: int = 24,
        cleanup_interval: int = 64,
        cleanup_limit: int = 512,
    ) -> None:
        raw_path = str(path).strip()
        if not raw_path:
            raise ValueError("SQLite policy grant store requires a database path")
        if raw_path == ":memory:":
            raise ValueError("SQLite policy grant store requires a file-backed database path")

        db_path = Path(raw_path).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(db_path)
        self._token_bytes = max(16, token_bytes)
        self._cleanup_interval = max(1, int(cleanup_interval))
        self._cleanup_limit = max(1, int(cleanup_limit))
        self._operation_count = 0
        self._metadata = MetaData()
        self._table = Table(
            "mcp_policy_grants",
            self._metadata,
            Column("grant_id", String, primary_key=True),
            Column("profile_id", String, nullable=False),
            Column("grant_type", String, nullable=False),
            Column("subject_type", String, nullable=False),
            Column("value", String, nullable=False),
            Column("actions_json", String, nullable=False),
            Column("effect", String, nullable=False),
            Column("session_id", String),
            Column("user_id", String),
            Column("granted_by", String),
            Column("reason", String),
            Column("expires_at_epoch_us", Integer, nullable=False),
            Column("ttl_seconds", Integer, nullable=False),
            Column("created_at_epoch_us", Integer, nullable=False),
        )
        Index(
            "idx_mcp_policy_grants_lookup",
            self._table.c.profile_id,
            self._table.c.grant_type,
            self._table.c.expires_at_epoch_us,
        )
        self._engine: Engine = create_engine(
            URL.create("sqlite", database=self.path),
            connect_args={"timeout": timeout_seconds, "check_same_thread": False},
            future=True,
        )
        self._metadata.create_all(self._engine)

    def create_grant(
        self,
        *,
        profile_id: str,
        grant_type: str,
        subject_type: str,
        value: str,
        ttl_seconds: int,
        actions: tuple[str, ...] = (),
        effect: str = "allow",
        session_id: str | None = None,
        user_id: str | None = None,
        granted_by: str | None = None,
        reason: str | None = None,
    ) -> PolicyGrant:
        """Create one validated grant with a generated opaque id."""

        normalized_profile_id, grant_type, subject_type, normalized_value = validate_grant_request(
            profile_id=profile_id,
            grant_type=grant_type,
            subject_type=subject_type,
            value=value,
        )
        now_us = _now_us()
        ttl = max(1, int(ttl_seconds))
        grant = PolicyGrant(
            grant_id=secrets.token_urlsafe(self._token_bytes),
            profile_id=normalized_profile_id,
            grant_type=grant_type,
            subject_type=subject_type,
            value=normalized_value,
            expires_at=(now_us + ttl * 1_000_000) / 1_000_000,
            ttl_seconds=ttl,
            actions=tuple(actions),
            effect=effect,
            session_id=session_id,
            user_id=user_id,
            granted_by=granted_by,
            reason=reason,
        )
        with self._engine.begin() as connection:
            self._maybe_cleanup_expired(connection, now_us=now_us)
            connection.execute(
                self._table.insert().values(
                    grant_id=grant.grant_id,
                    profile_id=grant.profile_id,
                    grant_type=grant.grant_type,
                    subject_type=grant.subject_type,
                    value=grant.value,
                    actions_json=json.dumps(list(grant.actions)),
                    effect=grant.effect,
                    session_id=grant.session_id,
                    user_id=grant.user_id,
                    granted_by=grant.granted_by,
                    reason=grant.reason,
                    expires_at_epoch_us=now_us + ttl * 1_000_000,
                    ttl_seconds=ttl,
                    created_at_epoch_us=now_us,
                )
            )
        return grant

    def list_active_grants(
        self,
        *,
        profile_id: str | None = None,
        grant_type: str | None = None,
    ) -> list[PolicyGrant]:
        """Return active grants filtered by profile and grant type."""

        now_us = _now_us()
        conditions = [self._table.c.expires_at_epoch_us > now_us]
        if profile_id is not None:
            conditions.append(self._table.c.profile_id == profile_id)
        if grant_type is not None:
            conditions.append(self._table.c.grant_type == grant_type)
        # Read paths never delete expired rows; cleanup runs on writes so
        # read transactions keep shared locks under concurrent load.
        with self._engine.begin() as connection:
            rows = connection.execute(select(self._table).where(*conditions)).mappings().all()
        return [_grant_from_row(row) for row in rows]

    def revoke_grant(self, grant_id: str) -> PolicyGrant | None:
        """Remove one grant by id, returning it when it was still active."""

        now_us = _now_us()
        with self._engine.begin() as connection:
            self._maybe_cleanup_expired(connection, now_us=now_us)
            row = (
                connection.execute(
                    select(self._table).where(self._table.c.grant_id == grant_id)
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            connection.execute(delete(self._table).where(self._table.c.grant_id == grant_id))
        if int(row["expires_at_epoch_us"]) <= now_us:
            return None
        return _grant_from_row(row)

    def find_active_grant(
        self,
        *,
        profile_id: str,
        grant_type: str,
        subject_type: str,
        value: str,
        session_id: str | None = None,
    ) -> PolicyGrant | None:
        """Return one active grant matching the normalized subject, if any."""

        try:
            normalized_profile_id, _, _, normalized_value = validate_grant_request(
                profile_id=profile_id,
                grant_type=grant_type,
                subject_type=subject_type,
                value=value,
            )
        except ValueError:
            return None
        now_us = _now_us()
        with self._engine.begin() as connection:
            rows = (
                connection.execute(
                    select(self._table).where(
                        self._table.c.profile_id == normalized_profile_id,
                        self._table.c.grant_type == grant_type,
                        self._table.c.subject_type == subject_type,
                        self._table.c.value == normalized_value,
                        self._table.c.expires_at_epoch_us > now_us,
                    )
                )
                .mappings()
                .all()
            )
        best: PolicyGrant | None = None
        for row in rows:
            grant = _grant_from_row(row)
            if not grant.matches_session(session_id):
                continue
            if best is None or (best.session_id is None and grant.session_id is not None):
                best = grant
        return best

    def close(self) -> None:
        """Dispose the underlying SQLAlchemy engine."""

        self._engine.dispose()

    def _maybe_cleanup_expired(self, connection: Any, *, now_us: int) -> None:
        self._operation_count += 1
        if self._operation_count % self._cleanup_interval != 0:
            return
        expired_ids = (
            connection.execute(
                select(self._table.c.grant_id)
                .where(self._table.c.expires_at_epoch_us <= now_us)
                .limit(self._cleanup_limit)
            )
            .scalars()
            .all()
        )
        if not expired_ids:
            return
        connection.execute(
            delete(self._table).where(
                self._table.c.expires_at_epoch_us <= now_us,
                self._table.c.grant_id.in_(expired_ids),
            )
        )


def _now_us() -> int:
    return int(time.time() * 1_000_000)


def _grant_from_row(row: Mapping[str, Any]) -> PolicyGrant:
    raw_actions = json.loads(str(row["actions_json"]))
    return PolicyGrant(
        grant_id=str(row["grant_id"]),
        profile_id=str(row["profile_id"]),
        grant_type=str(row["grant_type"]),
        subject_type=str(row["subject_type"]),
        value=str(row["value"]),
        expires_at=int(row["expires_at_epoch_us"]) / 1_000_000,
        ttl_seconds=int(row["ttl_seconds"]),
        actions=tuple(str(action) for action in raw_actions),
        effect=str(row["effect"]),
        session_id=row["session_id"],
        user_id=row["user_id"],
        granted_by=row["granted_by"],
        reason=row["reason"],
    )


__all__ = ["SQLitePolicyGrantStore"]
