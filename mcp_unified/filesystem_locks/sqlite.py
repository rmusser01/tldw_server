"""SQLite-backed advisory lock leases for MCP filesystem tools."""

from __future__ import annotations

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
    and_,
    create_engine,
    delete,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .models import (
    FilesystemLockConflict,
    FilesystemLockLease,
    FilesystemLockMissing,
)


class SQLiteFilesystemLockManager:
    """SQLite-backed advisory lock lease manager for cooperating local processes."""

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
            raise ValueError("SQLite filesystem lock manager requires a database path")
        if raw_path == ":memory:":
            raise ValueError(
                "SQLite filesystem lock manager requires a file-backed database path"
            )

        db_path = Path(path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(db_path)
        self._token_bytes = max(16, token_bytes)
        self._cleanup_interval = max(1, int(cleanup_interval))
        self._cleanup_limit = max(1, int(cleanup_limit))
        self._operation_count = 0
        self._metadata = MetaData()
        self._table = Table(
            "mcp_filesystem_lock_leases",
            self._metadata,
            Column("workspace_key", String, primary_key=True),
            Column("path", String, primary_key=True),
            Column("lease_id", String, nullable=False),
            Column("owner", String, nullable=False),
            Column("expires_at_epoch_us", Integer, nullable=False),
            Column("ttl_seconds", Integer, nullable=False),
            Column("workspace_id", String),
            Column("session_id", String),
            Column("updated_at_epoch_us", Integer, nullable=False),
        )
        Index("idx_mcp_fs_lock_expires_at", self._table.c.expires_at_epoch_us)
        self._engine: Engine = create_engine(
            URL.create("sqlite", database=self.path),
            connect_args={"timeout": timeout_seconds, "check_same_thread": False},
            future=True,
        )
        self._metadata.create_all(self._engine)

    def acquire(
        self,
        *,
        workspace_key: str,
        path: str,
        owner: str,
        ttl_seconds: int,
        lease_id: str | None = None,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[FilesystemLockLease, bool]:
        """Acquire a new lease or renew the current lease when the token matches."""

        now_us = _now_us()
        ttl = max(1, int(ttl_seconds))
        expires_at_epoch_us = now_us + ttl * 1_000_000
        if lease_id is not None:
            return self._renew(
                workspace_key=workspace_key,
                path=path,
                owner=owner,
                ttl_seconds=ttl,
                lease_id=lease_id,
                workspace_id=workspace_id,
                session_id=session_id,
                now_us=now_us,
                expires_at_epoch_us=expires_at_epoch_us,
            )
        return self._acquire_new(
            workspace_key=workspace_key,
            path=path,
            owner=owner,
            ttl_seconds=ttl,
            workspace_id=workspace_id,
            session_id=session_id,
            now_us=now_us,
            expires_at_epoch_us=expires_at_epoch_us,
        )

    def release(
        self,
        *,
        workspace_key: str,
        path: str,
        lease_id: str,
    ) -> FilesystemLockLease | None:
        """Release a lease when the caller presents the active token."""

        now_us = _now_us()
        token = lease_id.strip()
        with self._engine.begin() as connection:
            self._maybe_cleanup_expired(connection, now_us=now_us)
            row = self._select_row(connection, workspace_key=workspace_key, path=path)
            if row is None:
                return None
            if int(row["expires_at_epoch_us"]) <= now_us:
                self._delete_key(
                    connection,
                    workspace_key=workspace_key,
                    path=path,
                    expires_at_or_before_us=now_us,
                )
                return None
            active = _lease_from_row(row)
            if active.lease_id != token:
                raise FilesystemLockConflict(active)

            result = connection.execute(
                delete(self._table).where(
                    self._table.c.workspace_key == workspace_key,
                    self._table.c.path == path,
                    self._table.c.lease_id == token,
                    self._table.c.expires_at_epoch_us > now_us,
                )
            )
            if not _affected_rows(result.rowcount):
                return None
            return active

    def validate(
        self,
        *,
        workspace_key: str,
        path: str,
        lease_id: str,
    ) -> FilesystemLockLease:
        """Return the active lease when the caller presents the current token."""

        now_us = _now_us()
        with self._engine.begin() as connection:
            row = self._select_row(connection, workspace_key=workspace_key, path=path)
            if row is None:
                raise FilesystemLockMissing()
            if int(row["expires_at_epoch_us"]) <= now_us:
                self._delete_key(
                    connection,
                    workspace_key=workspace_key,
                    path=path,
                    expires_at_or_before_us=now_us,
                )
                raise FilesystemLockMissing()
            active = _lease_from_row(row)
            if active.lease_id != lease_id:
                raise FilesystemLockConflict(active)
            return active

    def close(self) -> None:
        """Dispose the underlying SQLAlchemy engine."""

        self._engine.dispose()

    def _acquire_new(
        self,
        *,
        workspace_key: str,
        path: str,
        owner: str,
        ttl_seconds: int,
        workspace_id: str | None,
        session_id: str | None,
        now_us: int,
        expires_at_epoch_us: int,
    ) -> tuple[FilesystemLockLease, bool]:
        values = {
            "workspace_key": workspace_key,
            "path": path,
            "lease_id": secrets.token_urlsafe(self._token_bytes),
            "owner": owner,
            "expires_at_epoch_us": expires_at_epoch_us,
            "ttl_seconds": ttl_seconds,
            "workspace_id": workspace_id,
            "session_id": session_id,
            "updated_at_epoch_us": now_us,
        }
        statement = sqlite_insert(self._table).values(**values)
        upsert = statement.on_conflict_do_update(
            index_elements=[self._table.c.workspace_key, self._table.c.path],
            set_={
                "lease_id": statement.excluded.lease_id,
                "owner": statement.excluded.owner,
                "expires_at_epoch_us": statement.excluded.expires_at_epoch_us,
                "ttl_seconds": statement.excluded.ttl_seconds,
                "workspace_id": statement.excluded.workspace_id,
                "session_id": statement.excluded.session_id,
                "updated_at_epoch_us": statement.excluded.updated_at_epoch_us,
            },
            where=self._table.c.expires_at_epoch_us <= now_us,
        )
        with self._engine.begin() as connection:
            self._maybe_cleanup_expired(connection, now_us=now_us)
            result = connection.execute(upsert)
            if _affected_rows(result.rowcount):
                return _lease_from_row(values), False
            row = self._select_row(connection, workspace_key=workspace_key, path=path)
        raise self._classify_missing_or_conflict(row, now_us=now_us)

    def _renew(
        self,
        *,
        workspace_key: str,
        path: str,
        owner: str,
        ttl_seconds: int,
        lease_id: str,
        workspace_id: str | None,
        session_id: str | None,
        now_us: int,
        expires_at_epoch_us: int,
    ) -> tuple[FilesystemLockLease, bool]:
        values = {
            "workspace_key": workspace_key,
            "path": path,
            "lease_id": lease_id,
            "owner": owner,
            "expires_at_epoch_us": expires_at_epoch_us,
            "ttl_seconds": ttl_seconds,
            "workspace_id": workspace_id,
            "session_id": session_id,
            "updated_at_epoch_us": now_us,
        }
        statement = (
            update(self._table)
            .where(
                self._table.c.workspace_key == workspace_key,
                self._table.c.path == path,
                self._table.c.lease_id == lease_id,
                self._table.c.expires_at_epoch_us > now_us,
            )
            .values(
                owner=owner,
                expires_at_epoch_us=expires_at_epoch_us,
                ttl_seconds=ttl_seconds,
                workspace_id=workspace_id,
                session_id=session_id,
                updated_at_epoch_us=now_us,
            )
        )
        with self._engine.begin() as connection:
            self._maybe_cleanup_expired(connection, now_us=now_us)
            result = connection.execute(statement)
            if _affected_rows(result.rowcount):
                return _lease_from_row(values), True
            row = self._select_row(connection, workspace_key=workspace_key, path=path)
        raise self._classify_missing_or_conflict(row, now_us=now_us)

    def _select_row(
        self,
        connection: Any,
        *,
        workspace_key: str,
        path: str,
    ) -> Mapping[str, Any] | None:
        return (
            connection.execute(
                select(self._table).where(
                    self._table.c.workspace_key == workspace_key,
                    self._table.c.path == path,
                )
            )
            .mappings()
            .first()
        )

    def _delete_key(
        self,
        connection: Any,
        *,
        workspace_key: str,
        path: str,
        expires_at_or_before_us: int | None = None,
    ) -> None:
        conditions = [
            self._table.c.workspace_key == workspace_key,
            self._table.c.path == path,
        ]
        if expires_at_or_before_us is not None:
            conditions.append(
                self._table.c.expires_at_epoch_us <= expires_at_or_before_us
            )
        connection.execute(delete(self._table).where(*conditions))

    def _maybe_cleanup_expired(self, connection: Any, *, now_us: int) -> None:
        self._operation_count += 1
        if self._operation_count % self._cleanup_interval != 0:
            return
        expired_rows = (
            connection.execute(
                select(self._table.c.workspace_key, self._table.c.path)
                .where(self._table.c.expires_at_epoch_us <= now_us)
                .limit(self._cleanup_limit)
            )
            .mappings()
            .all()
        )
        if not expired_rows:
            return
        keys = [
            and_(
                self._table.c.workspace_key == str(row["workspace_key"]),
                self._table.c.path == str(row["path"]),
            )
            for row in expired_rows
        ]
        connection.execute(
            delete(self._table).where(
                self._table.c.expires_at_epoch_us <= now_us,
                or_(*keys),
            )
        )

    def _classify_missing_or_conflict(
        self,
        row: Mapping[str, Any] | None,
        *,
        now_us: int,
    ) -> FilesystemLockMissing | FilesystemLockConflict:
        if row is None or int(row["expires_at_epoch_us"]) <= now_us:
            return FilesystemLockMissing()
        return FilesystemLockConflict(_lease_from_row(row))


def _now_us() -> int:
    return int(time.time() * 1_000_000)


def _lease_from_row(row: Mapping[str, Any]) -> FilesystemLockLease:
    return FilesystemLockLease(
        workspace_key=str(row["workspace_key"]),
        path=str(row["path"]),
        lease_id=str(row["lease_id"]),
        owner=str(row["owner"]),
        expires_at=int(row["expires_at_epoch_us"]) / 1_000_000,
        ttl_seconds=int(row["ttl_seconds"]),
        workspace_id=row["workspace_id"],
        session_id=row["session_id"],
    )


def _affected_rows(rowcount: int | None) -> bool:
    return rowcount is not None and rowcount > 0


__all__ = ["SQLiteFilesystemLockManager"]
