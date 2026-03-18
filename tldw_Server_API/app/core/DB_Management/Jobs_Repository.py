"""Repository boundary for Jobs persistence.

This module keeps backend-specific SQL for create-time job operations in one
place so `JobManager` can stay focused on policy and orchestration. The public
API exposes a lightweight `JobsSession` wrapper plus repository methods that
either manage their own transaction scope or reuse a caller-supplied session.
"""

from __future__ import annotations

import contextlib
import sqlite3
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime, timezone as _tz
from pathlib import Path
from typing import Any, Iterator

from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection

try:
    import psycopg  # type: ignore
    from psycopg.rows import dict_row  # type: ignore
except ImportError:  # pragma: no cover - postgres is optional
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


def _normalize_sqlite_datetime(value: datetime | None) -> str | None:
    """Convert datetimes into the naive UTC string format stored in SQLite."""
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(_tz.utc).replace(tzinfo=None)
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_postgres_datetime(value: datetime | None) -> datetime | None:
    """Normalize datetimes into timezone-aware UTC values for PostgreSQL."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=_tz.utc)
    return value.astimezone(_tz.utc)


@dataclass(slots=True)
class JobsSession:
    """Wrap a live Jobs DB connection for a single backend-specific session."""

    backend: str
    conn: Any


class JobsRepository:
    """Encapsulate Jobs create-time SQL for SQLite and PostgreSQL backends."""

    def __init__(
        self,
        *,
        backend: str,
        db_path: Path | None = None,
        db_url: str | None = None,
    ) -> None:
        """Build a repository bound to either a SQLite path or Postgres DSN."""
        self.backend = backend
        self.db_path = Path(db_path) if db_path is not None else None
        self.db_url = db_url

    @classmethod
    def for_sqlite(cls, db_path: Path) -> "JobsRepository":
        """Create a repository configured for the SQLite jobs database."""
        return cls(backend="sqlite", db_path=db_path)

    @classmethod
    def for_postgres(cls, db_url: str) -> "JobsRepository":
        """Create a repository configured for the PostgreSQL jobs database."""
        return cls(backend="postgres", db_url=db_url)

    def _connect(self) -> Any:
        """Open a backend-specific connection with the expected local policy."""
        if self.backend == "postgres":
            if psycopg is None:  # pragma: no cover - guarded by optional dependency
                raise RuntimeError("psycopg is required for postgres jobs repositories")
            return psycopg.connect(self.db_url)
        conn = sqlite3.connect(self.db_path)
        with contextlib.suppress(sqlite3.Error):
            configure_sqlite_connection(conn)
        conn.row_factory = sqlite3.Row
        return conn

    @contextlib.contextmanager
    def session(self) -> Iterator[JobsSession]:
        """Yield a managed session that commits on success and rolls back on failure."""
        conn = self._connect()
        session = JobsSession(backend=self.backend, conn=conn)
        try:
            yield session
            conn.commit()
        except Exception:
            with contextlib.suppress(Exception):
                conn.rollback()
            raise
        finally:
            conn.close()

    def count_active_jobs_for_user(
        self,
        user_id: str,
        *,
        session: JobsSession | None = None,
    ) -> int:
        """Count queued or processing jobs for a user within an optional session."""
        if session is None:
            with self.session() as managed_session:
                return self.count_active_jobs_for_user(user_id, session=managed_session)

        if session.backend == "postgres":
            with session.conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT COUNT(*) AS c FROM jobs WHERE owner_user_id = %s AND status IN ('queued', 'processing')",
                    (user_id,),
                )
                row = cur.fetchone()
                return int(row["c"] if row else 0)

        row = session.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE owner_user_id = ? AND status IN ('queued', 'processing')",
            (user_id,),
        ).fetchone()
        return int(row[0] if row else 0)

    def insert_job(
        self,
        *,
        domain: str,
        queue: str,
        job_type: str,
        payload_json: str,
        owner_user_id: str | None,
        project_id: int | None,
        batch_group: str | None,
        idempotency_key: str | None,
        priority: int,
        max_retries: int,
        available_at: datetime | None,
        request_id: str | None,
        trace_id: str | None,
        created_at: datetime,
        session: JobsSession | None = None,
    ) -> dict[str, Any]:
        """Insert a queued job row and return the created record as a dict."""
        if session is None:
            with self.session() as managed_session:
                return self.insert_job(
                    domain=domain,
                    queue=queue,
                    job_type=job_type,
                    payload_json=payload_json,
                    owner_user_id=owner_user_id,
                    project_id=project_id,
                    batch_group=batch_group,
                    idempotency_key=idempotency_key,
                    priority=priority,
                    max_retries=max_retries,
                    available_at=available_at,
                    request_id=request_id,
                    trace_id=trace_id,
                    created_at=created_at,
                    session=managed_session,
                )

        uuid_val = str(_uuid.uuid4())

        if session.backend == "postgres":
            created_at_pg = _normalize_postgres_datetime(created_at)
            available_at_pg = _normalize_postgres_datetime(available_at)
            with session.conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    (
                        "INSERT INTO jobs (uuid, domain, queue, job_type, owner_user_id, project_id, batch_group, "
                        "idempotency_key, payload, result, status, priority, max_retries, retry_count, available_at, "
                        "created_at, updated_at, request_id, trace_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NULL, 'queued', %s, %s, 0, %s, %s, %s, %s, %s) "
                        "RETURNING *"
                    ),
                    (
                        uuid_val,
                        domain,
                        queue,
                        job_type,
                        owner_user_id,
                        project_id,
                        batch_group,
                        idempotency_key,
                        payload_json,
                        priority,
                        max_retries,
                        available_at_pg,
                        created_at_pg,
                        created_at_pg,
                        request_id,
                        trace_id,
                    ),
                )
                row = cur.fetchone()
                return dict(row) if row else {}

        created_at_sqlite = _normalize_sqlite_datetime(created_at)
        available_at_sqlite = _normalize_sqlite_datetime(available_at)
        session.conn.execute(
            """
            INSERT INTO jobs (
              uuid, domain, queue, job_type, owner_user_id, project_id, batch_group,
              idempotency_key, payload, result, status, priority, max_retries,
              retry_count, available_at, created_at, updated_at, request_id, trace_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'queued', ?, ?, 0, ?, ?, ?, ?, ?)
            """,
            (
                uuid_val,
                domain,
                queue,
                job_type,
                owner_user_id,
                project_id,
                batch_group,
                idempotency_key,
                payload_json,
                priority,
                max_retries,
                available_at_sqlite,
                created_at_sqlite,
                created_at_sqlite,
                request_id,
                trace_id,
            ),
        )
        job_id = session.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        row = session.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else {}
