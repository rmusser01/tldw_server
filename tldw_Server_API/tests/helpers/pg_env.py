"""Centralized Postgres DSN builder for tests.

Precedence for connection settings (container/dev-first):
1) `JOBS_DB_URL` (explicit for Jobs/PG tests), then `POSTGRES_TEST_DSN`
2) `TEST_DATABASE_URL` (used by some AuthNZ tests), then `DATABASE_URL`
3) Container-style envs: `POSTGRES_TEST_*` then `POSTGRES_*`
4) Project defaults aligned with dev compose (tldw_user/TestPassword123!/tldw_content on 127.0.0.1:5432)

This order avoids accidentally picking an unrelated global `TEST_DATABASE_URL`
for Jobs tests while still allowing suites that rely on it to work when no
Jobs-specific DSN is provided.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote, unquote, urlparse


@dataclass
class PGEnv:
    host: str
    port: int
    user: str
    password: str
    database: str
    dsn: str


def _parse_dsn(dsn: str) -> Optional[tuple[str, int, str, str, str]]:
    try:
        p = urlparse(dsn)
        if not p.scheme.startswith("postgres"):
            return None
        host = p.hostname or "127.0.0.1"
        port = int(p.port or 5432)
        user = unquote(p.username or "tldw_user")
        password = unquote(p.password or "TestPassword123!")
        db = unquote((p.path or "/tldw_test").lstrip("/") or "tldw_test")
        return host, port, user, password, db
    except Exception:
        return None


def get_pg_env() -> PGEnv:

    # Prefer Jobs-specific DSNs first, then an explicit test DSN if set
    raw_dsn = (
        os.getenv("JOBS_DB_URL")
        or os.getenv("POSTGRES_TEST_DSN")
        or os.getenv("TEST_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or ""
    ).strip()
    parsed = _parse_dsn(raw_dsn) if raw_dsn else None
    if parsed:
        host, port, user, password, db = parsed
        return PGEnv(host=host, port=port, user=user, password=password, database=db, dsn=raw_dsn)

    # Build from POSTGRES_TEST_* / POSTGRES_* first (container-style), then TEST_DB_* fallbacks
    host = os.getenv("POSTGRES_TEST_HOST") or os.getenv("POSTGRES_HOST") or os.getenv("TEST_DB_HOST") or "127.0.0.1"
    port = int(os.getenv("POSTGRES_TEST_PORT") or os.getenv("POSTGRES_PORT") or os.getenv("TEST_DB_PORT") or "5432")
    user = os.getenv("POSTGRES_TEST_USER") or os.getenv("POSTGRES_USER") or os.getenv("TEST_DB_USER") or "tldw_user"
    password = (
        os.getenv("POSTGRES_TEST_PASSWORD")
        or os.getenv("POSTGRES_PASSWORD")
        or os.getenv("TEST_DB_PASSWORD")
        or "TestPassword123!"
    )
    db = os.getenv("POSTGRES_TEST_DB") or os.getenv("POSTGRES_DB") or os.getenv("TEST_DB_NAME") or "tldw_content"
    dsn = f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{quote(db, safe='')}"
    return PGEnv(host=host, port=port, user=user, password=password, database=db, dsn=dsn)


def pg_dsn() -> str:
    """Return the DSN string honoring the standard precedence."""
    return get_pg_env().dsn
