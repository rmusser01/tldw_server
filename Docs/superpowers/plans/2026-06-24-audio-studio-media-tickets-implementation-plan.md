# Audio Studio Media Tickets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DB-backed Audio Studio media tickets so large audio can stream through native browser playback and large or non-audio artifacts can download without putting API keys, JWTs, provider secrets, or filesystem paths into DOM URLs.

**Architecture:** Authenticated mint requests create short-lived, scoped bearer tickets in the global AuthNZ database using only SHA-256 token hashes at rest. Unauthenticated redemption uses the token hash to recover the owning user/project/artifact, reopens the correct user-scoped Collections DB, repeats artifact/path validation, and streams the file with playback or download semantics. The frontend keeps small Blob previews unchanged and uses runtime-only ticket URLs for over-threshold or unknown-size audio and direct user-click downloads.

**Tech Stack:** FastAPI, Pydantic, Loguru, `DatabaseBackend`, SQLite/PostgreSQL schema bootstrap, React, Ant Design, Vitest, pytest, Bandit.

---

## File Structure

- Create `tldw_Server_API/app/core/Audio_Studio/media_tickets.py`
  - Owns token generation, token hashing, central table creation, ticket row mapping, lookup, audit touch, revocation, cleanup, and atomic download consumption.
- Modify `tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql`
  - Adds the `audio_studio_media_tickets` table and indexes for new SQLite AuthNZ DB bootstrap.
- Modify `tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql`
  - Adds the same table and indexes for new PostgreSQL AuthNZ DB bootstrap.
- Modify `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py`
  - Adds the table to embedded fallback schema statements so `UserDatabase_v2` remains complete when schema files are absent.
- Create `tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_media_ticket_store.py`
  - Tests central storage, raw-token exclusion, expiry lookup, cleanup, and atomic single-use consumption on SQLite.
- Modify `tldw_Server_API/app/api/v1/schemas/audio_studio_schemas.py`
  - Adds ticket purpose enum, mint request model, and response model.
- Modify `tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py`
  - Adds mint and redeem endpoints, generalizes artifact path helpers, adds generic download MIME checks, and reuses range support for playback tickets.
- Modify `tldw_Server_API/app/core/Logging/access_log_middleware.py`
  - Redacts `/api/v1/audio-studio/media-tickets/{token}` before binding or emitting access log paths.
- Create `tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_media_tickets_api.py`
  - Tests mint/redeem behavior, range playback, single-use downloads, cross-user rejection, dangerous blocks, revalidation failures, and no `Cross-Origin-Resource-Policy: same-origin`.
- Create `tldw_Server_API/tests/Logging/test_access_log_redaction.py`
  - Tests app access log token redaction.
- Modify `apps/packages/ui/src/services/audio-studio.ts`
  - Adds ticket types and `mintAudioStudioArtifactMediaTicket(projectId, artifactId, purpose)` that resolves a browser-usable runtime URL.
- Modify `apps/packages/ui/src/services/__tests__/audio-studio.test.ts`
  - Tests ticket mint payloads, absolute URL resolution, and backend-provided `ticket_url` preference.
- Modify `apps/packages/ui/src/components/Option/AudioStudio/TimelineEditor.tsx`
  - Keeps small Blob preview/download behavior; adds ticket playback for over-threshold or unknown-size audio; adds click-only download tickets for large audio and non-audio artifacts; retries playback remint once on media error.
- Modify `apps/packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx`
  - Updates old large/unknown-size expectations and adds ticket playback/download tests.
- Modify `Docs/Audio_Studio.md`
  - Documents the media ticket contract, TTLs, security model, log redaction, and reverse-proxy redaction operator responsibility.
- Modify `backlog/tasks/task-2358 - Add-Audio-Studio-large-artifact-WebUI-transport.md`
  - Adds this plan path, records verification commands/results, and updates final status when implementation completes.

## Task 1: Central Media Ticket Store

**Files:**
- Create: `tldw_Server_API/app/core/Audio_Studio/media_tickets.py`
- Modify: `tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql`
- Modify: `tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql`
- Modify: `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py`
- Test: `tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_media_ticket_store.py`

- [ ] **Step 1: Write failing store tests**

Create `tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_media_ticket_store.py`:

```python
"""Unit tests for Audio Studio media ticket central storage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Audio_Studio.media_tickets import (
    AudioStudioMediaTicketStore,
    hash_media_ticket_token,
)
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory


pytestmark = pytest.mark.unit


@pytest.fixture()
def ticket_store(tmp_path):
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "users.db"))
    )
    return AudioStudioMediaTicketStore(backend)


def _expires(minutes: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


def test_create_ticket_stores_hash_not_raw_token(ticket_store: AudioStudioMediaTicketStore) -> None:
    raw_token, row = ticket_store.create_ticket(
        user_id=7,
        project_id="project-1",
        artifact_id="artifact-1",
        purpose="playback",
        expires_at=_expires(30),
        created_by_auth_mode="single_user",
    )

    assert len(raw_token) >= 32
    assert row.token_hash == hash_media_ticket_token(raw_token)
    assert row.user_id == 7
    assert row.project_id == "project-1"
    assert row.artifact_id == "artifact-1"
    assert row.purpose == "playback"

    stored = ticket_store.backend.execute(
        "SELECT token_hash, project_id, artifact_id FROM audio_studio_media_tickets"
    ).one
    assert stored["token_hash"] == row.token_hash
    assert raw_token not in str(stored)


def test_lookup_rejects_unknown_and_returns_existing_ticket(ticket_store: AudioStudioMediaTicketStore) -> None:
    raw_token, created = ticket_store.create_ticket(
        user_id=7,
        project_id="project-1",
        artifact_id="artifact-1",
        purpose="download",
        expires_at=_expires(10),
        created_by_auth_mode=None,
    )

    assert ticket_store.get_by_raw_token(raw_token).id == created.id
    assert ticket_store.get_by_raw_token("not-the-ticket") is None


def test_download_ticket_is_consumed_once_atomically(ticket_store: AudioStudioMediaTicketStore) -> None:
    raw_token, created = ticket_store.create_ticket(
        user_id=7,
        project_id="project-1",
        artifact_id="artifact-1",
        purpose="download",
        expires_at=_expires(10),
        created_by_auth_mode="jwt",
    )

    consumed = ticket_store.consume_download_ticket(created.token_hash)
    repeated = ticket_store.consume_download_ticket(created.token_hash)

    assert consumed is not None
    assert consumed.consumed_at is not None
    assert repeated is None
    assert ticket_store.get_by_raw_token(raw_token).consumed_at is not None


def test_playback_ticket_touch_does_not_consume(ticket_store: AudioStudioMediaTicketStore) -> None:
    raw_token, created = ticket_store.create_ticket(
        user_id=7,
        project_id="project-1",
        artifact_id="artifact-1",
        purpose="playback",
        expires_at=_expires(30),
        created_by_auth_mode="jwt",
    )

    touched = ticket_store.touch_redeemed(created.token_hash)

    assert touched is not None
    assert ticket_store.get_by_raw_token(raw_token).consumed_at is None
    assert ticket_store.get_by_raw_token(raw_token).last_redeemed_at is not None


def test_cleanup_removes_old_expired_consumed_and_revoked_rows(ticket_store: AudioStudioMediaTicketStore) -> None:
    expired_raw, _ = ticket_store.create_ticket(
        user_id=7,
        project_id="project-1",
        artifact_id="expired",
        purpose="playback",
        expires_at=datetime.now(timezone.utc) - timedelta(days=2),
        created_by_auth_mode=None,
    )
    consumed_raw, consumed = ticket_store.create_ticket(
        user_id=7,
        project_id="project-1",
        artifact_id="consumed",
        purpose="download",
        expires_at=_expires(10),
        created_by_auth_mode=None,
    )
    revoked_raw, revoked = ticket_store.create_ticket(
        user_id=7,
        project_id="project-1",
        artifact_id="revoked",
        purpose="playback",
        expires_at=_expires(10),
        created_by_auth_mode=None,
    )
    active_raw, _ = ticket_store.create_ticket(
        user_id=7,
        project_id="project-1",
        artifact_id="active",
        purpose="playback",
        expires_at=_expires(30),
        created_by_auth_mode=None,
    )

    assert ticket_store.consume_download_ticket(consumed.token_hash) is not None
    ticket_store.revoke_ticket(revoked.token_hash)
    ticket_store.backend.execute(
        "UPDATE audio_studio_media_tickets SET consumed_at = ? WHERE token_hash = ?",
        ("2000-01-01T00:00:00Z", consumed.token_hash),
    )
    ticket_store.backend.execute(
        "UPDATE audio_studio_media_tickets SET revoked_at = ? WHERE token_hash = ?",
        ("2000-01-01T00:00:00Z", revoked.token_hash),
    )
    deleted = ticket_store.cleanup(retention=timedelta(seconds=0))

    assert deleted >= 3
    assert ticket_store.get_by_raw_token(expired_raw) is None
    assert ticket_store.get_by_raw_token(consumed_raw) is None
    assert ticket_store.get_by_raw_token(revoked_raw) is None
    assert ticket_store.get_by_raw_token(active_raw) is not None
```

- [ ] **Step 2: Run store tests and verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_media_ticket_store.py -v
```

Expected: fail during import with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Audio_Studio.media_tickets'`.

- [ ] **Step 3: Implement the ticket store**

Create `tldw_Server_API/app/core/Audio_Studio/media_tickets.py`:

```python
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
        expires_at=str(row["expires_at"]),
        consumed_at=row.get("consumed_at"),
        revoked_at=row.get("revoked_at"),
        created_at=str(row["created_at"]),
        created_by_auth_mode=row.get("created_by_auth_mode"),
        last_redeemed_at=row.get("last_redeemed_at"),
    )


class AudioStudioMediaTicketStore:
    """DB-backed scoped bearer tickets for Audio Studio artifact media."""

    def __init__(self, backend: DatabaseBackend):
        self.backend = backend
        self.ensure_schema()

    @property
    def _param(self) -> str:
        return "%s" if self.backend.backend_type == BackendType.POSTGRESQL else "?"

    def _params(self, count: int) -> str:
        return ", ".join([self._param] * count)

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
        parameter_slots = self._params(8)
        sql = (
            "INSERT INTO audio_studio_media_tickets "
            "(token_hash, user_id, project_id, artifact_id, purpose, expires_at, created_at, created_by_auth_mode) "
            f"VALUES ({parameter_slots})"
        )
        if self.backend.backend_type == BackendType.POSTGRESQL:
            sql += " RETURNING id"
        result = self.backend.execute(
            sql,
            (token_hash, user_id, project_id, artifact_id, purpose, expiry, now, created_by_auth_mode),
        )
        row = self.get_by_hash(token_hash)
        if row is None:
            raise RuntimeError("audio_studio_media_ticket_insert_failed")
        return raw_token, row

    def get_by_hash(self, token_hash: str) -> AudioStudioMediaTicketRow | None:
        result = self.backend.execute(
            f"SELECT * FROM audio_studio_media_tickets WHERE token_hash = {self._param}",
            (token_hash,),
        )
        return _row_to_ticket(result.first)

    def get_by_raw_token(self, raw_token: str) -> AudioStudioMediaTicketRow | None:
        return self.get_by_hash(hash_media_ticket_token(raw_token))

    def touch_redeemed(self, token_hash: str) -> AudioStudioMediaTicketRow | None:
        now = to_db_timestamp(utc_now())
        self.backend.execute(
            f"UPDATE audio_studio_media_tickets SET last_redeemed_at = {self._param} WHERE token_hash = {self._param}",
            (now, token_hash),
        )
        return self.get_by_hash(token_hash)

    def consume_download_ticket(self, token_hash: str) -> AudioStudioMediaTicketRow | None:
        now = to_db_timestamp(utc_now())
        sql = (
            "UPDATE audio_studio_media_tickets "
            f"SET consumed_at = {self._param}, last_redeemed_at = {self._param} "
            f"WHERE token_hash = {self._param} AND purpose = {self._param} "
            "AND consumed_at IS NULL AND revoked_at IS NULL "
            f"AND expires_at > {self._param}"
        )
        result = self.backend.execute(sql, (now, now, token_hash, "download", now))
        if result.rowcount != 1:
            return None
        return self.get_by_hash(token_hash)

    def revoke_ticket(self, token_hash: str) -> AudioStudioMediaTicketRow | None:
        now = to_db_timestamp(utc_now())
        self.backend.execute(
            f"UPDATE audio_studio_media_tickets SET revoked_at = {self._param} WHERE token_hash = {self._param}",
            (now, token_hash),
        )
        return self.get_by_hash(token_hash)

    def cleanup(self, *, retention: timedelta) -> int:
        cutoff = to_db_timestamp(utc_now() - retention)
        result = self.backend.execute(
            "DELETE FROM audio_studio_media_tickets "
            f"WHERE expires_at < {self._param} OR consumed_at < {self._param} OR revoked_at < {self._param}",
            (cutoff, cutoff, cutoff),
        )
        return max(int(result.rowcount or 0), 0)
```

- [ ] **Step 4: Add central schema bootstrap SQL**

Append this SQLite block before the indexes section in `tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql`:

```sql
-- Audio Studio scoped media tickets
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
);
```

Append these SQLite indexes with the other `CREATE INDEX` statements:

```sql
CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_hash ON audio_studio_media_tickets(token_hash);
CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_expiry ON audio_studio_media_tickets(expires_at);
CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_artifact ON audio_studio_media_tickets(user_id, project_id, artifact_id);
```

Append this PostgreSQL block near the bootstrap tables in `tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql`:

```sql
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
);
```

Append these PostgreSQL indexes with the other indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_hash ON audio_studio_media_tickets(token_hash);
CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_expiry ON audio_studio_media_tickets(expires_at);
CREATE INDEX IF NOT EXISTS idx_audio_studio_media_tickets_artifact ON audio_studio_media_tickets(user_id, project_id, artifact_id);
```

- [ ] **Step 5: Add embedded fallback schema statements**

In `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py`, add the same table and index statements to `_default_schema_statements_sqlite()` and `_default_schema_statements_postgres()` after `auth_audit_log`. Use SQLite `TEXT` timestamps and PostgreSQL `TIMESTAMPTZ`, matching Step 4.

- [ ] **Step 6: Run store tests and verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_media_ticket_store.py -v
```

Expected: all tests in `test_audio_studio_media_ticket_store.py` pass.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add tldw_Server_API/app/core/Audio_Studio/media_tickets.py tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_media_ticket_store.py
git commit -m "feat(audio-studio): add central media ticket store"
```

## Task 2: Backend Ticket API And Redemption

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/audio_studio_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py`
- Test: `tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_media_tickets_api.py`

- [ ] **Step 1: Write failing ticket API integration tests**

Create `tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_media_tickets_api.py`:

```python
"""Integration tests for Audio Studio artifact media tickets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio_studio import router as audio_studio_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.db_config import AuthDatabaseConfig
from tldw_Server_API.app.core.Audio_Studio.media_tickets import hash_media_ticket_token
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


pytestmark = pytest.mark.integration

MEDIA_BYTES = b"0123456789abcdefghijklmnopqrstuvwxyz"


@pytest.fixture()
def client_audio_studio_tickets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    AuthDatabaseConfig().reset()

    app = FastAPI()
    app.include_router(audio_studio_router, prefix="/api/v1")

    async def override_user():
        return User(id=1, username="tester", email="t@example.test", is_active=True, is_admin=True)

    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client, tmp_path
    app.dependency_overrides.clear()
    AuthDatabaseConfig().reset()


def _outputs_dir(user_id: int = 1) -> Path:
    outputs_dir = DatabasePaths.get_user_base_directory(user_id) / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def _write_user_output(filename: str, data: bytes = MEDIA_BYTES, *, user_id: int = 1) -> Path:
    path = _outputs_dir(user_id) / filename
    path.write_bytes(data)
    return path


def _create_project(client: TestClient, *, title: str = "Ticket project") -> dict:
    response = client.post("/api/v1/audio-studio/projects", json={"title": title, "workflow": "narration"})
    assert response.status_code == 200  # nosec B101
    return response.json()


def _create_artifact(
    project: dict,
    *,
    artifact_id: str = "artifact_ticket",
    storage_path: str = "clip.wav",
    mime_type: str | None = "audio/wav",
    artifact_type: str = "clip_audio",
    size_bytes: int | None = None,
    content_hash: str | None = None,
    user_id: int = 1,
    normalize_storage_path: bool = True,
) -> dict:
    db = CollectionsDatabase.for_user(user_id=user_id)
    project_row = db.get_audio_studio_project_by_project_id(project["project_id"])
    normalized_storage_path = (
        db.resolve_output_storage_path(storage_path)
        if normalize_storage_path and not Path(storage_path).is_absolute() and os.sep not in storage_path
        else storage_path
    )
    row = db.create_audio_studio_artifact(
        project_row_id=project_row.id,
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        provider="audio_studio",
        output_id=None,
        storage_path=normalized_storage_path,
        mime_type=mime_type,
        size_bytes=len(MEDIA_BYTES) if size_bytes is None else size_bytes,
        source_resource_kind="clip",
        source_resource_id="clip-1",
        source_revision_id=project["current_revision_id"],
        content_hash=content_hash or hashlib.sha256(MEDIA_BYTES).hexdigest(),
        metadata_json=json.dumps({"filename": storage_path}),
    )
    return {"artifact_id": row.artifact_id, "storage_path": row.storage_path}


def _mint_url(project_id: str, artifact_id: str) -> str:
    return f"/api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/tickets"


def test_mint_playback_ticket_and_redeem_range(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    _write_user_output("clip.wav")
    artifact = _create_artifact(project)

    mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})

    assert mint.status_code == 200  # nosec B101
    payload = mint.json()
    assert payload["ticket_path"].startswith("/api/v1/audio-studio/media-tickets/")
    assert payload["purpose"] == "playback"
    assert payload["artifact_id"] == artifact["artifact_id"]
    token = payload["ticket_path"].rsplit("/", 1)[1]

    users_db = AuthDatabaseConfig().get_user_database(client_id="ticket-test")
    rows = users_db.backend.execute("SELECT token_hash FROM audio_studio_media_tickets").rows
    assert rows == [{"token_hash": hash_media_ticket_token(token)}]  # nosec B101
    assert token not in str(rows)  # nosec B101

    redeemed = client.get(payload["ticket_path"], headers={"Range": "bytes=0-9"})

    assert redeemed.status_code == 206  # nosec B101
    assert redeemed.content == MEDIA_BYTES[:10]  # nosec B101
    assert redeemed.headers["content-range"] == f"bytes 0-9/{len(MEDIA_BYTES)}"  # nosec B101
    assert redeemed.headers["content-disposition"].startswith("inline;")  # nosec B101
    assert redeemed.headers["cache-control"] == "private, no-store"  # nosec B101
    assert redeemed.headers["referrer-policy"] == "no-referrer"  # nosec B101
    assert redeemed.headers["x-content-type-options"] == "nosniff"  # nosec B101
    assert redeemed.headers.get("cross-origin-resource-policy") != "same-origin"  # nosec B101


def test_rejects_playback_ticket_for_non_audio_artifact(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    _write_user_output("manifest.json", b'{"ok": true}')
    artifact = _create_artifact(
        project,
        artifact_id="artifact_json",
        storage_path="manifest.json",
        mime_type="application/json",
        artifact_type="export_manifest",
        size_bytes=len(b'{"ok": true}'),
        content_hash=hashlib.sha256(b'{"ok": true}').hexdigest(),
    )

    response = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})

    assert response.status_code == 415  # nosec B101


def test_download_ticket_for_non_audio_is_single_use_and_ignores_range(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    data = b'{"chapters": []}'
    _write_user_output("manifest.json", data)
    artifact = _create_artifact(
        project,
        artifact_id="artifact_manifest",
        storage_path="manifest.json",
        mime_type="application/json",
        artifact_type="export_manifest",
        size_bytes=len(data),
        content_hash=hashlib.sha256(data).hexdigest(),
    )

    mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "download"})
    assert mint.status_code == 200  # nosec B101
    ticket_path = mint.json()["ticket_path"]

    first = client.get(ticket_path, headers={"Range": "bytes=0-3"})
    second = client.get(ticket_path)

    assert first.status_code == 200  # nosec B101
    assert first.content == data  # nosec B101
    assert first.headers["content-disposition"].startswith("attachment;")  # nosec B101
    assert "accept-ranges" not in {key.lower(): value for key, value in first.headers.items()}  # nosec B101
    assert second.status_code == 410  # nosec B101
    assert second.json()["detail"] == "audio_studio_media_ticket_consumed"  # nosec B101


def test_ticket_mint_does_not_cross_users_or_projects(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client, title="owner project")
    other_project = _create_project(client, title="other project")
    _write_user_output("clip.wav")
    other_project_artifact = _create_artifact(other_project, artifact_id="artifact_other_project")

    wrong_project = client.post(
        _mint_url(project["project_id"], other_project_artifact["artifact_id"]),
        json={"purpose": "playback"},
    )
    assert wrong_project.status_code == 404  # nosec B101

    client.app.dependency_overrides[get_request_user] = (
        lambda: User(id=2, username="other", email="o@example.test", is_active=True, is_admin=False)
    )
    assert client.post(_mint_url(project["project_id"], other_project_artifact["artifact_id"]), json={"purpose": "playback"}).status_code == 404  # nosec B101


@pytest.mark.parametrize(
    ("filename", "mime_type"),
    [
        ("page.html", "text/html"),
        ("vector.svg", "image/svg+xml"),
        ("script.js", "application/javascript"),
        ("binary.exe", "application/x-msdownload"),
        ("runner.sh", "text/x-shellscript"),
    ],
)
def test_download_ticket_blocks_dangerous_content(client_audio_studio_tickets, filename: str, mime_type: str) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    data = b"unsafe"
    _write_user_output(filename, data)
    artifact = _create_artifact(
        project,
        artifact_id=f"artifact_{filename.replace('.', '_')}",
        storage_path=filename,
        mime_type=mime_type,
        artifact_type="export_artifact",
        size_bytes=len(data),
        content_hash=hashlib.sha256(data).hexdigest(),
    )

    response = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "download"})

    assert response.status_code == 415  # nosec B101


def test_ticket_redemption_returns_404_when_backing_file_is_removed(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    path = _write_user_output("clip.wav")
    artifact = _create_artifact(project, size_bytes=len(MEDIA_BYTES))
    mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})
    assert mint.status_code == 200  # nosec B101
    path.unlink()

    response = client.get(mint.json()["ticket_path"])

    assert response.status_code == 404  # nosec B101
    assert str(path) not in response.text  # nosec B101


def test_ticket_redemption_revalidates_size_mismatch(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    path = _write_user_output("clip.wav")
    artifact = _create_artifact(project, size_bytes=len(MEDIA_BYTES))
    mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})
    assert mint.status_code == 200  # nosec B101
    path.write_bytes(MEDIA_BYTES + b"changed")

    response = client.get(mint.json()["ticket_path"])

    assert response.status_code == 409  # nosec B101
    assert response.json()["detail"] == "audio_studio_artifact_size_mismatch"  # nosec B101


def test_ticket_redemption_rejects_symlink_escape_after_mint(client_audio_studio_tickets) -> None:
    client, tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    path = _write_user_output("clip.wav")
    artifact = _create_artifact(project, size_bytes=len(MEDIA_BYTES))
    mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})
    assert mint.status_code == 200  # nosec B101
    outside = tmp_path / "outside.wav"
    outside.write_bytes(MEDIA_BYTES)
    path.unlink()
    path.symlink_to(outside)

    response = client.get(mint.json()["ticket_path"])

    assert response.status_code == 400  # nosec B101
    assert response.json()["detail"] == "invalid_audio_studio_artifact_path"  # nosec B101
    assert str(outside) not in response.text  # nosec B101


def test_expired_and_revoked_tickets_return_stable_gone_errors(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    _write_user_output("clip.wav")
    artifact = _create_artifact(project, size_bytes=len(MEDIA_BYTES))

    expired_mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})
    revoked_mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})
    assert expired_mint.status_code == 200  # nosec B101
    assert revoked_mint.status_code == 200  # nosec B101
    expired_token = expired_mint.json()["ticket_path"].rsplit("/", 1)[1]
    revoked_token = revoked_mint.json()["ticket_path"].rsplit("/", 1)[1]

    users_db = AuthDatabaseConfig().get_user_database(client_id="ticket-test")
    users_db.backend.execute(
        "UPDATE audio_studio_media_tickets SET expires_at = ? WHERE token_hash = ?",
        ("2000-01-01T00:00:00Z", hash_media_ticket_token(expired_token)),
    )
    users_db.backend.execute(
        "UPDATE audio_studio_media_tickets SET revoked_at = ? WHERE token_hash = ?",
        ("2026-06-24T00:00:00Z", hash_media_ticket_token(revoked_token)),
    )

    expired_response = client.get(expired_mint.json()["ticket_path"])
    revoked_response = client.get(revoked_mint.json()["ticket_path"])

    assert expired_response.status_code == 410  # nosec B101
    assert expired_response.json()["detail"] == "audio_studio_media_ticket_expired"  # nosec B101
    assert revoked_response.status_code == 410  # nosec B101
    assert revoked_response.json()["detail"] == "audio_studio_media_ticket_revoked"  # nosec B101


def test_unknown_and_malformed_ticket_paths_are_generic_not_found(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets

    malformed = client.get("/api/v1/audio-studio/media-tickets/not valid")
    unknown = client.get("/api/v1/audio-studio/media-tickets/unknown-ticket-token")

    assert malformed.status_code == 404  # nosec B101
    assert unknown.status_code == 404  # nosec B101
```

- [ ] **Step 2: Run ticket API tests and verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_media_tickets_api.py -v
```

Expected: fail with route `404` for the mint endpoint or import errors for ticket schemas.

- [ ] **Step 3: Add ticket request and response schemas**

In `tldw_Server_API/app/api/v1/schemas/audio_studio_schemas.py`, add:

```python
class AudioStudioMediaTicketPurpose(str, Enum):
    PLAYBACK = "playback"
    DOWNLOAD = "download"


class AudioStudioMediaTicketCreate(_BaseAudioStudioModel):
    purpose: AudioStudioMediaTicketPurpose


class AudioStudioMediaTicketResponse(_BaseAudioStudioModel):
    ticket_path: str
    ticket_url: str | None = None
    expires_at: str
    purpose: AudioStudioMediaTicketPurpose
    artifact_id: str
```

Add `AudioStudioMediaTicketCreate`, `AudioStudioMediaTicketPurpose`, and `AudioStudioMediaTicketResponse` to the imports in `audio_studio.py`.

- [ ] **Step 4: Generalize artifact path and MIME helpers**

In `tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py`, replace audio-only helper names with generic helpers while keeping existing media endpoint behavior:

```python
_DANGEROUS_DOWNLOAD_SUFFIXES = {
    ".app", ".bat", ".cmd", ".com", ".cpl", ".dll", ".dmg", ".exe", ".hta",
    ".htm", ".html", ".jar", ".js", ".jse", ".mjs", ".msi", ".ps1", ".scr",
    ".sh", ".svg", ".vb", ".vbe", ".vbs", ".ws", ".wsf",
}

_DANGEROUS_DOWNLOAD_MIME_TYPES = {
    "application/javascript",
    "application/x-msdownload",
    "application/x-sh",
    "application/x-shellscript",
    "image/svg+xml",
    "text/html",
    "text/javascript",
    "text/x-shellscript",
}


def _audio_studio_artifact_roots(user_id: int | str) -> list[FileSystemPath]:
    return [
        DatabasePaths.get_user_base_directory(user_id) / "outputs",
        DatabasePaths.get_user_temp_outputs_dir(user_id),
    ]


def _resolve_contained_artifact_file(candidate: FileSystemPath, roots: list[FileSystemPath]) -> FileSystemPath:
    try:
        resolved_file = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="audio_studio_artifact_file_not_found") from exc
    except OSError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_audio_studio_artifact_path") from exc
    if not resolved_file.is_file():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_audio_studio_artifact_path")
    for root in roots:
        try:
            resolved_root = root.resolve(strict=True)
        except OSError:
            continue
        try:
            resolved_file.relative_to(resolved_root)
        except ValueError:
            continue
        return resolved_file
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_audio_studio_artifact_path")


def _resolve_audio_studio_artifact_path(
    *,
    collections_db: CollectionsDatabase,
    user_id: int | str,
    artifact: AudioStudioArtifactRow,
) -> FileSystemPath:
    raw_storage_path = str(artifact.storage_path or "").strip()
    if not raw_storage_path or "://" in raw_storage_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_audio_studio_artifact_path")
    roots = _audio_studio_artifact_roots(user_id)
    raw_path = FileSystemPath(raw_storage_path)
    if raw_path.is_absolute():
        return _resolve_contained_artifact_file(raw_path, roots)
    try:
        relative_filename = collections_db.resolve_output_storage_path(raw_storage_path)
    except (InvalidStoragePathError, InvalidStorageUserIdError, StorageUnavailableError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_audio_studio_artifact_path") from exc
    candidates = [
        _resolve_contained_artifact_file(root / relative_filename, roots)
        for root in roots
        if (root / relative_filename).exists()
    ]
    unique_candidates: list[FileSystemPath] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    if not unique_candidates:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="audio_studio_artifact_file_not_found")
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    if not _is_sha256_hex(artifact.content_hash):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="audio_studio_artifact_path_ambiguous")
    matches = [candidate for candidate in unique_candidates if _sha256_file(candidate) == artifact.content_hash.lower()]
    if len(matches) != 1:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="audio_studio_artifact_path_ambiguous")
    return matches[0]


def _normalize_download_mime(mime_type: str | None, path: FileSystemPath) -> str:
    normalized = str(mime_type or "").split(";", 1)[0].strip().lower()
    inferred = str(mimetypes.guess_type(path.name)[0] or "").split(";", 1)[0].strip().lower()
    if not normalized:
        normalized = inferred or "application/octet-stream"
    suffix = path.suffix.lower()
    if suffix in _DANGEROUS_DOWNLOAD_SUFFIXES or normalized in _DANGEROUS_DOWNLOAD_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="unsupported_audio_studio_artifact_download_type",
        )
    return normalized
```

Update the existing authenticated media endpoint to call `_resolve_audio_studio_artifact_path(..., user_id=current_user.id, ...)`. Keep `_normalize_audio_mime`, byte range parsing, and the current small authenticated media route behavior intact.

- [ ] **Step 5: Add ticket dependency and helpers**

In `audio_studio.py`, add imports:

```python
from datetime import datetime, timedelta
import re

from tldw_Server_API.app.core.AuthNZ.db_config import AuthDatabaseConfig
from tldw_Server_API.app.core.Audio_Studio.media_tickets import (
    AudioStudioMediaTicketRow,
    AudioStudioMediaTicketStore,
    hash_media_ticket_token,
    utc_now,
)
```

Add helper constants and functions:

```python
_MEDIA_TICKET_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,256}$")
_PLAYBACK_TICKET_TTL = timedelta(minutes=30)
_DOWNLOAD_TICKET_TTL = timedelta(minutes=10)


def get_audio_studio_media_ticket_store() -> AudioStudioMediaTicketStore:
    users_db = AuthDatabaseConfig().get_user_database(client_id="audio_studio_media_tickets")
    return AudioStudioMediaTicketStore(users_db.backend)


def _auth_mode_label(request: Request) -> str | None:
    if request.headers.get("authorization"):
        return "jwt"
    if request.headers.get("x-api-key"):
        return "single_user"
    return None


def _ticket_path(raw_token: str) -> str:
    return f"/api/v1/audio-studio/media-tickets/{raw_token}"


def _raise_invalid_media_ticket() -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="audio_studio_media_ticket_not_found")


def _ticket_is_expired(ticket: AudioStudioMediaTicketRow) -> bool:
    expiry = datetime.fromisoformat(ticket.expires_at.replace("Z", "+00:00"))
    return expiry <= utc_now()


def _validate_ticket_for_redemption(raw_token: str, store: AudioStudioMediaTicketStore) -> AudioStudioMediaTicketRow:
    if not _MEDIA_TICKET_TOKEN_PATTERN.fullmatch(raw_token):
        _raise_invalid_media_ticket()
    ticket = store.get_by_hash(hash_media_ticket_token(raw_token))
    if ticket is None:
        _raise_invalid_media_ticket()
    if ticket.revoked_at:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="audio_studio_media_ticket_revoked")
    if ticket.consumed_at:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="audio_studio_media_ticket_consumed")
    if _ticket_is_expired(ticket):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="audio_studio_media_ticket_expired")
    return ticket
```

- [ ] **Step 6: Add mint endpoint**

Add this endpoint before the existing `/projects/{project_id}/artifacts/{artifact_id}/media` route:

```python
@router.post(
    "/projects/{project_id}/artifacts/{artifact_id}/tickets",
    response_model=AudioStudioMediaTicketResponse,
)
async def create_audio_studio_artifact_media_ticket(
    request: Request,
    project_id: AudioStudioIdPath,
    artifact_id: AudioStudioIdPath,
    ticket_request: AudioStudioMediaTicketCreate,
    current_user: User = Depends(get_request_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
    ticket_store: AudioStudioMediaTicketStore = Depends(get_audio_studio_media_ticket_store),
) -> AudioStudioMediaTicketResponse:
    project = _load_project_or_404(collections_db, project_id)
    artifact = _load_audio_studio_artifact_or_404(collections_db, project, artifact_id)
    media_path = _resolve_audio_studio_artifact_path(
        collections_db=collections_db,
        user_id=current_user.id,
        artifact=artifact,
    )
    actual_size = media_path.stat().st_size
    if artifact.size_bytes is not None and artifact.size_bytes != actual_size:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="audio_studio_artifact_size_mismatch")

    purpose = ticket_request.purpose.value
    if purpose == "playback":
        _normalize_audio_mime(artifact.mime_type, media_path)
        ttl = _PLAYBACK_TICKET_TTL
    else:
        _normalize_download_mime(artifact.mime_type, media_path)
        ttl = _DOWNLOAD_TICKET_TTL

    ticket_store.cleanup(retention=timedelta(hours=1))
    raw_token, ticket = ticket_store.create_ticket(
        user_id=int(current_user.id),
        project_id=project.project_id,
        artifact_id=artifact.artifact_id,
        purpose=purpose,
        expires_at=utc_now() + ttl,
        created_by_auth_mode=_auth_mode_label(request),
    )
    return AudioStudioMediaTicketResponse(
        ticket_path=_ticket_path(raw_token),
        expires_at=ticket.expires_at,
        purpose=ticket_request.purpose,
        artifact_id=artifact.artifact_id,
    )
```

- [ ] **Step 7: Add redeem endpoint**

Add:

```python
@router.get("/media-tickets/{token}", response_model=None)
async def redeem_audio_studio_media_ticket(
    request: Request,
    token: str,
    ticket_store: AudioStudioMediaTicketStore = Depends(get_audio_studio_media_ticket_store),
) -> Response:
    ticket = _validate_ticket_for_redemption(token, ticket_store)
    if ticket.purpose == "download":
        consumed = ticket_store.consume_download_ticket(ticket.token_hash)
        if consumed is None:
            refreshed = ticket_store.get_by_hash(ticket.token_hash)
            if refreshed and refreshed.consumed_at:
                raise HTTPException(status_code=status.HTTP_410_GONE, detail="audio_studio_media_ticket_consumed")
            if refreshed and refreshed.revoked_at:
                raise HTTPException(status_code=status.HTTP_410_GONE, detail="audio_studio_media_ticket_revoked")
            if refreshed and _ticket_is_expired(refreshed):
                raise HTTPException(status_code=status.HTTP_410_GONE, detail="audio_studio_media_ticket_expired")
            _raise_invalid_media_ticket()
        ticket = consumed
    else:
        ticket_store.touch_redeemed(ticket.token_hash)

    collections_db = CollectionsDatabase.for_user(user_id=ticket.user_id)
    project = _load_project_or_404(collections_db, ticket.project_id)
    artifact = _load_audio_studio_artifact_or_404(collections_db, project, ticket.artifact_id)
    media_path = _resolve_audio_studio_artifact_path(
        collections_db=collections_db,
        user_id=ticket.user_id,
        artifact=artifact,
    )
    actual_size = media_path.stat().st_size
    if artifact.size_bytes is not None and artifact.size_bytes != actual_size:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="audio_studio_artifact_size_mismatch")

    if ticket.purpose == "playback":
        mime_type = _normalize_audio_mime(artifact.mime_type, media_path)
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, no-store",
            "Content-Disposition": _content_disposition(
                artifact=artifact,
                media_path=media_path,
                mime_type=mime_type,
                download=False,
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        }
        range_header = request.headers.get("range")
        if range_header is not None:
            start, end = _parse_single_byte_range(range_header, actual_size)
            content_length = end - start + 1
            return StreamingResponse(
                _iter_file_range(media_path, start=start, length=content_length),
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                headers={
                    **headers,
                    "Content-Range": f"bytes {start}-{end}/{actual_size}",
                    "Content-Length": str(content_length),
                    "Content-Type": mime_type,
                },
            )
        return FileResponse(str(media_path), media_type=mime_type, headers=headers)

    mime_type = _normalize_download_mime(artifact.mime_type, media_path)
    return FileResponse(
        str(media_path),
        media_type=mime_type,
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": _content_disposition(
                artifact=artifact,
                media_path=media_path,
                mime_type=mime_type,
                download=True,
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )
```

- [ ] **Step 8: Run backend ticket API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_media_tickets_api.py tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_artifact_media_api.py -v
```

Expected: both ticket API tests and existing artifact media tests pass.

- [ ] **Step 9: Commit Task 2**

Run:

```bash
git add tldw_Server_API/app/api/v1/schemas/audio_studio_schemas.py tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_media_tickets_api.py
git commit -m "feat(audio-studio): add artifact media tickets API"
```

## Task 3: Access Log Redaction

**Files:**
- Modify: `tldw_Server_API/app/core/Logging/access_log_middleware.py`
- Test: `tldw_Server_API/tests/Logging/test_access_log_redaction.py`

- [ ] **Step 1: Write failing redaction tests**

Create `tldw_Server_API/tests/Logging/test_access_log_redaction.py`:

```python
"""Tests for access log path redaction."""

from __future__ import annotations

from tldw_Server_API.app.core.Logging.access_log_middleware import redact_access_log_path


def test_redacts_audio_studio_media_ticket_token() -> None:
    path = "/api/v1/audio-studio/media-tickets/raw-secret-token-123"

    assert redact_access_log_path(path) == "/api/v1/audio-studio/media-tickets/[REDACTED]"


def test_redacts_ticket_token_without_changing_other_paths() -> None:
    assert redact_access_log_path("/api/v1/audio-studio/projects/p1") == "/api/v1/audio-studio/projects/p1"
    assert redact_access_log_path("/api/v1/audio-studio/media-tickets/token?download=1") == (
        "/api/v1/audio-studio/media-tickets/[REDACTED]"
    )
```

- [ ] **Step 2: Run redaction tests and verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Logging/test_access_log_redaction.py -v
```

Expected: fail with `ImportError` or `AttributeError` for `redact_access_log_path`.

- [ ] **Step 3: Implement path redaction**

Modify `tldw_Server_API/app/core/Logging/access_log_middleware.py`:

```python
import re

_AUDIO_STUDIO_MEDIA_TICKET_PATH = re.compile(
    r"(/api/v1/audio-studio/media-tickets/)[^/?#]+"
)


def redact_access_log_path(path: str) -> str:
    return _AUDIO_STUDIO_MEDIA_TICKET_PATH.sub(r"\1[REDACTED]", path)
```

Then change the `dispatch` method:

```python
path = redact_access_log_path(request.url.path)
```

The logger binding and formatted message must both use the redacted `path`.

- [ ] **Step 4: Run redaction tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Logging/test_access_log_redaction.py -v
```

Expected: all redaction tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add tldw_Server_API/app/core/Logging/access_log_middleware.py tldw_Server_API/tests/Logging/test_access_log_redaction.py
git commit -m "fix(logging): redact audio studio media ticket tokens"
```

## Task 4: Frontend Ticket Service Helper

**Files:**
- Modify: `apps/packages/ui/src/services/audio-studio.ts`
- Modify: `apps/packages/ui/src/services/__tests__/audio-studio.test.ts`

- [ ] **Step 1: Write failing service tests**

Modify `apps/packages/ui/src/services/__tests__/audio-studio.test.ts`:

```ts
const tldwClientMocks = vi.hoisted(() => ({
  getConfig: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) => tldwClientMocks.getConfig(...args)
  }
}))
```

Add `mintAudioStudioArtifactMediaTicket` to the imports and add tests:

```ts
  it("mints playback media tickets and resolves ticket paths against the configured server", async () => {
    tldwClientMocks.getConfig.mockResolvedValueOnce({
      serverUrl: "http://127.0.0.1:8000"
    })
    mocks.bgRequest.mockResolvedValueOnce({
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-playback",
      ticket_url: null,
      expires_at: "2026-06-24T12:00:00Z",
      purpose: "playback",
      artifact_id: "artifact-1"
    })

    const ticket = await mintAudioStudioArtifactMediaTicket("project 1", "artifact/1", "playback")

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/audio-studio/projects/project%201/artifacts/artifact%2F1/tickets",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { purpose: "playback" }
    })
    expect(ticket.ticket_url).toBe(
      "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-playback"
    )
    expect(ticket.ticket_path).toBe("/api/v1/audio-studio/media-tickets/ticket-playback")
  })

  it("prefers backend ticket_url when provided", async () => {
    tldwClientMocks.getConfig.mockResolvedValueOnce({
      serverUrl: "http://127.0.0.1:8000"
    })
    mocks.bgRequest.mockResolvedValueOnce({
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-download",
      ticket_url: "https://public.example.test/api/v1/audio-studio/media-tickets/ticket-download",
      expires_at: "2026-06-24T12:00:00Z",
      purpose: "download",
      artifact_id: "artifact-1"
    })

    const ticket = await mintAudioStudioArtifactMediaTicket("project-1", "artifact-1", "download")

    expect(ticket.ticket_url).toBe(
      "https://public.example.test/api/v1/audio-studio/media-tickets/ticket-download"
    )
  })
```

- [ ] **Step 2: Run service tests and verify they fail**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/services/__tests__/audio-studio.test.ts --maxWorkers=1
```

Expected: fail because `mintAudioStudioArtifactMediaTicket` is not exported.

- [ ] **Step 3: Implement service helper**

Modify `apps/packages/ui/src/services/audio-studio.ts`:

```ts
import { resolveBrowserRequestTransport } from "@/services/tldw/request-core"
import { tldwClient } from "@/services/tldw/TldwApiClient"
```

Add types:

```ts
export type AudioStudioMediaTicketPurpose = "playback" | "download"

export type AudioStudioMediaTicketResponse = {
  ticket_path: string
  ticket_url?: string | null
  expires_at: string
  purpose: AudioStudioMediaTicketPurpose
  artifact_id: string
}

export type AudioStudioResolvedMediaTicket = AudioStudioMediaTicketResponse & {
  ticket_url: string
}
```

Add helpers near `getAudioStudioArtifactMediaPath`:

```ts
const isAbsoluteHttpUrl = (value: string | null | undefined): value is string =>
  typeof value === "string" && /^https?:\/\//i.test(value)

const resolveAudioStudioTicketBrowserUrl = async (
  ticket: AudioStudioMediaTicketResponse
): Promise<string> => {
  if (isAbsoluteHttpUrl(ticket.ticket_url)) {
    return ticket.ticket_url
  }
  const config = await tldwClient.getConfig().catch(() => null)
  return resolveBrowserRequestTransport({
    config,
    path: ticket.ticket_path
  }).url
}

export const mintAudioStudioArtifactMediaTicket = async (
  projectId: string,
  artifactId: string,
  purpose: AudioStudioMediaTicketPurpose
): Promise<AudioStudioResolvedMediaTicket> => {
  const response = await bgRequest<AudioStudioMediaTicketResponse>({
    path: apiPath(`${projectPath(projectId)}/artifacts/${encodeURIComponent(artifactId)}/tickets`),
    method: "POST",
    headers: JSON_HEADERS,
    body: { purpose }
  })
  return {
    ...response,
    ticket_url: await resolveAudioStudioTicketBrowserUrl(response)
  }
}
```

- [ ] **Step 4: Run service tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/services/__tests__/audio-studio.test.ts --maxWorkers=1
```

Expected: `audio-studio` service tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add apps/packages/ui/src/services/audio-studio.ts apps/packages/ui/src/services/__tests__/audio-studio.test.ts
git commit -m "feat(audio-studio): add media ticket client helper"
```

## Task 5: Timeline Ticket Playback And Click-Only Downloads

**Files:**
- Modify: `apps/packages/ui/src/components/Option/AudioStudio/TimelineEditor.tsx`
- Modify: `apps/packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx`

- [ ] **Step 1: Write failing UI tests for ticket playback and downloads**

In `apps/packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx`, extend the service mock:

```ts
const audioStudioServiceMocks = vi.hoisted(() => ({
  fetchAudioStudioArtifactBlob: vi.fn(),
  mintAudioStudioArtifactMediaTicket: vi.fn()
}))
```

Update the `@/services/audio-studio` mock:

```ts
mintAudioStudioArtifactMediaTicket: (...args: unknown[]) =>
  audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket(...args),
```

In `beforeEach`, add:

```ts
audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket.mockResolvedValue({
  ticket_path: "/api/v1/audio-studio/media-tickets/ticket-playback",
  ticket_url: "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-playback",
  expires_at: "2026-06-24T12:00:00Z",
  purpose: "playback",
  artifact_id: "artifact-host"
})
```

Replace the old oversized-artifact unavailable test with:

```ts
  it("streams an oversized selected audio artifact with a playback ticket", async () => {
    const artifact = buildArtifact({
      size_bytes: 25 * 1024 * 1024 + 1
    })
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [artifact],
      isLoading: false,
      isError: false
    })
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0
        }
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000
        }
      ]
    })
    useAudioStudioStore.getState().setActiveWorkflow("podcast")

    render(<AudioStudioPage />)

    await waitFor(() =>
      expect(audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket).toHaveBeenCalledWith(
        "project-1",
        "artifact-host",
        "playback"
      )
    )
    expect(audioStudioServiceMocks.fetchAudioStudioArtifactBlob).not.toHaveBeenCalled()
    expect(await screen.findByLabelText("Selected clip audio preview")).toHaveAttribute(
      "src",
      "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-playback"
    )
    expect(screen.getByLabelText("Selected clip audio preview")).toHaveAttribute("referrerpolicy", "no-referrer")
  })
```

Add unknown-size audio playback coverage:

```ts
  it("streams an unknown-size selected audio artifact with a playback ticket", async () => {
    const artifact = buildArtifact({
      size_bytes: null
    })
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [artifact],
      isLoading: false,
      isError: false
    })
    setActiveProject({
      workflow: "narration",
      tracks: [{ track_id: "speech-track-1", name: "Narration", kind: "speech", order: 0 }],
      clips: [{
        clip_id: "clip-host",
        track_id: "speech-track-1",
        section_id: "section-1",
        artifact_id: "artifact-host",
        title: "Narration clip",
        clip_type: "speech",
        start_ms: 1000,
        duration_ms: 30000
      }]
    })
    useAudioStudioStore.getState().setActiveWorkflow("narration")

    render(<AudioStudioPage />)

    await waitFor(() =>
      expect(audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket).toHaveBeenCalledWith(
        "project-1",
        "artifact-host",
        "playback"
      )
    )
    expect(audioStudioServiceMocks.fetchAudioStudioArtifactBlob).not.toHaveBeenCalled()
    expect(await screen.findByLabelText("Selected clip audio preview")).toHaveAttribute(
      "src",
      "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-playback"
    )
  })
```

Add click-only download coverage:

```ts
  it("mints a download ticket only when the user clicks a large audio download", async () => {
    const artifact = buildArtifact({ size_bytes: 25 * 1024 * 1024 + 1 })
    audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket
      .mockResolvedValueOnce({
        ticket_path: "/api/v1/audio-studio/media-tickets/ticket-playback",
        ticket_url: "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-playback",
        expires_at: "2026-06-24T12:00:00Z",
        purpose: "playback",
        artifact_id: "artifact-host"
      })
      .mockResolvedValueOnce({
        ticket_path: "/api/v1/audio-studio/media-tickets/ticket-download",
        ticket_url: "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-download",
        expires_at: "2026-06-24T12:00:00Z",
        purpose: "download",
        artifact_id: "artifact-host"
      })
    projectHookMocks.useArtifacts.mockReturnValue({ data: [artifact], isLoading: false, isError: false })
    setActiveProject({
      workflow: "podcast",
      tracks: [{ track_id: "speech-track-1", name: "Dialogue", kind: "speech", order: 0 }],
      clips: [{
        clip_id: "clip-host",
        track_id: "speech-track-1",
        section_id: "section-1",
        artifact_id: "artifact-host",
        title: "Host intro",
        clip_type: "speech",
        start_ms: 1000,
        duration_ms: 30000
      }]
    })
    useAudioStudioStore.getState().setActiveWorkflow("podcast")
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})

    render(<AudioStudioPage />)
    const button = await screen.findByRole("button", { name: "Download selected clip audio" })
    expect(document.body.innerHTML).not.toContain("ticket-download")

    fireEvent.click(button)

    await waitFor(() =>
      expect(audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket).toHaveBeenLastCalledWith(
        "project-1",
        "artifact-host",
        "download"
      )
    )
    expect(anchorClick).toHaveBeenCalled()
    expect(document.body.innerHTML).not.toContain("ticket-download")
    anchorClick.mockRestore()
  })
```

Add non-audio download-only coverage:

```ts
  it("offers click-only download for non-audio artifacts without rendering an audio element", async () => {
    const artifact = buildArtifact({
      artifact_type: "analysis",
      mime_type: "application/json",
      size_bytes: null,
      metadata: { filename: "host-analysis.json" }
    })
    audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket.mockResolvedValueOnce({
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-json",
      ticket_url: "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-json",
      expires_at: "2026-06-24T12:00:00Z",
      purpose: "download",
      artifact_id: "artifact-host"
    })
    projectHookMocks.useArtifacts.mockReturnValue({ data: [artifact], isLoading: false, isError: false })
    setActiveProject({
      workflow: "podcast",
      tracks: [{ track_id: "speech-track-1", name: "Dialogue", kind: "speech", order: 0 }],
      clips: [{
        clip_id: "clip-host",
        track_id: "speech-track-1",
        artifact_id: "artifact-host",
        title: "Host intro",
        clip_type: "speech",
        start_ms: 1000,
        duration_ms: 30000
      }]
    })
    useAudioStudioStore.getState().setActiveWorkflow("podcast")
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})

    render(<AudioStudioPage />)
    expect(await screen.findByText("Selected clip artifact is download-only.")).toBeInTheDocument()
    expect(screen.queryByLabelText("Selected clip audio preview")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Download selected artifact" }))

    await waitFor(() =>
      expect(audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket).toHaveBeenCalledWith(
        "project-1",
        "artifact-host",
        "download"
      )
    )
    expect(anchorClick).toHaveBeenCalled()
    anchorClick.mockRestore()
  })
```

- [ ] **Step 2: Run UI tests and verify they fail**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx --maxWorkers=1
```

Expected: fail because TimelineEditor does not import or call `mintAudioStudioArtifactMediaTicket`.

- [ ] **Step 3: Implement ticket state in TimelineEditor**

Modify imports in `TimelineEditor.tsx`:

```ts
import { Download, Pause, Play, Save } from "lucide-react";
import {
  fetchAudioStudioArtifactBlob,
  mintAudioStudioArtifactMediaTicket,
  type AudioStudioArtifact,
} from "@/services/audio-studio";
```

Add state:

```ts
const [downloadError, setDownloadError] = useState<string | null>(null);
const [isDownloadLoading, setIsDownloadLoading] = useState(false);
const audioElementRef = useRef<HTMLAudioElement | null>(null);
const ticketRetryKeyRef = useRef<string | null>(null);
```

Add derived booleans:

```ts
const selectedArtifactCanUseBlob =
  selectedArtifactIsAudio &&
  selectedArtifactSizeKnown &&
  selectedArtifact.size_bytes <= MAX_BLOB_PREVIEW_BYTES;
const selectedArtifactShouldUseTicketPlayback =
  selectedArtifactIsAudio && !selectedArtifactCanUseBlob;
const selectedArtifactCanDownload = Boolean(selectedArtifact);
```

- [ ] **Step 4: Split preview loading between Blob and ticket playback**

Replace the current Blob-only `useEffect` condition with:

```ts
if (
  !activeProject ||
  !selectedClip ||
  !selectedArtifactId ||
  !selectedArtifact ||
  !selectedArtifactIsAudio ||
  !selectedPreviewKey
) {
  previewStateKeyRef.current = null;
  return () => {
    cancelled = true;
  };
}

previewStateKeyRef.current = selectedPreviewKey;
setIsPreviewLoading(true);

const loadPreview = selectedArtifactCanUseBlob
  ? fetchAudioStudioArtifactBlob(activeProject.project_id, selectedArtifact).then((blob) => {
      objectUrl = URL.createObjectURL(blob);
      return objectUrl;
    })
  : mintAudioStudioArtifactMediaTicket(
      activeProject.project_id,
      selectedArtifact.artifact_id,
      "playback",
    ).then((ticket) => ticket.ticket_url);

void loadPreview
  .then((url) => {
    if (cancelled) return;
    previewStateKeyRef.current = selectedPreviewKey;
    ticketRetryKeyRef.current = null;
    setPreviewUrl(url);
  })
  .catch(() => {
    if (!cancelled) {
      previewStateKeyRef.current = selectedPreviewKey;
      setPreviewError("Preview unavailable");
    }
  })
  .finally(() => {
    if (!cancelled) {
      setIsPreviewLoading(false);
    }
  });
```

The cleanup must revoke only `blob:` URLs:

```ts
if (objectUrl?.startsWith("blob:")) {
  URL.revokeObjectURL(objectUrl);
}
```

- [ ] **Step 5: Add one-time playback remint on media error**

Add:

```ts
const handleTicketPlaybackError = () => {
  if (
    !activeProject ||
    !selectedArtifact ||
    !selectedArtifactShouldUseTicketPlayback ||
    !selectedPreviewKey ||
    ticketRetryKeyRef.current === selectedPreviewKey
  ) {
    return;
  }
  ticketRetryKeyRef.current = selectedPreviewKey;
  const currentTime = audioElementRef.current?.currentTime ?? 0;
  setIsPreviewLoading(true);
  void mintAudioStudioArtifactMediaTicket(
    activeProject.project_id,
    selectedArtifact.artifact_id,
    "playback",
  )
    .then((ticket) => {
      if (previewStateKeyRef.current !== selectedPreviewKey) return;
      setPreviewUrl(ticket.ticket_url);
      window.setTimeout(() => {
        if (audioElementRef.current && Number.isFinite(currentTime)) {
          audioElementRef.current.currentTime = currentTime;
        }
      }, 0);
    })
    .catch(() => {
      if (previewStateKeyRef.current === selectedPreviewKey) {
        setPreviewError("Preview unavailable");
      }
    })
    .finally(() => {
      if (previewStateKeyRef.current === selectedPreviewKey) {
        setIsPreviewLoading(false);
      }
    });
};
```

Update `<audio>`:

```tsx
<audio
  ref={audioElementRef}
  aria-label="Selected clip audio preview"
  className="mt-3 w-full"
  controls
  referrerPolicy="no-referrer"
  src={visiblePreviewUrl}
  onError={handleTicketPlaybackError}
/>
```

- [ ] **Step 6: Add click-only download handler**

Add:

```ts
const triggerBrowserDownload = (url: string, filename?: string | null) => {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.rel = "noreferrer";
  anchor.referrerPolicy = "no-referrer";
  if (filename) {
    anchor.download = filename;
  }
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
};

const handleDownloadArtifact = async () => {
  if (!activeProject || !selectedArtifact) return;
  setDownloadError(null);
  setIsDownloadLoading(true);
  try {
    const ticket = await mintAudioStudioArtifactMediaTicket(
      activeProject.project_id,
      selectedArtifact.artifact_id,
      "download",
    );
    triggerBrowserDownload(ticket.ticket_url, downloadFilename);
  } catch {
    setDownloadError("Download unavailable");
  } finally {
    setIsDownloadLoading(false);
  }
};
```

- [ ] **Step 7: Update preview controls**

Keep the existing small Blob anchor only for `visiblePreviewUrl && selectedArtifactCanUseBlob`. For ticket-backed or non-audio artifacts, render a button instead:

```tsx
{selectedArtifactCanDownload && !selectedArtifactCanUseBlob ? (
  <Button
    size="small"
    icon={<Download className="h-4 w-4" />}
    onClick={handleDownloadArtifact}
    loading={isDownloadLoading}
  >
    {selectedArtifactIsAudio ? "Download selected clip audio" : "Download selected artifact"}
  </Button>
) : null}
```

Change non-audio copy to:

```tsx
<Text type="secondary" className="text-xs">
  Selected clip artifact is download-only.
</Text>
```

Remove the old hard block states `Artifact size is unavailable for browser preview.` and `Artifact is too large for browser preview.` for audio artifacts. Unknown-size and oversized audio now use ticket playback.

Render download errors:

```tsx
{downloadError ? (
  <Text type="danger" className="text-xs">
    {downloadError}
  </Text>
) : null}
```

- [ ] **Step 8: Run UI tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx --maxWorkers=1
```

Expected: Audio Studio page tests pass.

- [ ] **Step 9: Commit Task 5**

Run:

```bash
git add apps/packages/ui/src/components/Option/AudioStudio/TimelineEditor.tsx apps/packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx
git commit -m "feat(audio-studio): stream large artifacts with media tickets"
```

## Task 6: Documentation, Security Verification, And Backlog Finalization

**Files:**
- Modify: `Docs/Audio_Studio.md`
- Modify: `backlog/tasks/task-2358 - Add-Audio-Studio-large-artifact-WebUI-transport.md`

- [ ] **Step 1: Update Audio Studio documentation**

In `Docs/Audio_Studio.md`, add a section after the artifact playback section:

```markdown
## Large Artifact Media Tickets

Audio Studio supports short-lived media tickets for native browser playback and downloads when a Blob fetch would be too large or when the artifact is not audio-previewable.

- Mint endpoint: `POST /api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/tickets`
- Redeem endpoint: `GET /api/v1/audio-studio/media-tickets/{token}`
- Playback tickets are audio-only, reusable for 30 minutes, support `Range`, and use `Content-Disposition: inline`.
- Download tickets are single-use, expire after 10 minutes, ignore browser `Range` headers, and force `Content-Disposition: attachment`.
- The server stores only the SHA-256 hash of the ticket token.
- Redemption repeats ownership, artifact existence, safe-root containment, symlink, file size, MIME, and extension checks.
- Responses use `Cache-Control: private, no-store`, `Referrer-Policy: no-referrer`, and `X-Content-Type-Options: nosniff`.
- `Cross-Origin-Resource-Policy: same-origin` is intentionally not set for ticket media responses until WebUI and extension/shared UI playback compatibility is verified.

Application access logs redact media ticket tokens. Operators running a reverse proxy should also redact or suppress `/api/v1/audio-studio/media-tickets/{token}` in proxy access logs because the token is a short-lived bearer credential.
```

- [ ] **Step 2: Run focused backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_media_ticket_store.py tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_schemas.py tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_media_tickets_api.py tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_artifact_media_api.py tldw_Server_API/tests/Logging/test_access_log_redaction.py -v
```

Expected: all selected backend tests pass.

- [ ] **Step 3: Run focused frontend tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/services/__tests__/audio-studio.test.ts src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx --maxWorkers=1
```

Expected: selected frontend tests pass.

- [ ] **Step 4: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Audio_Studio/media_tickets.py tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py tldw_Server_API/app/core/Logging/access_log_middleware.py -f json -o /tmp/bandit_audio_studio_media_tickets.json
```

Expected: command exits `0` with no new high or medium severity findings in touched code. If Bandit reports findings, fix the changed code and rerun the command.

- [ ] **Step 5: Run diff whitespace check**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 6: Update Backlog task**

Update `backlog/tasks/task-2358 - Add-Audio-Studio-large-artifact-WebUI-transport.md`:

```markdown
## Implementation Notes

- Implemented DB-backed scoped media tickets in the AuthNZ/global database.
- Added playback tickets for native audio `Range` streaming and single-use download tickets for large/non-audio artifacts.
- Kept small Blob preview/download behavior unchanged.
- Added access-log redaction for media ticket token paths.
- Documented proxy log-redaction responsibility in `Docs/Audio_Studio.md`.

## Final Summary

Audio Studio large-artifact transport now uses short-lived scoped media tickets for native playback and downloads while preserving strict artifact ownership and safe-root validation.

## Definition of Done
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
```

Add the actual verification command outcomes under Implementation Notes with pass/fail details and any skipped tests.

- [ ] **Step 7: Commit documentation and task update**

Run:

```bash
git add Docs/Audio_Studio.md "backlog/tasks/task-2358 - Add-Audio-Studio-large-artifact-WebUI-transport.md"
git commit -m "docs(audio-studio): document media ticket transport"
```

## Final Verification Commands

Run all commands before reporting implementation complete:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_media_ticket_store.py tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_schemas.py tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_media_tickets_api.py tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_artifact_media_api.py tldw_Server_API/tests/Logging/test_access_log_redaction.py -v
```

```bash
cd apps/packages/ui && bunx vitest run src/services/__tests__/audio-studio.test.ts src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx --maxWorkers=1
```

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Audio_Studio/media_tickets.py tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py tldw_Server_API/app/core/Logging/access_log_middleware.py -f json -o /tmp/bandit_audio_studio_media_tickets.json
```

```bash
git diff --check
```

## Review Checklist

- [ ] Mint endpoint requires normal auth and validates ownership before issuing a ticket.
- [ ] Redeem endpoint is unauthenticated except for the bearer ticket and revalidates project, artifact, path, file, MIME, and size/hash state.
- [ ] Playback tickets are reusable, audio-only, 30-minute TTL, inline, and `Range` capable.
- [ ] Download tickets are single-use, 10-minute TTL, attachment-only, and ignore `Range`.
- [ ] Raw ticket tokens are never stored in the database.
- [ ] Ticket responses do not include filesystem paths, token hashes, API keys, JWTs, or provider secrets.
- [ ] App access logs redact ticket tokens.
- [ ] Reverse-proxy redaction is documented.
- [ ] `Cross-Origin-Resource-Policy: same-origin` is not set on ticket redemption responses.
- [ ] Small Blob preview/download behavior still passes existing tests.
- [ ] Oversized and unknown-size audio use ticket playback.
- [ ] Non-audio artifacts are download-only and direct user clicks mint download tickets.
