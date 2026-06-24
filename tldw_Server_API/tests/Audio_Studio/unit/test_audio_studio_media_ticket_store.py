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
