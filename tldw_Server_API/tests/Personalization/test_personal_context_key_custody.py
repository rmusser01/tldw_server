from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization.personal_context_key_provider import (
    ServerProfileKeyProvider,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ProfileStorageLockedError,
)
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    encoded_master_key,
)


def test_key_provider_creates_and_reopens_independent_profile_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"a"))
    db = PersonalizationDB.for_path(tmp_path / "Personalization.db")
    provider = ServerProfileKeyProvider(db)

    created = provider.create("profile-a")
    reopened = ServerProfileKeyProvider(PersonalizationDB.for_path(tmp_path / "Personalization.db")).load("profile-a")

    assert created == reopened
    assert len(created.encryption_key) == len(created.integrity_key) == 32
    assert created.encryption_key != created.integrity_key
    assert created.key_version == created.integrity_key_version == 1


def test_missing_master_key_locks_existing_profile(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"a"))
    db = PersonalizationDB.for_path(tmp_path / "Personalization.db")
    ServerProfileKeyProvider(db).create("profile-a")

    monkeypatch.delenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY")
    with pytest.raises(ProfileStorageLockedError, match="master key"):
        ServerProfileKeyProvider(db).load("profile-a")


def test_changed_master_key_never_replaces_profile_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"a"))
    db = PersonalizationDB.for_path(tmp_path / "Personalization.db")
    original = ServerProfileKeyProvider(db).create("profile-a")

    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"b"))
    with pytest.raises(ProfileStorageLockedError, match="unavailable"):
        ServerProfileKeyProvider(db).load("profile-a")

    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"a"))
    assert ServerProfileKeyProvider(db).load("profile-a") == original


@pytest.mark.parametrize(
    "configured",
    ["", "not-base64", encoded_master_key(b"x")[:-4], "eA=="],
)
def test_invalid_master_key_fails_closed(tmp_path, monkeypatch, configured) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", configured)
    db = PersonalizationDB.for_path(tmp_path / "Personalization.db")

    with pytest.raises(ProfileStorageLockedError, match="master key"):
        ServerProfileKeyProvider(db).create("profile-a")


def test_profile_keys_are_never_stored_unwrapped(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"a"))
    db_path = tmp_path / "Personalization.db"
    db = PersonalizationDB.for_path(db_path)
    material = ServerProfileKeyProvider(db).create("profile-a")

    durable = db_path.read_bytes()
    assert material.encryption_key not in durable
    assert material.integrity_key not in durable
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT length(wrapped_profile_key), length(wrapped_integrity_key), "
            "length(wrap_nonce), length(integrity_wrap_nonce) "
            "FROM personal_context_profile_keys WHERE profile_id = ?",
            ("profile-a",),
        ).fetchone()
    assert row is not None
    assert row[0] > 32 and row[1] > 32 and row[2:] == (12, 12)
