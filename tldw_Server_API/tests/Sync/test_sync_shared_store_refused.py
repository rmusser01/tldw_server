"""Sync v2 must refuse a store that every account would share.

Sync isolates accounts by storage location: each user gets their own SQLite
file. The schema cannot isolate them any other way. Its core tables carry no
owner column at all -- sync_envelopes, which holds the synced payloads,
sync_object_state, sync_current_heads, sync_device_cursors, sync_conflicts,
sync_attachments, sync_blob_chunks, sync_domain_state -- and no query in the
14.9k-line repository filters on one. Ownership exists only on the periphery:
datasets, devices, blobs.

SYNC_V2_DATABASE_URL and SYNC_V2_SQLITE_PATH override that location with one
fixed target and drop the user id, so in multi-user mode every account's notes,
chat messages and attachments land in one store with no way to separate them
afterwards. Adding predicates cannot fix that; the columns do not exist.

Single-user mode is unaffected, since there is only one account.
"""

import pytest

from tldw_Server_API.app.core.AuthNZ import settings as authnz_settings
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase, SyncStoreError

pytestmark = pytest.mark.unit


@pytest.fixture()
def multi_user(monkeypatch):
    monkeypatch.setattr(authnz_settings, "is_single_user_mode", lambda: False)


@pytest.fixture()
def single_user(monkeypatch):
    monkeypatch.setattr(authnz_settings, "is_single_user_mode", lambda: True)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("SYNC_V2_DATABASE_URL", raising=False)
    monkeypatch.delenv("SYNC_V2_SQLITE_PATH", raising=False)


def test_shared_postgres_url_is_refused_in_multi_user_mode(monkeypatch, multi_user):
    """The regression: every account's synced content into one database."""
    monkeypatch.setenv("SYNC_V2_DATABASE_URL", "postgresql://db.example/sync_v2")

    with pytest.raises(SyncStoreError, match="shared store"):
        SyncDatabase(user_id=7)


def test_shared_sqlite_path_is_refused_in_multi_user_mode(monkeypatch, multi_user, tmp_path):
    """An env-set file path drops the user id just as thoroughly as a DSN."""
    monkeypatch.setenv("SYNC_V2_SQLITE_PATH", str(tmp_path / "everyone.db"))

    with pytest.raises(SyncStoreError, match="shared store"):
        SyncDatabase(user_id=7)


def test_the_refusal_names_the_variables_to_unset(monkeypatch, multi_user):
    monkeypatch.setenv("SYNC_V2_DATABASE_URL", "postgresql://db.example/sync_v2")

    with pytest.raises(SyncStoreError, match="SYNC_V2_SQLITE_PATH"):
        SyncDatabase(user_id=7)


def test_single_user_mode_may_share_a_store(monkeypatch, single_user, tmp_path):
    """One account cannot collide with another; the override stays usable."""
    monkeypatch.setenv("SYNC_V2_SQLITE_PATH", str(tmp_path / "only-user.db"))

    db = SyncDatabase(user_id=1)

    assert db is not None


def test_the_per_user_default_is_untouched(multi_user, tmp_path, monkeypatch):
    """With no override, each account still gets its own file."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))

    db = SyncDatabase(user_id=7)

    assert db is not None
