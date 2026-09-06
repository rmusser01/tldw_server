"""Original-output history disposal cannot clear reused IDs or revive late links."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase
from tldw_Server_API.tests.Collections.test_reading_revision_mutations import db as db

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]

INCARNATION = "a" * 32
TOKEN = "b" * 32
DELETED_AT = "2026-09-05T12:00:00+00:00"


@pytest.fixture
def media(db, tmp_path):
    backend = db.backend if db.backend.backend_type == BackendType.POSTGRESQL else None
    instance = MediaDatabase(tmp_path / "media.db", client_id=db.user_id, backend=backend)
    yield instance
    instance.close_connection()


def create(media, *, user="780", incarnation=INCARNATION, conn=None, **kwargs):
    return media.create_tts_history_entry(
        user_id=user,
        text_hash="hash",
        output_id=7,
        artifact_ids=[7, "audio.mp3"],
        output_incarnation=incarnation,
        conn=conn,
        **kwargs,
    )


def dispose(media, **kwargs):
    return media.dispose_tts_output_instance(
        user_id="780",
        output_incarnation=INCARNATION,
        disposal_token=TOKEN,
        deleted_at=DELETED_AT,
        **kwargs,
    )


def history(media, history_id, user="780"):
    return media.get_tts_history_entry(user_id=user, history_id=history_id, include_deleted=True)


@pytest.mark.parametrize("insert_first", [False, True])
def test_disposal_clears_existing_and_late_links_without_deleting_history(media, insert_first):
    first = create(media) if insert_first else None
    dispose(media)
    late = create(media)
    for history_id in [first, late] if first else [late]:
        row = history(media, history_id)
        assert row["output_id"] is None and row["artifact_ids"] is None
        assert row["artifact_deleted_at"] is not None and not row["deleted"]


def test_replay_preserves_first_tombstone_and_isolates_users_incarnations_and_legacy(media):
    old = create(media)
    newer = create(media, incarnation="c" * 32)
    foreign = create(media, user="781")
    legacy = media.create_tts_history_entry(user_id="780", text_hash="legacy", output_id=7, artifact_ids=[7])
    dispose(media)
    media.dispose_tts_output_instance(
        user_id="780",
        output_incarnation=INCARNATION,
        disposal_token="d" * 32,
        deleted_at="2026-09-06T12:00:00+00:00",
    )
    assert history(media, old)["output_id"] is None
    for history_id, user in [(newer, "780"), (foreign, "781"), (legacy, "780")]:
        assert history(media, history_id, user)["output_id"] == 7
    state = media.execute_query(
        "SELECT * FROM tts_output_instances WHERE user_id = ? AND output_incarnation = ?", ("780", INCARNATION)
    ).fetchone()
    assert state["state"] == "disposed" and state["disposal_token"] == TOKEN
    assert state["disposed_at"] == DELETED_AT


@pytest.mark.parametrize("operation", ["insert", "dispose"])
def test_caller_transaction_rollback_preserves_receiver_and_history(media, operation):
    existing = create(media) if operation == "dispose" else None
    with pytest.raises(ValueError, match="rollback"):
        with media.transaction() as conn:
            if operation == "insert":
                create(media, conn=conn)
            else:
                dispose(media, conn=conn)
            raise ValueError("rollback")
    states = media.execute_query("SELECT state FROM tts_output_instances").fetchall()
    if operation == "insert":
        assert not states and media.count_tts_history(user_id="780") == 0
    else:
        assert states[0]["state"] == "live" and history(media, existing)["output_id"] == 7


@pytest.mark.parametrize("explicit_connection", [False, True])
def test_legacy_history_insert_cannot_commit_mixed_caller_transaction(media, explicit_connection):
    with pytest.raises(ValueError, match="rollback"):
        with media.transaction() as conn:
            create(media, conn=conn)
            create(media, incarnation=None, conn=conn if explicit_connection else None)
            raise ValueError("rollback")
    assert not media.execute_query("SELECT * FROM tts_output_instances").fetchall()
    assert media.count_tts_history(user_id="780") == 0


@pytest.mark.parametrize("dispose_first", [False, True])
def test_insert_and_disposal_serialize_both_commit_orders(media, dispose_first):
    create(media)  # Existing live receiver row, not just insert-on-conflict behavior.
    entered, release, competing = Event(), Event(), Event()

    def first():
        with media.transaction() as conn:
            result = dispose(media, conn=conn) if dispose_first else create(media, conn=conn)
            entered.set()
            assert release.wait(10)
            return result

    def second():
        assert entered.wait(10)
        competing.set()
        return create(media) if dispose_first else dispose(media)

    with ThreadPoolExecutor(max_workers=2) as pool:
        leading, trailing = pool.submit(first), pool.submit(second)
        try:
            assert entered.wait(10) and competing.wait(10)
            assert not trailing.done()
        finally:
            release.set()
        leading.result(timeout=10)
        trailing.result(timeout=10)
    rows = media.list_tts_history(user_id="780")
    assert len(rows) == 2
    assert all(history(media, row["id"])["artifact_ids"] is None and row["output_id"] is None for row in rows)


def test_v26_upgrade_and_reinitialization_preserve_history_and_receiver_state(media):
    assert media._CURRENT_SCHEMA_VERSION == 27
    legacy = media.create_tts_history_entry(user_id="780", text_hash="legacy", output_id=7)
    media.execute_query("DROP INDEX idx_tts_history_incarnation")
    media.execute_query("DROP TABLE tts_output_instances")
    media.execute_query("ALTER TABLE tts_history DROP COLUMN output_incarnation")
    media.execute_query("UPDATE schema_version SET version = 26", commit=True)
    for _ in range(2):
        backend = media.backend if media.backend_type == BackendType.POSTGRESQL else None
        reopened = MediaDatabase(media.db_path_str, client_id="780", backend=backend)
        try:
            assert history(reopened, legacy)["output_id"] == 7
            dispose(reopened)
            late = create(reopened)
            assert history(reopened, late)["output_id"] is None
            assert reopened.execute_query("SELECT version FROM schema_version").fetchone()["version"] == 27
        finally:
            reopened.close_connection()


@pytest.mark.parametrize("history_present", [False, True])
def test_upgrade_handles_optional_history_and_already_installed_receiver(media, history_present):
    if history_present:
        dispose(media)
    else:
        media.execute_query("DROP TABLE tts_output_instances")
        media.execute_query("DROP TABLE tts_history")
    media.execute_query("UPDATE schema_version SET version = 26", commit=True)
    media._initialize_schema()
    if not history_present:
        dispose(media)
    late = create(media)
    assert history(media, late)["output_id"] is None
    assert media.execute_query("SELECT version FROM schema_version").fetchone()["version"] == 27


def test_invalid_receiver_identity_or_timestamp_does_not_write(media):
    valid = {"user_id": "780", "output_incarnation": INCARNATION, "disposal_token": TOKEN, "deleted_at": DELETED_AT}
    for invalid in [
        {"user_id": None},
        {"user_id": ""},
        {"user_id": " "},
        {"output_incarnation": "not-an-incarnation"},
        {"disposal_token": "C" * 32},
        {"deleted_at": "2026-09-05T12:00:00"},
        {"deleted_at": "invalid"},
    ]:
        with pytest.raises(ValueError, match="^output_history_invalid$"):
            media.dispose_tts_output_instance(**(valid | invalid))
    with pytest.raises(ValueError, match="^output_history_invalid$"):
        create(media, incarnation="")
    assert not media.execute_query("SELECT * FROM tts_output_instances").fetchall()
    assert media.count_tts_history(user_id="780") == 0


@pytest.mark.parametrize("operation", ["insert", "dispose"])
def test_owned_transaction_rolls_back_receiver_when_history_write_fails(media, monkeypatch, operation):
    existing = create(media) if operation == "dispose" else None
    execute = media.execute_query

    def fail_history_write(query, *args, **kwargs):
        if query.startswith(("INSERT INTO tts_history ", "UPDATE tts_history SET artifact_deleted_at")):
            raise ValueError("injected history failure")
        return execute(query, *args, **kwargs)

    monkeypatch.setattr(media, "execute_query", fail_history_write)
    with pytest.raises(ValueError, match="injected history failure"):
        create(media) if operation == "insert" else dispose(media)
    rows = execute("SELECT state FROM tts_output_instances").fetchall()
    if operation == "insert":
        assert not rows and media.count_tts_history(user_id="780") == 0
    else:
        assert rows[0]["state"] == "live" and history(media, existing)["output_id"] == 7


def test_disposal_clears_soft_deleted_history_and_keeps_flags(media):
    old = create(media, deleted=True, favorite=True)
    dispose(media)
    late = create(media, deleted=True, favorite=True)
    for history_id in (old, late):
        row = history(media, history_id)
        assert row["deleted"] and row["favorite"]
        assert row["output_id"] is None and row["artifact_ids"] is None


def test_receiver_schema_rejects_missing_identity_and_impossible_states(media):
    for values in [
        (None, INCARNATION, "live", None, None),
        ("", INCARNATION, "live", None, None),
        ("780", None, "live", None, None),
        ("780", "short", "live", None, None),
        ("780", INCARNATION, None, None, None),
        ("780", INCARNATION, "invalid", None, None),
        ("780", INCARNATION, "live", TOKEN, DELETED_AT),
        ("780", INCARNATION, "disposed", None, DELETED_AT),
        ("780", INCARNATION, "disposed", TOKEN, None),
    ]:
        with pytest.raises(DatabaseError):
            with media.transaction() as conn:
                media.execute_query(
                    "INSERT INTO tts_output_instances (user_id, output_incarnation, state, disposal_token, disposed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    values,
                    connection=conn,
                    log_errors=False,
                )
    assert not media.execute_query("SELECT * FROM tts_output_instances").fetchall()
