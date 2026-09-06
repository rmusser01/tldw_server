"""Original-instance delivery survives outages without reclaiming completed files."""

import asyncio
import json
import time
from contextlib import contextmanager
from threading import Barrier, Event
from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase
from tldw_Server_API.tests.Collections.test_output_file_operations_storage import db as db
from tldw_Server_API.tests.Collections.test_output_file_operations_storage import run, service
from tldw_Server_API.tests.Collections.test_output_file_operations_storage import storage as storage

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


@pytest.fixture
def media(db, tmp_path):
    backend = db.backend if db.backend_type == BackendType.POSTGRESQL else None
    instance = MediaDatabase(tmp_path / "media.db", client_id=db.user_id, backend=backend)
    yield instance
    instance.close_connection()


def removal(db, storage):
    writer = service(db, storage)
    op = run(writer.prepare, kind="remove", output_id=storage[2].id, max_output_bytes=0)
    run(writer.publish_and_commit, op["token"])
    return writer, db.get_output_file_operation(op["token"], storage[1])


def history(media, incarnation, output_id):
    return media.create_tts_history_entry(
        user_id="780",
        text_hash="test",
        output_id=output_id,
        output_incarnation=incarnation,
        artifact_ids=[f"output:{output_id}"],
    )


def get_history(media, history_id):
    return media.get_tts_history_entry(user_id="780", history_id=history_id)


def test_invalid_creation_identity_rolls_back_output(db, monkeypatch):
    execute = db.backend.execute

    def invalid_identity(query, *args, **kwargs):
        if query.startswith("SELECT file_incarnation FROM outputs WHERE id = ? AND user_id = ?"):
            return SimpleNamespace(scalar=None)
        return execute(query, *args, **kwargs)

    monkeypatch.setattr(db.backend, "execute", invalid_identity)
    with pytest.raises(RuntimeError, match="output_history_identity_unavailable"):
        db.create_output_artifact_with_history_identity(
            type_="tts_audio",
            title="Invalid",
            format_="mp3",
            storage_path="invalid.mp3",
        )
    assert db.backend.execute("SELECT COUNT(*) FROM outputs").scalar == 0


def test_creation_fence_is_held_until_identity_capture(db, monkeypatch):
    if db.backend_type != BackendType.POSTGRESQL:
        pytest.skip("PostgreSQL independent-connection lock regression")
    execute = db.backend.execute

    def inspect_fence(query, *args, **kwargs):
        if query.startswith("SELECT file_incarnation FROM outputs WHERE id = ? AND user_id = ?"):
            # A competing writer cannot acquire the deletion/reuse fence in the
            # interval between the insert and its original-instance read.
            with pytest.raises(DatabaseError):
                with db.backend.transaction() as competitor:
                    execute(
                        "SELECT value FROM public.reading_revision_clock WHERE id = 1 FOR UPDATE NOWAIT",
                        (),
                        connection=competitor,
                    )
        return execute(query, *args, **kwargs)

    monkeypatch.setattr(db.backend, "execute", inspect_fence)
    output, identity = db.create_output_artifact_with_history_identity(
        type_="tts_audio",
        title="Fenced",
        format_="mp3",
        storage_path="fenced.mp3",
    )
    assert db.get_output_artifact(output.id).title == "Fenced"
    assert len(identity) == 32


def test_delivery_uses_original_identity_without_accessing_completed_files(db, storage, media):
    writer, op = removal(db, storage)
    old_identity = json.loads(op["original_json"])["incarnation"]
    old = history(media, old_identity, storage[2].id)
    newer = history(media, "e" * 32, storage[2].id)
    assert op["fs_done"] and op["reserved_bytes"] == 0
    assert not (storage[0] / "source.md").exists()
    # Delivery must work with the whole original volume offline/replaced.
    storage[0].rename(storage[0].with_name("offline"))
    storage[0].mkdir()
    (storage[0] / "source.md").write_bytes(b"replacement volume")
    assert run(writer.deliver_history_due, media)["delivered"] == 1
    assert get_history(media, old)["output_id"] is None
    assert get_history(media, newer)["output_id"] == storage[2].id
    late = history(media, old_identity, storage[2].id)
    assert get_history(media, late)["output_id"] is None
    assert (storage[0] / "source.md").read_bytes() == b"replacement volume"
    with pytest.raises(KeyError):
        db.get_output_file_operation(op["token"], storage[1])


@pytest.mark.parametrize("lost_ack", [False, True])
def test_outage_or_lost_ack_retains_effect_with_independent_backoff(db, storage, media, monkeypatch, lost_ack):
    writer, op = removal(db, storage)
    old = history(media, json.loads(op["original_json"])["incarnation"], storage[2].id)
    target = db if lost_ack else media
    method = "ack_output_file_effect" if lost_ack else "dispose_tts_output_instance"
    original = getattr(target, method)

    def unavailable(*args, **kwargs):
        raise OSError("private endpoint /secret/path")

    monkeypatch.setattr(target, method, unavailable)
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        assert run(writer.deliver_history_due, media)["retry"] == 1
    finally:
        logger.remove(sink)
    assert "private endpoint" not in "".join(messages) and "/secret/path" not in "".join(messages)
    retained = db.get_output_file_operation(op["token"], storage[1])
    assert retained["fs_done"] and retained["effects_pending"] == 1 and retained["reserved_bytes"] == 0
    assert retained["history_attempts"] == 1 and retained["history_retry_after"] > 0
    assert retained["history_error"] == "output_history_unavailable"
    assert retained["attempts"] == op["attempts"] and retained["last_error"] == op["last_error"]
    assert run(writer.deliver_history_due, media)["retry"] == 0
    assert run(writer.recover_due)["finished"] == 0
    assert (get_history(media, old)["output_id"] is None) is lost_ack
    monkeypatch.setattr(target, method, original)
    db.backend.execute("UPDATE output_file_operations SET history_retry_after = 0 WHERE token = ?", (op["token"],))
    assert run(writer.deliver_history_due, media)["delivered"] == 1
    assert get_history(media, old)["output_id"] is None


@pytest.mark.parametrize("problem", ["missing", "mismatched", "malformed"])
def test_ambiguous_effect_is_blocked_and_does_not_guess_recycled_id(db, storage, media, problem):
    writer, op = removal(db, storage)
    old = history(media, json.loads(op["original_json"])["incarnation"], storage[2].id)
    effects = json.loads(op["effects_json"])
    if problem == "missing":
        effects[0].pop("incarnation")
    elif problem == "mismatched":
        effects[0]["incarnation"] = "e" * 32
    else:
        effects[0]["deleted_at"] = "private malformed date"
    db.backend.execute(
        "UPDATE output_file_operations SET effects_json = ? WHERE token = ?", (json.dumps(effects), op["token"])
    )
    assert run(writer.deliver_history_due, media)["blocked"] == 1
    retained = db.get_output_file_operation(op["token"], storage[1])
    assert retained["history_error"] == "output_history_invalid"
    assert retained["history_retry_after"] == 2**63 - 1
    assert get_history(media, old)["output_id"] == storage[2].id
    assert run(writer.deliver_history_due, media)["blocked"] == 0


def test_acknowledgement_commit_with_lost_reply_does_not_resurrect_journal(db, storage, media, monkeypatch):
    writer, op = removal(db, storage)
    ack = db.ack_output_file_effect

    def commit_then_lose_reply(*args, **kwargs):
        assert ack(*args, **kwargs)
        raise OSError("lost acknowledgement")

    monkeypatch.setattr(db, "ack_output_file_effect", commit_then_lose_reply)
    assert run(writer.deliver_history_due, media)["retry"] == 1
    with pytest.raises(KeyError):
        db.get_output_file_operation(op["token"], storage[1])
    assert run(writer.deliver_history_due, media)["retry"] == 0
    assert media.execute_query("SELECT state FROM tts_output_instances").fetchone()["state"] == "disposed"


def test_retry_backoff_is_capped_without_expiring_delivery(db, storage):
    _, op = removal(db, storage)
    db.backend.execute(
        "UPDATE output_file_operations SET history_attempts = 2147483647 WHERE token = ?", (op["token"],)
    )
    before = int(time.time())
    db.record_output_history_failure(op["token"], storage[1], "output_history_unavailable")
    after = int(time.time())
    row = db.get_output_file_operation(op["token"], storage[1])
    assert row["history_attempts"] == 2147483647 and row["effects_pending"] == 1
    assert before + 3600 <= row["history_retry_after"] <= after + 3600


async def test_cancellation_drains_delivery_without_aborting_completed_files(db, storage, media, monkeypatch):
    writer = service(db, storage)
    op = await writer.prepare(kind="remove", output_id=storage[2].id, max_output_bytes=0)
    await writer.publish_and_commit(op["token"])
    entered, release = Event(), Event()
    original = media.dispose_tts_output_instance

    def waiting_receiver(**kwargs):
        entered.set()
        assert release.wait(10)
        return original(**kwargs)

    monkeypatch.setattr(media, "dispose_tts_output_instance", waiting_receiver)
    task = asyncio.create_task(writer.deliver_history_due(media))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(KeyError):
        db.get_output_file_operation(op["token"], storage[1])
    assert media.execute_query("SELECT state FROM tts_output_instances").fetchone()["state"] == "disposed"


async def test_concurrent_deliveries_commit_one_effect_without_resurrecting_journal(db, storage, media, monkeypatch):
    writer = service(db, storage)
    op = await writer.prepare(kind="remove", output_id=storage[2].id, max_output_bytes=0)
    await writer.publish_and_commit(op["token"])
    barrier = Barrier(2)
    original = media.dispose_tts_output_instance

    def simultaneous_receiver(**kwargs):
        barrier.wait(timeout=10)
        return original(**kwargs)

    monkeypatch.setattr(media, "dispose_tts_output_instance", simultaneous_receiver)
    results = await asyncio.gather(writer.deliver_history_due(media), writer.deliver_history_due(media))
    assert sum(result["delivered"] for result in results) == 1
    assert sum(result["skipped"] for result in results) == 1
    with pytest.raises(KeyError):
        db.get_output_file_operation(op["token"], storage[1])
    states = media.execute_query("SELECT state, disposal_token FROM tts_output_instances").fetchall()
    assert len(states) == 1 and states[0]["state"] == "disposed" and states[0]["disposal_token"] == op["token"]


def test_creation_returns_internal_identity_before_recycled_id_can_be_observed(db):
    fields = {
        "type_": "tts_audio",
        "title": "Speech",
        "format_": "mp3",
        "storage_path": "speech.mp3",
        "idempotency_key": "job-1",
    }
    output, identity = db.create_output_artifact_with_history_identity(**fields)
    assert identity == db.backend.execute("SELECT file_incarnation FROM outputs WHERE id = ?", (output.id,)).scalar
    assert db.create_output_artifact_with_history_identity(**fields) == (output, identity)
    assert "file_incarnation" not in vars(output)
    db.backend.execute("DELETE FROM outputs WHERE id = ?", (output.id,))
    db.backend.execute(
        "INSERT INTO outputs (id, user_id, type, title, format, storage_path, created_at, file_incarnation) "
        "VALUES (?, ?, 'tts_audio', 'Reused', 'mp3', 'speech.mp3', '2026-09-05', ?)",
        (output.id, db.user_id, "f" * 32),
    )
    assert identity != "f" * 32


def test_history_retry_schema_upgrade_is_repeatable_and_preserves_pending_work(db, storage):
    _, op = removal(db, storage)
    db.backend.execute("DROP INDEX idx_output_history_due", ())
    for column in ("history_attempts", "history_retry_after", "history_error"):
        db.backend.execute(f"ALTER TABLE output_file_operations DROP COLUMN {column}", ())
    db._ensure_reading_revision_schema()
    db._ensure_reading_revision_schema()
    restored = db.get_output_file_operation(op["token"], storage[1])
    assert restored["effects_json"] == op["effects_json"] and restored["fs_done"] == 1
    assert restored["history_attempts"] == restored["history_retry_after"] == 0
    assert restored["history_error"] is None


def test_delayed_failure_cannot_unblock_or_resurrect_delivered_work(db, storage, media):
    writer, op = removal(db, storage)
    db.record_output_history_failure(op["token"], storage[1], "output_history_invalid")
    blocked = db.get_output_file_operation(op["token"], storage[1])
    db.record_output_history_failure(op["token"], storage[1], "output_history_unavailable")
    assert db.get_output_file_operation(op["token"], storage[1]) == blocked
    db.backend.execute(
        "UPDATE output_file_operations SET history_retry_after = 0, history_error = NULL WHERE token = ?",
        (op["token"],),
    )
    assert run(writer.deliver_history_due, media)["delivered"] == 1
    db.record_output_history_failure(op["token"], storage[1], "output_history_unavailable")
    with pytest.raises(KeyError):
        db.get_output_file_operation(op["token"], storage[1])


def test_delivery_cannot_acknowledge_an_enclosing_receiver_transaction(db, storage, media):
    writer, op = removal(db, storage)
    with media.transaction():
        assert run(writer.deliver_history_due, media)["retry"] == 1
    assert db.get_output_file_operation(op["token"], storage[1])["effects_pending"] == 1
    assert not media.execute_query("SELECT * FROM tts_output_instances").fetchall()


def test_delivery_is_bounded_and_ignores_unfinished_files(db, storage, media):
    writer, op = removal(db, storage)
    output = db.create_output_artifact(type_="tts_audio", title="Other", format_="md", storage_path="other.md")
    (storage[0] / "other.md").write_bytes(b"other")
    prepared = run(writer.prepare, kind="remove", output_id=output.id, max_output_bytes=0)
    assert db.list_due_output_history_operations(storage[1], limit=1) == [op["token"]]
    for invalid in (0, 101, True):
        with pytest.raises(ValueError):
            run(writer.deliver_history_due, media, limit=invalid)
    assert run(writer.deliver_history_due, media, limit=1)["delivered"] == 1
    assert db.get_output_file_operation(prepared["token"], storage[1])["phase"] == "prepared"
    assert (storage[0] / "other.md").read_bytes() == b"other"


async def test_tts_job_keeps_creation_identity_when_output_is_disposed_before_history(db, storage, media, monkeypatch):
    from tldw_Server_API.app.core.TTS import tts_jobs_worker as worker

    class Speech:
        async def generate_speech(self, *args, **kwargs):
            yield b"audio"

    async def speech_service():
        return Speech()

    async def credentials(**kwargs):
        return 780, None, None

    captured = {}

    @contextmanager
    def outputs(**kwargs):
        yield db
        row = db.backend.execute("SELECT id, file_incarnation FROM outputs WHERE job_id = 456", ()).first
        captured.update(row)
        media.dispose_tts_output_instance(
            user_id=db.user_id,
            output_incarnation=row["file_incarnation"],
            disposal_token="a" * 32,
            deleted_at="2026-09-05T12:00:00+00:00",
        )
        db.backend.execute("DELETE FROM outputs WHERE id = ?", (row["id"],))
        db.backend.execute(
            "INSERT INTO outputs (id, user_id, type, title, format, storage_path, created_at, file_incarnation) "
            "VALUES (?, ?, 'tts_audio', 'New', 'mp3', 'new.mp3', '2026-09-05', ?)",
            (row["id"], db.user_id, "f" * 32),
        )

    monkeypatch.setattr(worker, "get_tts_service_v2", speech_service)
    monkeypatch.setattr(worker, "_resolve_tts_byok", credentials)
    monkeypatch.setattr(worker, "_open_media_db_for_history", lambda user: media)
    monkeypatch.setattr(worker, "_tts_history_config", lambda: {"enabled": True, "hash_key": "test-key"})
    monkeypatch.setattr(
        worker,
        "JobManager",
        lambda: SimpleNamespace(
            renew_job_lease=lambda *a, **k: True,
            update_job_progress=lambda *a, **k: True,
        ),
    )
    monkeypatch.setattr(worker, "emit_job_event", lambda *a, **k: None)
    monkeypatch.setattr(worker.DatabasePaths, "get_user_outputs_dir", lambda user: storage[0])
    monkeypatch.setattr(worker.CollectionsDatabase, "for_user", outputs)
    result = await worker._handle_tts_job(
        {
            "id": 456,
            "job_type": "tts_longform",
            "owner_user_id": db.user_id,
            "payload": {
                "speech_request": {
                    "model": "kokoro",
                    "input": "Hello",
                    "voice": "af_heart",
                    "response_format": "mp3",
                    "stream": False,
                }
            },
        }
    )
    row = media.execute_query(
        "SELECT output_id, output_incarnation, artifact_ids FROM tts_history WHERE job_id = 456"
    ).fetchone()
    assert row is not None and row["output_id"] is None and row["artifact_ids"] is None
    assert row["output_incarnation"] == captured["file_incarnation"]
    assert "output_incarnation" not in result and "file_incarnation" not in result
