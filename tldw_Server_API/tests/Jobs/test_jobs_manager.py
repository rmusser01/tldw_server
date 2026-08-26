import base64
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Jobs import migrations as jobs_migrations
from tldw_Server_API.app.core.Jobs.manager import (
    JobManager,
    JobPayloadDecryptionError,
)
from tldw_Server_API.app.core.Jobs.migrations import (
    _ensure_sqlite_archive_locators,
    ensure_jobs_tables,
)


@pytest.fixture()
def jobs_db(tmp_path):
    db_path = tmp_path / "jobs.db"
    ensure_jobs_tables(db_path)
    yield db_path


@pytest.mark.unit
def test_ensure_jobs_tables_uses_environment_path_when_no_path_is_passed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Jobs migrations use the profile environment path by default."""
    environment_path = tmp_path / "environment" / "jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(environment_path))

    resolved_path = ensure_jobs_tables()

    assert resolved_path == environment_path
    assert environment_path.exists()


@pytest.mark.unit
def test_ensure_jobs_tables_explicit_path_precedes_environment_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit jobs path takes precedence over the environment path."""
    environment_path = tmp_path / "environment" / "jobs.db"
    explicit_path = tmp_path / "explicit" / "jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(environment_path))

    resolved_path = ensure_jobs_tables(explicit_path)

    assert resolved_path == explicit_path
    assert explicit_path.exists()
    assert not environment_path.exists()


def test_sqlite_archive_collision_queries_live_in_db_management(jobs_db):
    from tldw_Server_API.app.core.DB_Management.jobs_sql_fragments import (
        fetch_slides_archive_collision_rows,
    )

    connection = sqlite3.connect(jobs_db)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            connection.execute(
                "INSERT INTO jobs "
                "(id, uuid, domain, queue, job_type, payload, status, created_at) "
                "VALUES (41, 'collision-uuid', 'slides', 'default', "
                "'presentation.generate', '{}', 'completed', DATETIME('now'))"
            )
            connection.execute(
                "INSERT INTO jobs_archive "
                "(id, uuid, domain, queue, job_type, payload, status, created_at) "
                "VALUES (41, 'collision-uuid', 'slides', 'default', "
                "'presentation.generate', '{}', 'completed', DATETIME('now'))"
            )

        collisions = fetch_slides_archive_collision_rows(
            connection,
            backend="sqlite",
            where_clause=" WHERE id = ?",
            params=(41,),
        )
    finally:
        connection.close()

    assert len(collisions) == 1
    active, archived = collisions[0]
    assert active["uuid"] == "collision-uuid"
    assert [row["uuid"] for row in archived] == ["collision-uuid"]


def _capture_sqlite_archive_query_plan(
    manager: JobManager,
    monkeypatch,
    **list_kwargs,
):
    connection = manager._connect()
    plan: list[str] = []

    class _PlanCapturingConnection:
        def execute(self, query, params=()):
            if query.lstrip().upper().startswith("SELECT"):
                plan.extend(
                    str(row[3])
                    for row in connection.execute(
                        "EXPLAIN QUERY PLAN " + query,
                        params,
                    ).fetchall()
                )
            return connection.execute(query, params)

        def close(self):
            connection.close()

    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: _PlanCapturingConnection(),
    )
    rows = manager.list_archived_jobs(**list_kwargs)
    return rows, plan


def test_jobs_models_owns_terminal_status_classification():
    from tldw_Server_API.app.core.Jobs import models

    assert frozenset({"completed", "failed", "cancelled", "quarantined"}) == (
        models.TERMINAL_JOB_STATUSES
    )
    assert all(models.is_terminal_job_status(status) for status in models.TERMINAL_JOB_STATUSES)
    assert models.is_terminal_job_status("processing") is False
    assert models.is_terminal_job_status(None) is False


def test_pg_ensure_ignores_optional_index_psycopg_error_after_required_indexes(
    monkeypatch,
):
    import psycopg

    from tldw_Server_API.app.core.Jobs import pg_migrations

    connections = []
    maintenance_calls = []

    class _Cursor:
        def __init__(self, phase):
            self.phase = phase

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, params=None):
            del params
            if self.phase == 3 and "idx_jobs_status_available_at" in str(query):
                raise psycopg.OperationalError("optional index unavailable")

    class _Connection:
        def __init__(self, phase):
            self.cursor_instance = _Cursor(phase)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return self.cursor_instance

        def commit(self):
            return None

    def _connect(_dsn, *, autocommit=False):
        del autocommit
        connection = _Connection(len(connections) + 1)
        connections.append(connection)
        return connection

    monkeypatch.setattr(psycopg, "connect", _connect)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Jobs.pg_util.negotiate_pg_dsn",
        lambda dsn: dsn,
    )
    monkeypatch.setattr(
        pg_migrations,
        "_ensure_pg_archive_locators",
        lambda _dsn: None,
    )
    monkeypatch.setattr(
        pg_migrations,
        "_ensure_pg_archive_batch_read_indexes",
        lambda _cursor: None,
    )
    monkeypatch.setattr(
        pg_migrations,
        "_mark_slides_audit_failure_pg",
        lambda _cursor: None,
    )
    monkeypatch.setattr(
        pg_migrations,
        "_audit_slides_generation_pg",
        lambda _cursor: (None, 0),
    )
    monkeypatch.setattr(
        pg_migrations,
        "ensure_job_events_pg",
        lambda _dsn: maintenance_calls.append("events"),
    )
    monkeypatch.setattr(
        pg_migrations,
        "ensure_job_counters_pg",
        lambda _dsn: maintenance_calls.append("counters"),
    )
    monkeypatch.delenv("JOBS_PG_RLS_ENABLE", raising=False)

    pg_migrations.ensure_jobs_tables_pg("postgresql://jobs.test/jobs")

    assert maintenance_calls == ["events", "counters"]


def test_create_and_acquire_and_complete(jobs_db):


    jm = JobManager(jobs_db)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export", "chatbooks_job_id": "abc"},
        owner_user_id="1",
    )
    assert job["status"] == "queued"

    nextj = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w1")
    assert nextj is not None
    assert nextj["status"] == "processing"
    ok = jm.renew_job_lease(int(nextj["id"]), seconds=30)
    assert ok
    ok2 = jm.complete_job(int(nextj["id"]))
    assert ok2
    got = jm.get_job(int(nextj["id"]))
    assert got["status"] == "completed"


def test_sqlite_create_job_rejects_secret_payload_without_persisting(jobs_db, monkeypatch):
    monkeypatch.setenv("JOBS_SECRET_REJECT", "1")
    monkeypatch.setenv("JOBS_SECRET_REDACT", "0")
    jm = JobManager(jobs_db)
    sentinel = "sk-sqlite-secret-rejection-sentinel"

    with pytest.raises(ValueError, match="Payload appears to contain secrets") as exc_info:
        jm.create_job(
            domain="secret-reject-regression",
            queue="default",
            job_type="sentinel",
            payload={"api_key": sentinel},
            owner_user_id="secret-owner",
        )

    assert sentinel not in str(exc_info.value)
    assert jm.list_jobs(
        domain="secret-reject-regression",
        job_type="sentinel",
        owner_user_id="secret-owner",
    ) == []


def test_update_job_result_merges(jobs_db):
    jm = JobManager(jobs_db)
    job = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="embeddings_pipeline",
        payload={"media_id": 1},
        owner_user_id="1",
    )
    ok1 = jm.update_job_result(int(job["id"]), result={"total_chunks": 12})
    assert ok1
    ok2 = jm.update_job_result(int(job["id"]), result={"embedding_count": 12})
    assert ok2
    updated = jm.get_job(int(job["id"]))
    assert updated["result"]["total_chunks"] == 12
    assert updated["result"]["embedding_count"] == 12


def test_acquire_decrypts_payload(jobs_db, monkeypatch):


    from tldw_Server_API.app.core.Security.crypto import encrypt_json_blob

    monkeypatch.setenv("JOBS_ENCRYPT_SECURE", "true")
    key = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODkwMTIzNDU2Nzg5MDEy"[:44]
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", key)
    if encrypt_json_blob({"probe": True}) is None:
        pytest.skip("Crypto backend unavailable; skipping encryption test")

    jm = JobManager(jobs_db)
    payload = {"secret": "value", "count": 1}
    job = jm.create_job(domain="secure", queue="default", job_type="t", payload=payload, owner_user_id="1")

    conn = sqlite3.connect(jobs_db)
    try:
        raw = conn.execute("SELECT payload FROM jobs WHERE id = ?", (int(job["id"]),)).fetchone()[0]
    finally:
        conn.close()
    raw_obj = json.loads(raw) if raw else {}
    assert isinstance(raw_obj, dict) and ("_encrypted" in raw_obj or raw_obj.get("_enc") == "aesgcm:v1")

    acq = jm.acquire_next_job(domain="secure", queue="default", lease_seconds=5, worker_id="w1")
    assert acq is not None
    assert acq["payload"] == payload


def test_replace_job_payload_rejects_mismatched_uuid_sqlite(jobs_db):
    jm = JobManager(jobs_db)
    original = {"version": "original"}
    job = jm.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=original,
        owner_user_id="1",
    )

    replaced = jm.replace_job_payload(
        int(job["id"]),
        payload={"version": "replacement"},
        expected_uuid="stale-job-uuid",
        expected_domain="prompt_studio",
    )

    assert replaced is False
    assert jm.get_job(int(job["id"]))["payload"] == original


def test_replace_job_payload_rejects_mismatched_domain_sqlite(jobs_db):
    jm = JobManager(jobs_db)
    original = {"version": "original"}
    job = jm.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=original,
        owner_user_id="1",
    )

    replaced = jm.replace_job_payload(
        int(job["id"]),
        payload={"version": "replacement"},
        expected_uuid=str(job["uuid"]),
        expected_domain="other",
    )

    assert replaced is False
    assert jm.get_job(int(job["id"]))["payload"] == original


def test_replace_job_payload_rejects_stale_uuid_after_sqlite_id_reuse(
    jobs_db,
):
    jm = JobManager(jobs_db)
    stale_job = jm.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload={"version": "stale"},
        owner_user_id="1",
    )
    acquired = jm.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=30,
        worker_id="worker-a",
    )
    assert acquired is not None
    assert jm.complete_job(int(acquired["id"]), enforce=False)
    assert jm.prune_jobs(
        statuses=["completed"],
        older_than_days=0,
        domain="prompt_studio",
    ) == 1

    replacement_job = jm.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload={"version": "new-owner"},
        owner_user_id="2",
    )
    assert int(replacement_job["id"]) == int(stale_job["id"])

    replaced = jm.replace_job_payload(
        int(stale_job["id"]),
        payload={"version": "stale-worker-overwrite"},
        expected_uuid=str(stale_job["uuid"]),
        expected_domain="prompt_studio",
    )

    assert replaced is False
    assert jm.get_job(int(replacement_job["id"]))["payload"] == {
        "version": "new-owner"
    }


def test_replace_job_payload_preserves_encryption_at_rest_and_decrypts_on_read(
    jobs_db,
    monkeypatch,
):
    from tldw_Server_API.app.core.Security.crypto import encrypt_json_blob

    monkeypatch.setenv("JOBS_ENCRYPT_SECURE", "true")
    monkeypatch.setenv(
        "WORKFLOWS_ARTIFACT_ENC_KEY",
        base64.b64encode(b"2" * 32).decode("ascii"),
    )
    if encrypt_json_blob({"probe": True}) is None:
        pytest.skip("Crypto backend unavailable; skipping encryption test")

    jm = JobManager(jobs_db)
    job = jm.create_job(
        domain="secure",
        queue="default",
        job_type="optimization",
        payload={"version": "original"},
        owner_user_id="1",
    )
    replacement = {"version": "replacement"}

    assert jm.replace_job_payload(
        int(job["id"]),
        payload=replacement,
        expected_uuid=str(job["uuid"]),
        expected_domain="secure",
    )

    conn = sqlite3.connect(jobs_db)
    try:
        raw = conn.execute(
            "SELECT payload FROM jobs WHERE id = ?",
            (int(job["id"]),),
        ).fetchone()[0]
    finally:
        conn.close()
    raw_payload = json.loads(raw)
    assert isinstance(raw_payload.get("_encrypted"), dict)
    assert jm.get_job(int(job["id"]))["payload"] == replacement


def test_replace_archived_job_payload_is_guarded_and_clears_stale_compressed_copy(
    jobs_db,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "0")
    jm = JobManager(jobs_db)
    original = {"authorization": "legacy-secret", "version": "legacy"}
    job = jm.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=original,
        owner_user_id="1",
    )
    job_id = int(job["id"])
    assert jm.cancel_job(job_id, reason="archive regression")
    assert jm.prune_jobs(
        statuses=["cancelled"],
        older_than_days=0,
        domain="prompt_studio",
    ) == 1

    conn = sqlite3.connect(jobs_db)
    try:
        conn.execute(
            "UPDATE jobs_archive SET payload = ?, payload_compressed = ? "
            "WHERE id = ?",
            (json.dumps(original), "stale-compressed-secret", job_id),
        )
        conn.commit()
    finally:
        conn.close()

    assert jm.replace_archived_job_payload(
        job_id,
        payload={"version": "wrong-uuid"},
        expected_uuid="stale-job-uuid",
        expected_domain="prompt_studio",
    ) is False
    assert jm.replace_archived_job_payload(
        job_id,
        payload={"version": "wrong-domain"},
        expected_uuid=str(job["uuid"]),
        expected_domain="other",
    ) is False
    assert jm.get_job_or_archived(job_id, domain="prompt_studio")["payload"] == original

    replacement = {"version": "secured"}
    assert jm.replace_archived_job_payload(
        job_id,
        payload=replacement,
        expected_uuid=str(job["uuid"]),
        expected_domain="prompt_studio",
    ) is True
    archived = jm.list_archived_jobs(
        domain="prompt_studio",
        status="cancelled",
        job_type="optimization",
        limit=10,
    )
    assert [row["uuid"] for row in archived] == [job["uuid"]]
    assert archived[0]["payload"] == replacement

    conn = sqlite3.connect(jobs_db)
    try:
        raw = conn.execute(
            "SELECT payload_compressed FROM jobs_archive WHERE id = ? AND uuid = ?",
            (job_id, str(job["uuid"])),
        ).fetchone()
    finally:
        conn.close()
    assert raw == (None,)


def test_prune_archive_normalizes_terminal_lease_identity_only(
    jobs_db,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    jm = JobManager(jobs_db)
    terminal = jm.create_job(
        domain="archive-lease-convergence",
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="1",
    )
    processing = jm.create_job(
        domain="archive-lease-convergence",
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="1",
    )
    conn = sqlite3.connect(jobs_db)
    try:
        with conn:
            conn.execute(
                "UPDATE jobs SET status='completed', completed_at=DATETIME('now','-1 day'), "
                "leased_until=DATETIME('now','+1 hour'), worker_id='legacy-worker', "
                "lease_id='legacy-lease' WHERE id=?",
                (int(terminal["id"]),),
            )
            conn.execute(
                "UPDATE jobs SET status='processing', created_at=DATETIME('now','-1 day'), "
                "leased_until=DATETIME('now','+1 hour'), worker_id='active-worker', "
                "lease_id='active-lease' WHERE id=?",
                (int(processing["id"]),),
            )
    finally:
        conn.close()

    assert jm.prune_jobs(
        statuses=["completed", "processing"],
        older_than_days=0,
        domain="archive-lease-convergence",
    ) == 2

    conn = sqlite3.connect(jobs_db)
    try:
        rows = conn.execute(
            "SELECT uuid, status, leased_until, worker_id, lease_id FROM jobs_archive "
            "WHERE domain='archive-lease-convergence'"
        ).fetchall()
    finally:
        conn.close()
    archived = {row[0]: row[1:] for row in rows}
    assert archived[str(terminal["uuid"])] == ("completed", None, None, None)
    assert archived[str(processing["uuid"])][0] == "processing"
    assert archived[str(processing["uuid"])][2:] == ("active-worker", "active-lease")
    assert archived[str(processing["uuid"])][1] is not None


def test_list_archived_jobs_paginates_reused_ids_with_same_created_at(
    jobs_db,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    jm = JobManager(jobs_db)
    archived_jobs = []
    for version in ("first", "second"):
        job = jm.create_job(
            domain="prompt_studio",
            queue="default",
            job_type="optimization",
            payload={"version": version},
            owner_user_id="1",
        )
        assert jm.cancel_job(int(job["id"]), reason="pagination regression")
        assert jm.prune_jobs(
            statuses=["cancelled"],
            older_than_days=0,
            domain="prompt_studio",
        ) == 1
        archived_jobs.append(job)

    assert archived_jobs[0]["id"] == archived_jobs[1]["id"]
    shared_created_at = "2026-01-01 00:00:00"
    conn = sqlite3.connect(jobs_db)
    try:
        conn.execute(
            "UPDATE jobs_archive SET created_at = ?",
            (shared_created_at,),
        )
        conn.commit()
    finally:
        conn.close()

    first_page = jm.list_archived_jobs(
        domain="prompt_studio",
        status="cancelled",
        job_type="optimization",
        limit=1,
    )
    assert len(first_page) == 1
    second_page = jm.list_archived_jobs(
        domain="prompt_studio",
        status="cancelled",
        job_type="optimization",
        created_before=datetime.fromisoformat(shared_created_at),
        before_id=int(first_page[0]["id"]),
        before_uuid=str(first_page[0]["uuid"]),
        before_archive_locator=first_page[0]["_archive_locator"],
        limit=1,
    )

    assert {
        str(first_page[0]["uuid"]),
        str(second_page[0]["uuid"]),
    } == {str(job["uuid"]) for job in archived_jobs}


def test_list_archived_jobs_paginates_same_second_microsecond_timestamps(
    jobs_db,
):
    jm = JobManager(jobs_db)
    archived_id = 17
    timestamp_versions = (
        ("2026-01-01 00:00:00.900000", "newest", "archive-newest"),
        ("2026-01-01T00:00:00.500000", "middle", "archive-middle"),
        ("2026-01-01 00:00:00.100000+00:00", "oldest", "archive-oldest"),
    )
    conn = sqlite3.connect(jobs_db)
    try:
        conn.executemany(
            "INSERT INTO jobs_archive "
            "(id, uuid, domain, queue, job_type, payload, status, created_at) "
            "VALUES (?, ?, 'prompt_studio', 'default', 'optimization', ?, "
            "'cancelled', ?)",
            [
                (
                    archived_id,
                    archived_uuid,
                    json.dumps({"version": version}),
                    timestamp,
                )
                for timestamp, version, archived_uuid in timestamp_versions
            ],
        )
        conn.commit()
    finally:
        conn.close()

    rows = []
    cursor = {}
    for _ in timestamp_versions:
        page = jm.list_archived_jobs(
            domain="prompt_studio",
            status="cancelled",
            job_type="optimization",
            limit=1,
            **cursor,
        )
        assert len(page) == 1
        row = page[0]
        rows.append(row)
        cursor = {
            "created_before": datetime.fromisoformat(
                str(row["_archive_cursor_created_at"])
            ),
            "before_id": archived_id,
            "before_uuid": str(row["_archive_cursor_uuid"]),
            "before_archive_locator": row["_archive_locator"],
        }

    assert [row["payload"]["version"] for row in rows] == [
        "newest",
        "middle",
        "oldest",
    ]
    assert len({row["_archive_locator"] for row in rows}) == len(rows)
    assert jm.list_archived_jobs(
        domain="prompt_studio",
        status="cancelled",
        job_type="optimization",
        limit=1,
        **cursor,
    ) == []


def test_list_archived_jobs_paginates_submillisecond_ties_by_locator(
    jobs_db,
):
    jm = JobManager(jobs_db)
    conn = sqlite3.connect(jobs_db)
    try:
        conn.executemany(
            "INSERT INTO jobs_archive "
            "(id, uuid, domain, queue, job_type, payload, status, created_at) "
            "VALUES (23, ?, 'prompt_studio', 'default', "
            "'optimization', ?, 'cancelled', ?)",
            (
                ("submillisecond-first", json.dumps({"version": "first"}), "2026-01-01 00:00:00.100900"),
                ("submillisecond-second", json.dumps({"version": "second"}), "2026-01-01 00:00:00.100800"),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    rows = []
    cursor = {}
    for _ in range(2):
        page = jm.list_archived_jobs(
            domain="prompt_studio",
            job_type="optimization",
            limit=1,
            **cursor,
        )
        assert len(page) == 1
        row = page[0]
        rows.append(row)
        cursor = {
            "created_before": datetime.fromisoformat(
                str(row["_archive_cursor_created_at"])
            ),
            "before_id": int(row["id"]),
            "before_uuid": str(row["_archive_cursor_uuid"]),
            "before_archive_locator": row["_archive_locator"],
        }

    assert {row["payload"]["version"] for row in rows} == {"first", "second"}
    assert len({row["_archive_locator"] for row in rows}) == 2
    assert jm.list_archived_jobs(
        domain="prompt_studio",
        job_type="optimization",
        limit=1,
        **cursor,
    ) == []


def test_list_archived_jobs_rejects_partial_pagination_cursor(jobs_db):
    jm = JobManager(jobs_db)

    with pytest.raises(
        ValueError,
        match="complete archive cursor",
    ):
        jm.list_archived_jobs(
            created_before=datetime(2026, 1, 1),
            before_id=1,
            before_uuid="legacy-job",
        )


def test_get_job_or_archived_selects_newest_reused_id_and_supports_stable_identity(
    jobs_db,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    jm = JobManager(jobs_db)
    archived_jobs = []
    for version in ("older", "newer"):
        job = jm.create_job(
            domain="prompt_studio",
            queue="default",
            job_type="optimization",
            payload={"version": version},
            owner_user_id="1",
        )
        assert jm.cancel_job(int(job["id"]), reason="identity regression")
        assert jm.prune_jobs(
            statuses=["cancelled"],
            older_than_days=0,
            domain="prompt_studio",
        ) == 1
        archived_jobs.append(job)

    older, newer = archived_jobs
    assert int(older["id"]) == int(newer["id"])

    newest_row = jm.get_job_or_archived(
        int(newer["id"]),
        domain="prompt_studio",
    )
    older_row = jm.get_job_or_archived(
        int(older["id"]),
        domain="prompt_studio",
        job_uuid=str(older["uuid"]),
    )

    assert newest_row is not None
    assert newest_row["uuid"] == newer["uuid"]
    assert newest_row["payload"] == {"version": "newer"}
    assert newest_row["_archive_locator"] is not None
    assert older_row is not None
    assert older_row["uuid"] == older["uuid"]
    assert older_row["payload"] == {"version": "older"}
    assert jm.get_job_or_archived(
        int(older["id"]),
        domain="prompt_studio",
        archive_locator=older_row["_archive_locator"],
    )["uuid"] == older["uuid"]


def test_sqlite_prune_locks_scan_through_archive_and_rejects_stale_replacement(
    jobs_db,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("JOBS_SECRET_REDACT", "0")
    monkeypatch.setenv("JOBS_SECRET_REJECT", "0")
    prune_manager = JobManager(jobs_db)
    replace_manager = JobManager(jobs_db)
    job = prune_manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload={
            "version": "legacy",
            "authorization": "legacy-secret",
        },
        owner_user_id="1",
    )
    assert prune_manager.cancel_job(int(job["id"]), reason="archive race")

    scan_started = threading.Event()
    release_scan = threading.Event()
    original_secure = prune_manager._secured_prompt_archive_payload

    def _blocking_secure(payload, *, queue):
        scan_started.set()
        assert release_scan.wait(timeout=5)
        return original_secure(payload, queue=queue)

    monkeypatch.setattr(
        prune_manager,
        "_secured_prompt_archive_payload",
        _blocking_secure,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        prune_future = executor.submit(
            prune_manager.prune_jobs,
            statuses=["cancelled"],
            older_than_days=0,
            domain="prompt_studio",
        )
        assert scan_started.wait(timeout=5)
        replace_future = executor.submit(
            replace_manager.replace_job_payload,
            int(job["id"]),
            payload={"version": "concurrent-replacement"},
            expected_uuid=str(job["uuid"]),
            expected_domain="prompt_studio",
        )
        try:
            with pytest.raises(FutureTimeoutError):
                replace_future.result(timeout=0.1)
        finally:
            release_scan.set()

        assert prune_future.result(timeout=5) == 1
        assert replace_future.result(timeout=5) is False

    archived = prune_manager.get_job_or_archived(
        int(job["id"]),
        domain="prompt_studio",
    )
    assert archived is not None
    assert archived["payload"] == {"version": "legacy"}


def test_prune_scrubs_prompt_optimization_payload_before_archiving(
    jobs_db,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("JOBS_SECRET_REDACT", "0")
    monkeypatch.setenv("JOBS_SECRET_REJECT", "0")
    jm = JobManager(jobs_db)
    sentinel = "legacy-provider-secret"
    job = jm.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload={
            "optimization_id": 17,
            "authorization": sentinel,
            "optimization_config": {
                "model_config": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": sentinel,
                }
            },
        },
        owner_user_id="1",
    )
    assert jm.cancel_job(int(job["id"]), reason="archive security regression")

    assert jm.prune_jobs(
        statuses=["cancelled"],
        older_than_days=0,
        domain="prompt_studio",
        job_type="optimization",
    ) == 1

    archived = jm.get_job_or_archived(
        int(job["id"]),
        domain="prompt_studio",
    )
    assert archived is not None
    serialized = json.dumps(archived["payload"], sort_keys=True)
    assert sentinel not in serialized
    assert "authorization" not in archived["payload"]


def test_jobs_archive_has_migration_scan_index(jobs_db):
    conn = sqlite3.connect(jobs_db)
    try:
        indexes = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list(jobs_archive)").fetchall()
        }
    finally:
        conn.close()

    assert "idx_jobs_archive_migration" in indexes


_SQLITE_ARCHIVE_BATCH_READ_INDEX_COLUMNS = {
    "idx_jobs_archive_lookup_id": [
        ("id", False),
        ("archive_id", True),
    ],
    "idx_jobs_archive_batch_group_scope": [
        ("batch_group", False),
        ("domain", False),
        ("owner_user_id", False),
        ("job_type", False),
        ("archive_id", True),
    ],
}


def _read_sqlite_archive_batch_index_columns(db_path, index_name):
    conn = sqlite3.connect(db_path)
    try:
        return [
            (str(row[2]), bool(row[3]))
            for row in conn.execute(
                f"PRAGMA index_xinfo({index_name})"
            ).fetchall()
            if bool(row[5])
        ]
    finally:
        conn.close()


def test_jobs_archive_batch_read_indexes_are_created_and_recreated(jobs_db):
    def _read_index_columns():
        return {
            index_name: _read_sqlite_archive_batch_index_columns(
                jobs_db, index_name
            )
            for index_name in _SQLITE_ARCHIVE_BATCH_READ_INDEX_COLUMNS
        }

    assert _read_index_columns() == _SQLITE_ARCHIVE_BATCH_READ_INDEX_COLUMNS

    conn = sqlite3.connect(jobs_db)
    try:
        for index_name in _SQLITE_ARCHIVE_BATCH_READ_INDEX_COLUMNS:
            conn.execute(f"DROP INDEX {index_name}")
        conn.commit()
    finally:
        conn.close()

    ensure_jobs_tables(jobs_db)

    assert _read_index_columns() == _SQLITE_ARCHIVE_BATCH_READ_INDEX_COLUMNS


@pytest.mark.parametrize(
    ("index_name", "misdefined_ddl"),
    (
        (
            "idx_jobs_archive_lookup_id",
            "CREATE INDEX idx_jobs_archive_lookup_id "
            "ON jobs_archive(id, archive_id)",
        ),
        (
            "idx_jobs_archive_batch_group_scope",
            "CREATE INDEX idx_jobs_archive_batch_group_scope "
            "ON jobs_archive(domain, batch_group, owner_user_id, job_type, "
            "archive_id)",
        ),
    ),
)
def test_sqlite_archive_batch_read_index_migration_repairs_misdefined_index(
    jobs_db,
    index_name,
    misdefined_ddl,
):
    conn = sqlite3.connect(jobs_db)
    try:
        conn.execute(f"DROP INDEX {index_name}")
        conn.execute(misdefined_ddl)
        conn.commit()
    finally:
        conn.close()

    ensure_jobs_tables(jobs_db)

    assert _read_sqlite_archive_batch_index_columns(
        jobs_db, index_name
    ) == _SQLITE_ARCHIVE_BATCH_READ_INDEX_COLUMNS[index_name]


@pytest.mark.parametrize(
    "index_name",
    tuple(_SQLITE_ARCHIVE_BATCH_READ_INDEX_COLUMNS),
)
def test_sqlite_archive_batch_read_index_migration_rejects_name_collision(
    jobs_db,
    index_name,
):
    conn = sqlite3.connect(jobs_db)
    try:
        conn.execute(f"DROP INDEX {index_name}")
        conn.execute("CREATE TABLE archive_index_name_owner (id INTEGER)")
        conn.execute(
            f"CREATE INDEX {index_name} ON archive_index_name_owner(id)"
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(RuntimeError, match=f"{index_name} belongs to another table"):
        ensure_jobs_tables(jobs_db)

    conn = sqlite3.connect(jobs_db)
    try:
        owner = conn.execute(
            "SELECT tbl_name FROM sqlite_master "
            "WHERE type = 'index' AND name = ?",
            (index_name,),
        ).fetchone()[0]
    finally:
        conn.close()

    assert owner == "archive_index_name_owner"


def test_sqlite_archive_cursor_index_handles_invalid_legacy_timestamps(
    tmp_path,
):
    db_path = tmp_path / "legacy-invalid-archive-timestamps.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE jobs_archive ("
            "id INTEGER, uuid TEXT, domain TEXT NOT NULL, queue TEXT NOT NULL, "
            "job_type TEXT NOT NULL, payload TEXT, result TEXT, "
            "payload_compressed TEXT, result_compressed TEXT, "
            "status TEXT NOT NULL, created_at TEXT, archived_at TEXT)"
        )
        conn.executemany(
            "INSERT INTO jobs_archive "
            "(id, uuid, domain, queue, job_type, payload, status, "
            "created_at, archived_at) VALUES (?, NULL, 'prompt_studio', "
            "'default', 'optimization', '{}', 'cancelled', ?, ?)",
            (
                (1, "now", "not-a-timestamp"),
                (2, "invalid-created-at", "invalid-archived-at"),
                (3, "2461041.5", "invalid-archived-at"),
                (4, "12:34:56", "invalid-archived-at"),
                (5, "2026-02-30 12:00:00", "invalid-archived-at"),
                (6, "2026-01-01 24:00:00", "invalid-archived-at"),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    ensure_jobs_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        indexes = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list(jobs_archive)").fetchall()
        }
    finally:
        conn.close()
    rows = JobManager(db_path).list_archived_jobs(
        domain="prompt_studio",
        job_type="optimization",
        limit=10,
    )

    assert "idx_jobs_archive_cursor_v2" in indexes
    assert len(rows) == 6
    assert {
        row["_archive_cursor_created_at"]
        for row in rows
    } == {
        "0001-01-01 00:00:00",
        "2026-01-02 00:00:00.000",
        "2026-03-02 12:00:00.000",
    }
    assert all(
        datetime.fromisoformat(str(row["_archive_cursor_created_at"]))
        for row in rows
    )


def test_sqlite_archive_first_page_uses_cursor_index_without_temp_sort(
    jobs_db,
    monkeypatch,
):
    conn = sqlite3.connect(jobs_db)
    try:
        conn.execute(
            "INSERT INTO jobs_archive "
            "(id, uuid, domain, queue, job_type, payload, status, created_at) "
            "VALUES (1, 'first-page-plan', 'prompt_studio', 'default', "
            "'optimization', '{}', 'cancelled', "
            "'2026-01-01 00:00:00.123456')"
        )
        conn.commit()
    finally:
        conn.close()

    rows, plan = _capture_sqlite_archive_query_plan(
        JobManager(jobs_db),
        monkeypatch,
        domain="prompt_studio",
        status=None,
        job_type="optimization",
        limit=1,
    )

    assert len(rows) == 1
    assert any("idx_jobs_archive_cursor_v2" in detail for detail in plan)
    assert not any("TEMP B-TREE" in detail for detail in plan)


def test_sqlite_archive_full_cursor_uses_cursor_index_without_temp_sort(
    jobs_db,
    monkeypatch,
):
    conn = sqlite3.connect(jobs_db)
    try:
        conn.executemany(
            "INSERT INTO jobs_archive "
            "(id, uuid, domain, queue, job_type, payload, status, created_at) "
            "VALUES (3, ?, 'prompt_studio', 'default', "
            "'optimization', '{}', 'cancelled', ?)",
            (
                ("full-cursor-newer", "2026-01-01 00:00:00.900000"),
                ("full-cursor-older", "2026-01-01 00:00:00.100000"),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    manager = JobManager(jobs_db)
    first = manager.list_archived_jobs(
        domain="prompt_studio",
        status=None,
        job_type="optimization",
        limit=1,
    )[0]
    rows, plan = _capture_sqlite_archive_query_plan(
        manager,
        monkeypatch,
        domain="prompt_studio",
        status=None,
        job_type="optimization",
        created_before=datetime.fromisoformat(
            str(first["_archive_cursor_created_at"])
        ),
        before_id=int(first["id"]),
        before_uuid=str(first["_archive_cursor_uuid"]),
        before_archive_locator=first["_archive_locator"],
        limit=1,
    )

    assert len(rows) == 1
    assert any("idx_jobs_archive_cursor_v2" in detail for detail in plan)
    assert not any("TEMP B-TREE" in detail for detail in plan)


def test_sqlite_archive_id_forward_migration_backfills_and_assigns_new_rows(
    tmp_path,
):
    db_path = tmp_path / "legacy-jobs-archive.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE jobs_archive ("
            "id INTEGER, uuid TEXT, domain TEXT NOT NULL, queue TEXT NOT NULL, "
            "job_type TEXT NOT NULL, payload TEXT, result TEXT, "
            "payload_compressed TEXT, result_compressed TEXT, "
            "status TEXT NOT NULL, created_at TEXT, archived_at TEXT)"
        )
        conn.execute(
            "INSERT INTO jobs_archive "
            "(id, uuid, domain, queue, job_type, payload, status) "
            "VALUES (1, NULL, 'prompt_studio', 'default', "
            "'optimization', '{}', 'cancelled')"
        )
        conn.commit()
    finally:
        conn.close()

    ensure_jobs_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        first_archive_id = conn.execute(
            "SELECT archive_id FROM jobs_archive WHERE id = 1"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO jobs_archive "
            "(id, uuid, domain, queue, job_type, payload, status) "
            "VALUES (2, NULL, 'prompt_studio', 'default', "
            "'optimization', '{}', 'cancelled')"
        )
        conn.commit()
        second_archive_id = conn.execute(
            "SELECT archive_id FROM jobs_archive WHERE id = 2"
        ).fetchone()[0]
    finally:
        conn.close()

    assert first_archive_id is not None
    assert second_archive_id is not None
    assert second_archive_id != first_archive_id


def test_sqlite_legacy_archive_locator_survives_delete_gap_vacuum_and_paginates(
    tmp_path,
):
    db_path = tmp_path / "legacy-jobs-archive-vacuum.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE jobs_archive ("
            "archive_id INTEGER, id INTEGER, uuid TEXT, "
            "domain TEXT NOT NULL, queue TEXT NOT NULL, "
            "job_type TEXT NOT NULL, payload TEXT, result TEXT, "
            "payload_compressed TEXT, result_compressed TEXT, "
            "status TEXT NOT NULL, created_at TEXT, archived_at TEXT)"
        )
        conn.executemany(
            "INSERT INTO jobs_archive "
            "(archive_id, id, uuid, domain, queue, job_type, payload, status, "
            "created_at) VALUES (?, ?, ?, 'prompt_studio', 'default', "
            "'optimization', ?, 'cancelled', '2026-01-01 00:00:00')",
            (
                (50, 1, "legacy-explicit", json.dumps({"version": "first"})),
                (None, 2, "legacy-gap", json.dumps({"version": "second"})),
                (None, 3, "legacy-tail", json.dumps({"version": "third"})),
            ),
        )
        conn.execute(
            "CREATE TRIGGER trg_jobs_archive_id "
            "AFTER INSERT ON jobs_archive FOR EACH ROW "
            "WHEN NEW.archive_id IS NULL BEGIN "
            "UPDATE jobs_archive SET archive_id = NEW.rowid "
            "WHERE rowid = NEW.rowid; END"
        )
        conn.commit()
    finally:
        conn.close()

    ensure_jobs_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        migrated = dict(
            conn.execute(
                "SELECT id, archive_id FROM jobs_archive ORDER BY id"
            ).fetchall()
        )
        conn.execute("DELETE FROM jobs_archive WHERE id = 2")
        conn.commit()
        conn.execute("VACUUM")
        conn.execute(
            "INSERT INTO jobs_archive "
            "(id, uuid, domain, queue, job_type, payload, status, created_at) "
            "VALUES (4, 'post-vacuum', 'prompt_studio', 'default', "
            "'optimization', ?, 'cancelled', '2026-01-01 00:00:00')",
            (json.dumps({"version": "fourth"}),),
        )
        conn.commit()
        inserted_locator = int(
            conn.execute(
                "SELECT archive_id FROM jobs_archive WHERE id = 4"
            ).fetchone()[0]
        )
        archive_id_indexes = {
            str(row[1]): bool(row[2])
            for row in conn.execute("PRAGMA index_list(jobs_archive)").fetchall()
        }
    finally:
        conn.close()

    assert int(migrated[2]) > 50
    assert int(migrated[3]) > int(migrated[2])
    assert inserted_locator > int(migrated[3])
    assert archive_id_indexes.get("idx_jobs_archive_id") is True

    manager = JobManager(db_path)
    seen: list[int] = []
    cursor: dict[str, object] = {}
    while True:
        page = manager.list_archived_jobs(
            domain="prompt_studio",
            job_type="optimization",
            status="cancelled",
            limit=1,
            **cursor,
        )
        if not page:
            break
        row = page[0]
        seen.append(int(row["_archive_locator"]))
        cursor = {
            "created_before": datetime.fromisoformat(
                str(row["_archive_cursor_created_at"])
            ),
            "before_id": int(row["id"]),
            "before_uuid": str(row["_archive_cursor_uuid"]),
            "before_archive_locator": row["_archive_locator"],
        }

    assert len(seen) == len(set(seen)) == 3
    assert set(seen) == {50, int(migrated[3]), inserted_locator}


def test_sqlite_archive_locator_migration_rolls_back_on_uniqueness_failure(
    tmp_path,
):
    db_path = tmp_path / "corrupt-legacy-jobs-archive.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE jobs_archive ("
            "archive_id INTEGER, id INTEGER, uuid TEXT, "
            "domain TEXT NOT NULL, queue TEXT NOT NULL, "
            "job_type TEXT NOT NULL, payload TEXT, result TEXT, "
            "payload_compressed TEXT, result_compressed TEXT, "
            "status TEXT NOT NULL, created_at TEXT, archived_at TEXT)"
        )
        conn.executemany(
            "INSERT INTO jobs_archive "
            "(archive_id, id, uuid, domain, queue, job_type, payload, status) "
            "VALUES (?, ?, ?, 'prompt_studio', 'default', "
            "'optimization', '{}', 'cancelled')",
            (
                (7, 1, "duplicate-one"),
                (7, 2, "duplicate-two"),
                (None, 3, "unmigrated"),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(sqlite3.IntegrityError):
        ensure_jobs_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        unmigrated_locator = conn.execute(
            "SELECT archive_id FROM jobs_archive WHERE id = 3"
        ).fetchone()[0]
        indexes = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list(jobs_archive)").fetchall()
        }
        triggers = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'trigger' AND tbl_name = 'jobs_archive'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert unmigrated_locator is None
    assert "idx_jobs_archive_id" not in indexes
    assert "trg_jobs_archive_id" not in triggers


def test_sqlite_archive_locator_migration_rejects_text_affinity(tmp_path):
    db_path = tmp_path / "text-legacy-jobs-archive.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE jobs_archive ("
            "archive_id TEXT, id INTEGER, uuid TEXT, "
            "domain TEXT NOT NULL, queue TEXT NOT NULL, "
            "job_type TEXT NOT NULL, payload TEXT, result TEXT, "
            "payload_compressed TEXT, result_compressed TEXT, "
            "status TEXT NOT NULL, created_at TEXT, archived_at TEXT)"
        )
        conn.execute(
            "INSERT INTO jobs_archive "
            "(archive_id, id, domain, queue, job_type, payload, status) "
            "VALUES ('10', 1, 'prompt_studio', 'default', 'optimization', "
            "'{}', 'cancelled')"
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(RuntimeError, match="INTEGER affinity"):
        ensure_jobs_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        column_type = next(
            str(row[2])
            for row in conn.execute("PRAGMA table_info(jobs_archive)").fetchall()
            if str(row[1]) == "archive_id"
        )
        stored_locator = conn.execute(
            "SELECT archive_id, typeof(archive_id) FROM jobs_archive"
        ).fetchone()
    finally:
        conn.close()

    assert column_type == "TEXT"
    assert stored_locator == ("10", "text")


@pytest.mark.parametrize(
    ("object_type", "object_name", "object_ddl"),
    (
        (
            "index",
            "idx_jobs_archive_id",
            "CREATE INDEX idx_jobs_archive_id ON unrelated_archive_owner(id)",
        ),
        (
            "trigger",
            "trg_jobs_archive_id",
            "CREATE TRIGGER trg_jobs_archive_id "
            "AFTER INSERT ON unrelated_archive_owner BEGIN SELECT 1; END",
        ),
    ),
)
def test_sqlite_archive_locator_migration_preserves_cross_table_object_name(
    tmp_path,
    object_type,
    object_name,
    object_ddl,
):
    db_path = tmp_path / f"cross-table-{object_type}.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE unrelated_archive_owner (id INTEGER)")
        conn.execute(object_ddl)
        conn.execute(
            "CREATE TABLE jobs_archive ("
            "id INTEGER, uuid TEXT, domain TEXT NOT NULL, "
            "queue TEXT NOT NULL, job_type TEXT NOT NULL, payload TEXT, "
            "result TEXT, payload_compressed TEXT, result_compressed TEXT, "
            "status TEXT NOT NULL, created_at TEXT, archived_at TEXT)"
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(RuntimeError, match="belongs to another table"):
        ensure_jobs_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        owner = conn.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = ? AND name = ?",
            (object_type, object_name),
        ).fetchone()[0]
    finally:
        conn.close()

    assert owner == "unrelated_archive_owner"


def test_sqlite_archive_migration_uses_scoped_busy_timeout(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "JOBS_SQLITE_ARCHIVE_MIGRATION_BUSY_TIMEOUT_MS",
        "4321",
    )
    db_path = tmp_path / "legacy-busy-timeout.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE jobs_archive ("
            "id INTEGER, archive_id INTEGER, uuid TEXT, "
            "domain TEXT NOT NULL, queue TEXT NOT NULL, "
            "job_type TEXT NOT NULL, status TEXT NOT NULL)"
        )

        _ensure_sqlite_archive_locators(conn)

        busy_timeout_ms = int(
            conn.execute("PRAGMA busy_timeout").fetchone()[0]
        )
    finally:
        conn.close()

    assert busy_timeout_ms == 4321


def test_sqlite_ensure_applies_archive_timeout_before_schema_ddl(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "JOBS_SQLITE_ARCHIVE_MIGRATION_BUSY_TIMEOUT_MS",
        "4321",
    )
    raw_connect = sqlite3.connect
    observed_timeouts: list[int] = []

    class _ConnectionProxy:
        def __init__(self, connection):
            self._connection = connection

        def __enter__(self):
            self._connection.__enter__()
            return self

        def __exit__(self, *args):
            return self._connection.__exit__(*args)

        def executescript(self, script):
            observed_timeouts.append(
                int(
                    self._connection.execute(
                        "PRAGMA busy_timeout"
                    ).fetchone()[0]
                )
            )
            return self._connection.executescript(script)

        def __getattr__(self, name):
            return getattr(self._connection, name)

    def _connect(*args, **kwargs):
        return _ConnectionProxy(raw_connect(*args, **kwargs))

    monkeypatch.setattr(jobs_migrations.sqlite3, "connect", _connect)
    jobs_migrations.ensure_jobs_tables(tmp_path / "pre-ddl-timeout.sqlite")

    assert observed_timeouts == [4321]


def test_sqlite_ensure_fails_closed_before_archive_locator_verification(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        jobs_migrations,
        "JOBS_SQLITE_DDL",
        "CREATE TABLE broken jobs schema",
    )

    with pytest.raises(sqlite3.OperationalError):
        jobs_migrations.ensure_jobs_tables(tmp_path / "invalid-schema.sqlite")


def test_sqlite_archive_compression_updates_only_new_duplicate_identity(
    jobs_db,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "1")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", "1")
    conn = sqlite3.connect(jobs_db)
    try:
        conn.execute(
            "INSERT INTO jobs_archive "
            "(id, uuid, domain, queue, job_type, payload, result, status, "
            "created_at) VALUES (77, NULL, 'archive-regression', 'default', "
            "'export', ?, ?, 'cancelled', '2020-01-01 00:00:00')",
            (
                json.dumps({"version": "old"}),
                json.dumps({"result": "old"}),
            ),
        )
        conn.execute(
            "INSERT INTO jobs "
            "(id, uuid, domain, queue, job_type, payload, result, status, "
            "created_at, completed_at) VALUES "
            "(77, NULL, 'archive-regression', 'default', 'export', ?, ?, "
            "'cancelled', '2020-01-02 00:00:00', '2020-01-02 00:00:00')",
            (
                json.dumps({"version": "new"}),
                json.dumps({"result": "new"}),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    manager = JobManager(jobs_db)
    assert manager.prune_jobs(
        statuses=["cancelled"],
        older_than_days=0,
        domain="archive-regression",
        job_type="export",
    ) == 1

    archived = manager.list_archived_jobs(
        domain="archive-regression",
        status="cancelled",
        job_type="export",
        limit=10,
    )
    conn = sqlite3.connect(jobs_db)
    try:
        compression_state = conn.execute(
            "SELECT payload IS NULL, payload_compressed IS NOT NULL "
            "FROM jobs_archive ORDER BY archive_id"
        ).fetchall()
    finally:
        conn.close()
    assert len(archived) == 2
    assert len({row["_archive_locator"] for row in archived}) == 2
    assert {row["payload"]["version"] for row in archived} == {"old", "new"}
    assert compression_state == [(0, 0), (1, 1)]


def test_sqlite_archive_without_compression_does_not_materialize_returning_rows(
    jobs_db,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "0")
    manager = JobManager(jobs_db)
    job = manager.create_job(
        domain="archive-regression",
        queue="default",
        job_type="export",
        payload={"large": "payload"},
        owner_user_id="1",
    )
    assert manager.cancel_job(int(job["id"]))

    raw_connection = manager._connect()
    archive_insert_sql: list[str] = []

    class _CursorProxy:
        def __init__(self, cursor, *, archive_insert):
            self._cursor = cursor
            self._archive_insert = archive_insert

        def fetchall(self):
            if self._archive_insert:
                raise AssertionError("archive INSERT rows were materialized")
            return self._cursor.fetchall()

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    class _ConnectionProxy:
        def __enter__(self):
            raw_connection.__enter__()
            return self

        def __exit__(self, *args):
            return raw_connection.__exit__(*args)

        def execute(self, query, params=()):
            is_archive_insert = "INSERT INTO jobs_archive" in str(query)
            if is_archive_insert:
                archive_insert_sql.append(str(query))
            return _CursorProxy(
                raw_connection.execute(query, params),
                archive_insert=is_archive_insert,
            )

        def __getattr__(self, name):
            return getattr(raw_connection, name)

    monkeypatch.setattr(manager, "_connect", lambda: _ConnectionProxy())

    assert manager.prune_jobs(
        statuses=["cancelled"],
        older_than_days=0,
        domain="archive-regression",
        job_type="export",
    ) == 1
    assert len(archive_insert_sql) == 1
    assert "RETURNING" not in archive_insert_sql[0]


def test_archive_list_raises_typed_decryption_error(jobs_db, monkeypatch):
    from tldw_Server_API.app.core.Security.crypto import encrypt_json_blob

    monkeypatch.setenv(
        "WORKFLOWS_ARTIFACT_ENC_KEY",
        base64.b64encode(b"A" * 32).decode("ascii"),
    )
    envelope = encrypt_json_blob({"authorization": "legacy-secret"})
    if envelope is None:
        pytest.skip("Crypto backend unavailable; skipping encryption test")

    conn = sqlite3.connect(jobs_db)
    try:
        conn.execute(
            "INSERT INTO jobs_archive "
            "(id, uuid, domain, queue, job_type, payload, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                "encrypted-history",
                "prompt_studio",
                "default",
                "optimization",
                json.dumps({"_encrypted": envelope}),
                "cancelled",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setenv(
        "WORKFLOWS_ARTIFACT_ENC_KEY",
        base64.b64encode(b"B" * 32).decode("ascii"),
    )

    with pytest.raises(JobPayloadDecryptionError, match="could not be decrypted"):
        JobManager(jobs_db).list_archived_jobs(
            domain="prompt_studio",
            status="cancelled",
            job_type="optimization",
            fail_on_decryption_error=True,
        )


def test_rotate_encryption_keys_respects_filters_sqlite(jobs_db, monkeypatch):


    from tldw_Server_API.app.core.Security.crypto import encrypt_json_blob

    monkeypatch.setenv("JOBS_ENCRYPT", "true")
    old_key = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODkwMTIzNDU2Nzg5MDEy"[:44]
    new_key = "MDEyMzQ1Njc4OTAxMjM0NTY3ODkwQUJDREVGR0hJSktMTU5PUFFSU1RVVldY"[:44]
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", old_key)
    if encrypt_json_blob({"probe": True}) is None:
        pytest.skip("Crypto backend unavailable; skipping encryption test")

    jm = JobManager(jobs_db)
    jm.create_job(domain="d1", queue="default", job_type="t", payload={"x": 1}, owner_user_id="1")
    jm.create_job(domain="d2", queue="default", job_type="t", payload={"x": 2}, owner_user_id="1")

    count = jm.rotate_encryption_keys(
        domain="d1",
        old_key_b64=old_key,
        new_key_b64=new_key,
        fields=["payload"],
        dry_run=True,
    )
    assert count == 1


def test_retryable_fail_and_backoff(jobs_db):


    jm = JobManager(jobs_db)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="import",
        payload={"action": "import", "chatbooks_job_id": "xyz"},
        owner_user_id="1",
        max_retries=2,
    )
    j = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w2")
    assert j is not None
    assert int(j["id"]) == int(job["id"])
    # Retryable fail schedules back to queued
    ok = jm.fail_job(int(j["id"]), error="boom", retryable=True, backoff_seconds=1)
    assert ok
    j2 = jm.get_job(int(j["id"]))
    assert j2["status"] in ("queued", "failed")
    if j2["status"] == "queued":
        assert j2["retry_count"] >= 1


def test_cancel_paths(jobs_db):


    jm = JobManager(jobs_db)
    j1 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    # cancel queued
    ok = jm.cancel_job(int(j1["id"]))
    assert ok
    j1r = jm.get_job(int(j1["id"]))
    assert j1r["status"] == "cancelled"

    # cancel request on processing
    j2 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w3")
    assert acq is not None
    assert int(acq["id"]) == int(j2["id"])
    ok2 = jm.cancel_job(int(acq["id"]))
    assert ok2
    j2r = jm.get_job(int(acq["id"]))
    # either processing with cancel_requested_at set, or cancelled if race
    assert j2r["status"] in ("processing", "cancelled")
    if j2r["status"] == "processing":
        assert j2r.get("cancel_requested_at") is not None


def test_idempotency_key_returns_existing(jobs_db):


    jm = JobManager(jobs_db)
    idem_key = "cb-export-uniq-key"
    j1 = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export"},
        owner_user_id="1",
        idempotency_key=idem_key,
    )
    assert j1["status"] == "queued"
    # Second create with same idempotency key should return the same row
    j2 = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export"},
        owner_user_id="1",
        idempotency_key=idem_key,
    )
    assert int(j2["id"]) == int(j1["id"])  # idempotent
    assert j2["status"] == "queued"


def test_available_at_scheduling_delays_acquire(jobs_db):


    jm = JobManager(jobs_db)
    future = datetime.utcnow() + timedelta(seconds=1)
    jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export"},
        owner_user_id="1",
        available_at=future,
    )
    # Should not acquire before available_at
    j = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w4")
    assert j is None
    # Wait for availability window
    import time as _t
    _t.sleep(1.2)
    j2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w4")
    assert j2 is not None
    assert j2["status"] == "processing"


def test_create_job_backfills_missing_batch_group(tmp_path, monkeypatch):


    db_path = tmp_path / "jobs_legacy.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE jobs (
              id INTEGER PRIMARY KEY,
              uuid TEXT UNIQUE,
              domain TEXT NOT NULL,
              queue TEXT NOT NULL,
              job_type TEXT NOT NULL,
              owner_user_id TEXT,
              project_id INTEGER,
              idempotency_key TEXT,
              payload TEXT,
              result TEXT,
              status TEXT NOT NULL,
              priority INTEGER DEFAULT 5,
              max_retries INTEGER DEFAULT 3,
              retry_count INTEGER DEFAULT 0,
              available_at TEXT,
              created_at TEXT,
              updated_at TEXT,
              request_id TEXT,
              trace_id TEXT
            );
            CREATE TABLE job_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              job_id INTEGER,
              domain TEXT,
              queue TEXT,
              job_type TEXT,
              event_type TEXT NOT NULL,
              attrs_json TEXT,
              owner_user_id TEXT,
              request_id TEXT,
              trace_id TEXT,
              created_at TEXT NOT NULL DEFAULT (DATETIME('now'))
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    import tldw_Server_API.app.core.Jobs.manager as jobs_manager

    def _no_migrate(path=None):
        return path if path is not None else db_path

    monkeypatch.setattr(jobs_manager, "ensure_jobs_tables", _no_migrate, raising=True)

    jm = jobs_manager.JobManager(db_path)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export"},
        owner_user_id="1",
    )
    assert job["status"] == "queued"

    conn = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        assert "batch_group" in cols
    finally:
        conn.close()


def test_count_jobs_backfills_missing_batch_group(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs_legacy_count.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE jobs (
              id INTEGER PRIMARY KEY,
              uuid TEXT UNIQUE,
              domain TEXT NOT NULL,
              queue TEXT NOT NULL,
              job_type TEXT NOT NULL,
              owner_user_id TEXT,
              project_id INTEGER,
              idempotency_key TEXT,
              payload TEXT,
              result TEXT,
              status TEXT NOT NULL,
              priority INTEGER DEFAULT 5,
              max_retries INTEGER DEFAULT 3,
              retry_count INTEGER DEFAULT 0,
              available_at TEXT,
              created_at TEXT,
              updated_at TEXT,
              request_id TEXT,
              trace_id TEXT
            );
            INSERT INTO jobs (
              id, uuid, domain, queue, job_type, owner_user_id, payload, status,
              priority, max_retries, retry_count, created_at, updated_at
            ) VALUES (
              1, 'job-1', 'chatbooks', 'default', 'export', '1', '{}', 'queued',
              5, 3, 0, '2026-04-30 00:00:00', '2026-04-30 00:00:00'
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    import tldw_Server_API.app.core.Jobs.manager as jobs_manager

    def _no_migrate(path=None):
        return path if path is not None else db_path

    monkeypatch.setattr(jobs_manager, "ensure_jobs_tables", _no_migrate, raising=True)

    jm = jobs_manager.JobManager(db_path)
    assert jm.count_jobs(domain="chatbooks", batch_group="batch-1") == 0

    conn = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        assert "batch_group" in cols
    finally:
        conn.close()


def test_dependencies_gate_acquire_and_unblock(jobs_db):


    jm = JobManager(jobs_db)
    root = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_root",
        payload={"step": "root"},
        owner_user_id="1",
        priority=5,
    )
    child = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_child",
        payload={"step": "child"},
        owner_user_id="1",
        priority=1,
    )
    assert jm.add_job_dependency(child["uuid"], root["uuid"])

    first = jm.acquire_next_job(domain="embeddings", queue="default", lease_seconds=5, worker_id="w1")
    assert first is not None
    assert first["uuid"] == root["uuid"]
    assert jm.complete_job(int(first["id"]))

    second = jm.acquire_next_job(domain="embeddings", queue="default", lease_seconds=5, worker_id="w1")
    assert second is not None
    assert second["uuid"] == child["uuid"]


def test_dependency_failure_cancels_children(jobs_db):


    jm = JobManager(jobs_db)
    root = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_root",
        payload={"step": "root"},
        owner_user_id="1",
    )
    child = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_child",
        payload={"step": "child"},
        owner_user_id="1",
    )
    assert jm.add_job_dependency(child["uuid"], root["uuid"])

    first = jm.acquire_next_job(domain="embeddings", queue="default", lease_seconds=5, worker_id="w2")
    assert first is not None
    assert jm.fail_job(int(first["id"]), error="boom", retryable=False)

    child_row = jm.get_job(int(child["id"]))
    assert child_row["status"] == "cancelled"


def test_dependency_cancel_cascades(jobs_db):


    jm = JobManager(jobs_db)
    root = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_root",
        payload={"step": "root"},
        owner_user_id="1",
    )
    child = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_child",
        payload={"step": "child"},
        owner_user_id="1",
    )
    assert jm.add_job_dependency(child["uuid"], root["uuid"])

    assert jm.cancel_job(int(root["id"]))
    child_row = jm.get_job(int(child["id"]))
    assert child_row["status"] == "cancelled"


def test_dependency_cycle_rejected(jobs_db):


    jm = JobManager(jobs_db)
    a = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_a",
        payload={"step": "a"},
        owner_user_id="1",
    )
    b = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_b",
        payload={"step": "b"},
        owner_user_id="1",
    )
    assert jm.add_job_dependency(b["uuid"], a["uuid"])
    with pytest.raises(ValueError):
        jm.add_job_dependency(a["uuid"], b["uuid"])
