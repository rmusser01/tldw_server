import contextlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs

from tldw_Server_API.app.core.Jobs.manager import JobManager


@pytest.fixture(autouse=True)
def _setup_pg_env(jobs_pg_dsn, monkeypatch):
     # Standardize env per test and avoid DDL races inside JobManager
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    try:
        import tldw_Server_API.app.core.Jobs.manager as _jm
        monkeypatch.setattr(_jm, "ensure_jobs_tables_pg", lambda url: url, raising=False)
    except Exception:
        _ = None
    yield


def _new_pg_manager():


    return JobManager(None, backend="postgres", db_url=os.getenv("JOBS_DB_URL"))


def _walk_pg_explain_nodes(value):
    if isinstance(value, dict):
        if "Node Type" in value:
            yield value
        for child in value.values():
            yield from _walk_pg_explain_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_pg_explain_nodes(child)


def test_pg_create_acquire_complete_idempotent():


    jm = _new_pg_manager()
    idem = "pg-idem-1"
    j1 = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export", "chatbooks_job_id": "p1"},
        owner_user_id="1",
        idempotency_key=idem,
    )
    j2 = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export", "chatbooks_job_id": "p1"},
        owner_user_id="1",
        idempotency_key=idem,
    )
    assert int(j1["id"]) == int(j2["id"])  # idempotent
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=10, worker_id="w1")
    assert acq is not None
    assert acq["status"] == "processing"
    ok = jm.complete_job(int(acq["id"]))
    assert ok


def test_pg_exact_scoped_idempotency_lookup_is_active_first_and_archive_aware(monkeypatch):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", "graph-suggestions")
    jm = _new_pg_manager()
    job = jm.create_job(
        domain="notes",
        queue="graph-suggestions",
        job_type="note_graph_suggestions",
        payload={"run_id": "pg-run-lookup"},
        owner_user_id="pg-owner-lookup",
        idempotency_key="pg-run-lookup",
        max_retries=0,
    )

    lookup = jm.get_job_or_archived_by_idempotency_key(
        idempotency_key="pg-run-lookup",
        domain="notes",
        queue="graph-suggestions",
        job_type="note_graph_suggestions",
        owner_user_id="pg-owner-lookup",
    )
    assert lookup is not None and lookup["uuid"] == job["uuid"] and lookup["archived"] is False
    assert (
        jm.get_job_or_archived_by_idempotency_key(
            idempotency_key="pg-run-lookup",
            domain="notes",
            queue="wrong-queue",
            job_type="note_graph_suggestions",
            owner_user_id="pg-owner-lookup",
        )
        is None
    )

    leased = jm.acquire_next_job(
        domain="notes",
        queue="graph-suggestions",
        lease_seconds=30,
        worker_id="pg-lookup-worker",
    )
    assert leased is not None and leased["uuid"] == job["uuid"]
    assert jm.complete_job(int(leased["id"]))
    conn = jm._connect()
    try:
        with conn, jm._pg_cursor(conn) as cur:
            cur.execute(
                "UPDATE jobs SET completed_at=%s WHERE uuid=%s",
                (datetime(2000, 1, 1, tzinfo=timezone.utc), job["uuid"]),
            )
    finally:
        conn.close()
    assert jm.prune_jobs(statuses=["completed"], older_than_days=31) == 1

    archived = jm.get_job_or_archived_by_idempotency_key(
        idempotency_key="pg-run-lookup",
        domain="notes",
        queue="graph-suggestions",
        job_type="note_graph_suggestions",
        owner_user_id="pg-owner-lookup",
    )
    assert archived is not None and archived["uuid"] == job["uuid"] and archived["archived"] is True


def test_pg_create_job_rejects_secret_payload_without_persisting(monkeypatch):
    monkeypatch.setenv("JOBS_SECRET_REJECT", "1")
    monkeypatch.setenv("JOBS_SECRET_REDACT", "0")
    jm = _new_pg_manager()
    sentinel = "sk-postgres-secret-rejection-sentinel"

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


def test_pg_replace_job_payload_honors_uuid_and_domain_guards():
    jm = _new_pg_manager()
    original = {"version": "original"}
    job = jm.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=original,
        owner_user_id="1",
    )
    job_id = int(job["id"])

    assert jm.replace_job_payload(
        job_id,
        payload={"version": "wrong-uuid"},
        expected_uuid="stale-job-uuid",
        expected_domain="prompt_studio",
    ) is False
    assert jm.replace_job_payload(
        job_id,
        payload={"version": "wrong-domain"},
        expected_uuid=str(job["uuid"]),
        expected_domain="other",
    ) is False
    assert jm.get_job(job_id)["payload"] == original

    replacement = {"version": "replacement"}
    assert jm.replace_job_payload(
        job_id,
        payload=replacement,
        expected_uuid=str(job["uuid"]),
        expected_domain="prompt_studio",
    ) is True
    assert jm.get_job(job_id)["payload"] == replacement


def test_pg_replace_job_payload_holds_row_lock_through_serialization(
    monkeypatch,
):
    jm = _new_pg_manager()
    job = jm.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload={"version": "original"},
        owner_user_id="1",
    )
    job_id = int(job["id"])
    entered_serialization = threading.Event()
    release_serialization = threading.Event()
    original_encrypt = jm._maybe_encrypt_json

    def _blocking_encrypt(payload, domain):
        entered_serialization.set()
        assert release_serialization.wait(timeout=5)
        return original_encrypt(payload, domain)

    monkeypatch.setattr(jm, "_maybe_encrypt_json", _blocking_encrypt)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            jm.replace_job_payload,
            job_id,
            payload={"version": "replacement"},
            expected_uuid=str(job["uuid"]),
            expected_domain="prompt_studio",
        )
        assert entered_serialization.wait(timeout=5)

        contender = psycopg.connect(os.environ["JOBS_DB_URL"])
        try:
            with pytest.raises(psycopg.errors.LockNotAvailable):
                with contender, contender.cursor() as cur:
                    cur.execute("SET LOCAL lock_timeout = '100ms'")
                    cur.execute(
                        "UPDATE jobs SET payload = %s::jsonb WHERE id = %s",
                        ('{"version":"contender"}', job_id),
                    )
        finally:
            contender.close()
            release_serialization.set()

        assert future.result(timeout=5) is True

    assert jm.get_job(job_id)["payload"] == {"version": "replacement"}


def test_pg_replace_archived_job_payload_is_guarded_and_clears_compressed_copy(
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "0")
    jm = _new_pg_manager()
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

    conn = psycopg.connect(os.environ["JOBS_DB_URL"])
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs_archive SET payload = %s::jsonb, "
                "payload_compressed = %s WHERE id = %s AND uuid = %s",
                (
                    '{"authorization":"legacy-secret","version":"legacy"}',
                    b"stale-compressed-secret",
                    job_id,
                    str(job["uuid"]),
                ),
            )
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
    assert any(
        row["uuid"] == job["uuid"] and row["payload"] == replacement
        for row in archived
    )

    conn = psycopg.connect(os.environ["JOBS_DB_URL"])
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload_compressed FROM jobs_archive WHERE id = %s AND uuid = %s",
                (job_id, str(job["uuid"])),
            )
            raw = cur.fetchone()
    finally:
        conn.close()
    assert raw == (None,)


def test_pg_prune_archive_normalizes_terminal_lease_identity_only(monkeypatch):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    jm = _new_pg_manager()
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
    conn = psycopg.connect(os.environ["JOBS_DB_URL"])
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status='completed', completed_at=NOW() - interval '1 day', "
                "leased_until=NOW() + interval '1 hour', worker_id='legacy-worker', "
                "lease_id='legacy-lease' WHERE id=%s",
                (int(terminal["id"]),),
            )
            cur.execute(
                "UPDATE jobs SET status='processing', created_at=NOW() - interval '1 day', "
                "leased_until=NOW() + interval '1 hour', worker_id='active-worker', "
                "lease_id='active-lease' WHERE id=%s",
                (int(processing["id"]),),
            )
    finally:
        conn.close()

    assert jm.prune_jobs(
        statuses=["completed", "processing"],
        older_than_days=0,
        domain="archive-lease-convergence",
    ) == 2

    conn = psycopg.connect(os.environ["JOBS_DB_URL"])
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT uuid, status, leased_until, worker_id, lease_id FROM jobs_archive "
                "WHERE domain='archive-lease-convergence'"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    archived = {str(row[0]): row[1:] for row in rows}
    assert archived[str(terminal["uuid"])] == ("completed", None, None, None)
    assert archived[str(processing["uuid"])][0] == "processing"
    assert archived[str(processing["uuid"])][2:] == ("active-worker", "active-lease")
    assert archived[str(processing["uuid"])][1] is not None


def test_pg_list_archived_jobs_paginates_reused_ids_with_same_created_at(
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    jm = _new_pg_manager()
    jobs = [
        jm.create_job(
            domain="prompt_studio",
            queue="default",
            job_type="optimization",
            payload={"version": version},
            owner_user_id="1",
        )
        for version in ("first", "second")
    ]
    for job in jobs:
        assert jm.cancel_job(int(job["id"]), reason="pagination regression")
    assert jm.prune_jobs(
        statuses=["cancelled"],
        older_than_days=0,
        domain="prompt_studio",
    ) == 2

    shared_id = int(jobs[0]["id"])
    shared_created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    conn = psycopg.connect(os.environ["JOBS_DB_URL"])
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs_archive SET id = %s, created_at = %s "
                "WHERE uuid IN (%s, %s)",
                (
                    shared_id,
                    shared_created_at,
                    str(jobs[0]["uuid"]),
                    str(jobs[1]["uuid"]),
                ),
            )
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
        created_before=shared_created_at,
        before_id=shared_id,
        before_uuid=str(first_page[0]["uuid"]),
        before_archive_locator=str(first_page[0]["_archive_locator"]),
        limit=1,
    )

    assert {
        str(first_page[0]["uuid"]),
        str(second_page[0]["uuid"]),
    } == {str(job["uuid"]) for job in jobs}


def test_pg_archive_compression_updates_only_new_duplicate_identity(
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "1")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", "1")
    manager = _new_pg_manager()
    conn = psycopg.connect(os.environ["JOBS_DB_URL"])
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO jobs_archive "
                "(id, uuid, domain, queue, job_type, payload, result, status, "
                "created_at) VALUES (77, NULL, 'archive-regression', "
                "'default', 'export', %s::jsonb, %s::jsonb, 'cancelled', %s)",
                (
                    '{"version":"old"}',
                    '{"result":"old"}',
                    datetime(2020, 1, 1, tzinfo=timezone.utc),
                ),
            )
            cur.execute(
                "INSERT INTO jobs "
                "(id, uuid, domain, queue, job_type, payload, result, status, "
                "created_at, completed_at) VALUES "
                "(77, NULL, 'archive-regression', 'default', 'export', "
                "%s::jsonb, %s::jsonb, 'cancelled', %s, %s)",
                (
                    '{"version":"new"}',
                    '{"result":"new"}',
                    datetime(2020, 1, 2, tzinfo=timezone.utc),
                    datetime(2020, 1, 2, tzinfo=timezone.utc),
                ),
            )
    finally:
        conn.close()

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
    conn = psycopg.connect(os.environ["JOBS_DB_URL"])
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload IS NULL, payload_compressed IS NOT NULL "
                "FROM jobs_archive ORDER BY archive_id"
            )
            compression_state = cur.fetchall()
    finally:
        conn.close()
    assert len(archived) == 2
    assert len({row["_archive_locator"] for row in archived}) == 2
    assert {row["payload"]["version"] for row in archived} == {"old", "new"}
    assert compression_state == [(False, False), (True, True)]


def test_pg_archive_without_compression_does_not_materialize_returning_rows(
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "0")
    manager = _new_pg_manager()
    job = manager.create_job(
        domain="archive-regression",
        queue="default",
        job_type="export",
        payload={"large": "payload"},
        owner_user_id="1",
    )
    assert manager.cancel_job(int(job["id"]))
    original_pg_cursor = manager._pg_cursor
    archive_insert_sql: list[str] = []

    class _CursorProxy:
        def __init__(self, cursor):
            self._cursor = cursor
            self._archive_insert = False

        def execute(self, query, params=()):
            self._archive_insert = "INSERT INTO jobs_archive" in str(query)
            if self._archive_insert:
                archive_insert_sql.append(str(query))
            return self._cursor.execute(query, params)

        def fetchall(self):
            if self._archive_insert:
                raise AssertionError("archive INSERT rows were materialized")
            return self._cursor.fetchall()

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    @contextlib.contextmanager
    def _recording_pg_cursor(conn):
        with original_pg_cursor(conn) as cursor:
            yield _CursorProxy(cursor)

    monkeypatch.setattr(manager, "_pg_cursor", _recording_pg_cursor)

    assert manager.prune_jobs(
        statuses=["cancelled"],
        older_than_days=0,
        domain="archive-regression",
        job_type="export",
    ) == 1
    assert len(archive_insert_sql) == 1
    assert "RETURNING" not in archive_insert_sql[0]


def test_pg_cancel_job_honors_atomic_identity_guards():
    jm = _new_pg_manager()
    job = jm.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload={"optimization_id": 17},
        owner_user_id="1",
    )

    assert jm.cancel_job(
        int(job["id"]),
        expected_uuid="stale-job-uuid",
        expected_domain="prompt_studio",
        expected_job_type="optimization",
    ) is False
    assert jm.cancel_job(
        int(job["id"]),
        expected_uuid=str(job["uuid"]),
        expected_domain="other",
        expected_job_type="optimization",
    ) is False
    assert jm.cancel_job(
        int(job["id"]),
        expected_uuid=str(job["uuid"]),
        expected_domain="prompt_studio",
        expected_job_type="evaluation",
    ) is False
    assert jm.get_job(int(job["id"]))["status"] == "queued"

    assert jm.cancel_job(
        int(job["id"]),
        expected_uuid=str(job["uuid"]),
        expected_domain="prompt_studio",
        expected_job_type="optimization",
    ) is True


def test_pg_prune_scrubs_prompt_optimization_payload_before_archiving(
    monkeypatch,
):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    monkeypatch.setenv("JOBS_SECRET_REDACT", "0")
    monkeypatch.setenv("JOBS_SECRET_REJECT", "0")
    jm = _new_pg_manager()
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
    assert sentinel not in str(archived["payload"])
    assert "authorization" not in archived["payload"]


def test_pg_archive_migration_index_exists():
    conn = psycopg.connect(os.environ["JOBS_DB_URL"])
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_indexes WHERE schemaname = current_schema() "
                "AND tablename = 'jobs_archive' "
                "AND indexname = 'idx_jobs_archive_migration'"
            )
            row = cur.fetchone()
    finally:
        conn.close()

    assert row == (1,)


@pytest.mark.parametrize("with_cursor", [False, True])
def test_pg_archive_status_unfiltered_plan_uses_cursor_index_without_sort(
    with_cursor,
):
    cursor_time_sql = (
        "COALESCE(created_at, archived_at, "
        "TIMESTAMPTZ '0001-01-01 00:00:00+00')"
    )
    query = (
        "EXPLAIN (FORMAT JSON, COSTS FALSE) "
        "SELECT archive_id FROM jobs_archive "
        "WHERE domain = %s AND job_type = %s"
    )
    params = ["prompt_studio", "optimization"]
    if with_cursor:
        query += (
            f" AND ({cursor_time_sql} < %s OR "
            f"({cursor_time_sql} = %s AND "
            "(id < %s OR (id = %s AND "
            "(COALESCE(uuid, '') < %s OR "
            "(COALESCE(uuid, '') = %s AND archive_id < %s))))))"
        )
        params.extend(
            [
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                17,
                17,
                "cursor-uuid",
                "cursor-uuid",
                99,
            ]
        )
    query += (
        f" ORDER BY {cursor_time_sql} DESC, id DESC, "
        "COALESCE(uuid, '') DESC, archive_id DESC LIMIT 1"
    )

    conn = psycopg.connect(os.environ["JOBS_DB_URL"])
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SET LOCAL enable_seqscan = off")
            cur.execute(query, tuple(params))
            explain = cur.fetchone()[0]
    finally:
        conn.close()
    nodes = list(_walk_pg_explain_nodes(explain))

    assert any(
        node.get("Index Name") == "idx_jobs_archive_cursor_v2"
        for node in nodes
    )
    assert not any(node.get("Node Type") == "Sort" for node in nodes)


def test_pg_archive_null_uuid_cursor_uses_exact_row_locator():
    jm = _new_pg_manager()
    shared_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    conn = psycopg.connect(os.environ["JOBS_DB_URL"])
    try:
        with conn, conn.cursor() as cur:
            for version in ("first", "second"):
                cur.execute(
                    "INSERT INTO jobs_archive "
                    "(id, uuid, domain, queue, job_type, payload, status, "
                    "created_at, archived_at) "
                    "VALUES (%s, NULL, %s, %s, %s, %s::jsonb, %s, NULL, %s)",
                    (
                        91,
                        "prompt_studio",
                        "default",
                        "optimization",
                        '{"version":"' + version + '","authorization":"secret"}',
                        "cancelled",
                        shared_time,
                    ),
                )
    finally:
        conn.close()

    first_page = jm.list_archived_jobs(
        domain="prompt_studio",
        status="cancelled",
        job_type="optimization",
        limit=1,
    )
    assert len(first_page) == 1
    first = first_page[0]
    assert jm.replace_archived_job_payload(
        int(first["id"]),
        payload={"version": "secured"},
        expected_domain="prompt_studio",
        expected_archive_locator=first["_archive_locator"],
    )
    second_page = jm.list_archived_jobs(
        domain="prompt_studio",
        status="cancelled",
        job_type="optimization",
        created_before=first["_archive_cursor_created_at"],
        before_id=int(first["id"]),
        before_uuid="",
        before_archive_locator=first["_archive_locator"],
        limit=1,
    )

    assert len(second_page) == 1
    assert second_page[0]["_archive_locator"] != first["_archive_locator"]


def test_pg_concurrent_acquire_skip_locked():


    jm = _new_pg_manager()
    # Seed 4 jobs
    ids = []
    for i in range(4):
        j = jm.create_job(
            domain="chatbooks",
            queue="default",
            job_type="export",
            payload={"action": "export", "chatbooks_job_id": f"pj{i}"},
            owner_user_id="1",
        )
        ids.append(int(j["id"]))

    def acq_one(tag):

        jmx = _new_pg_manager()
        got = jmx.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id=tag)
        return got["id"] if got else None

    with ThreadPoolExecutor(max_workers=3) as ex:
        f1 = ex.submit(acq_one, "wA")
        f2 = ex.submit(acq_one, "wB")
        f3 = ex.submit(acq_one, "wC")
        r1, r2, r3 = f1.result(), f2.result(), f3.result()

    got_ids = {r for r in (r1, r2, r3) if r is not None}
    # Expect at least 2 distinct jobs acquired without conflict
    assert len(got_ids) >= 2


def test_pg_reschedule_persists_updates():


    jm = _new_pg_manager()
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export"},
        owner_user_id="1",
        available_at=future,
    )
    before = jm.get_job(int(job["id"]))["available_at"]
    assert before is not None
    count = jm.reschedule_jobs(
        domain="chatbooks",
        queue="default",
        job_type="export",
        status="queued",
        set_now=True,
        dry_run=False,
    )
    assert count >= 1
    after = jm.get_job(int(job["id"]))["available_at"]
    assert after is None
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w-resched")
    assert acq and int(acq["id"]) == int(job["id"])
