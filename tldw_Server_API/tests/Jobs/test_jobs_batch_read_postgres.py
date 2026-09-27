import base64
import json
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = pytest.mark.pg_jobs


def _manager(jobs_pg_dsn: str) -> JobManager:
    return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)


def _create_job(
    manager: JobManager,
    *,
    domain: str = "claims",
    owner_user_id: str = "1",
    job_type: str = "claims_generate_analytics_export",
    batch_group: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return manager.create_job(
        domain=domain,
        queue="default",
        job_type=job_type,
        payload=payload or {"export_id": f"{domain}-{owner_user_id}-{job_type}"},
        owner_user_id=owner_user_id,
        batch_group=batch_group,
    )


def _archive_completed_job(
    manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch_group: str | None = None,
    job_type: str = "claims_generate_analytics_export",
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    job = _create_job(
        manager,
        batch_group=batch_group,
        job_type=job_type,
        payload=payload,
    )
    acquired = manager.acquire_next_job(
        domain="claims",
        queue="default",
        lease_seconds=30,
        worker_id=f"worker-{job['id']}",
    )
    assert acquired is not None
    assert int(acquired["id"]) == int(job["id"])
    assert manager.complete_job(
        int(job["id"]),
        result=result or {"ok": True},
        enforce=False,
    )
    assert manager.prune_jobs(
        statuses=["completed"],
        older_than_days=0,
        domain="claims",
        job_type=job_type,
    ) == 1
    return job


def _insert_archived_row(
    manager: JobManager,
    *,
    job_id: int,
    domain: str = "claims",
    owner_user_id: str = "1",
    job_type: str = "claims_generate_analytics_export",
    payload: dict[str, Any] | None = None,
) -> int:
    connection = manager._connect()
    try:
        with connection, manager._pg_cursor(connection) as cursor:
            cursor.execute(
                "INSERT INTO jobs_archive "
                "(id, uuid, domain, queue, job_type, owner_user_id, payload, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s) "
                "RETURNING archive_id",
                (
                    job_id,
                    f"archive-{job_id}-{domain}-{owner_user_id}-{payload}",
                    domain,
                    "default",
                    job_type,
                    owner_user_id,
                    json.dumps(payload or {}),
                    "completed",
                ),
            )
            row = cursor.fetchone()
            assert row is not None
            return int(row["archive_id"])
    finally:
        connection.close()


def test_get_jobs_by_ids_postgres_scopes_normalizes_and_reads_archives(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_ENCRYPT", "1")
    monkeypatch.setenv(
        "WORKFLOWS_ARTIFACT_ENC_KEY",
        base64.b64encode(b"1" * 32).decode("ascii"),
    )
    manager = _manager(jobs_pg_dsn)
    archived = _archive_completed_job(
        manager,
        monkeypatch,
        job_type="claims_archived_export",
        payload={"export_id": "archived-secret"},
        result={"artifact": "archived-result"},
    )
    active = _create_job(
        manager,
        job_type="claims_active_export",
        payload={"export_id": "active-secret"},
    )
    acquired = manager.acquire_next_job(
        domain="claims",
        queue="default",
        lease_seconds=30,
        worker_id="active-worker",
    )
    assert acquired is not None
    assert int(acquired["id"]) == int(active["id"])
    assert manager.complete_job(
        int(active["id"]),
        result={"artifact": "active-result"},
        enforce=False,
    )
    foreign_owner = _create_job(
        manager,
        owner_user_id="2",
        job_type="claims_foreign_export",
    )
    other_domain = _create_job(
        manager,
        domain="media",
        job_type="media_export",
    )

    rows = manager.get_jobs_by_ids(
        [
            int(active["id"]),
            int(archived["id"]),
            int(active["id"]),
            int(foreign_owner["id"]),
            int(other_domain["id"]),
        ],
        domain="claims",
        owner_user_id="1",
        include_archived=True,
    )

    active_single = manager.get_job(int(active["id"]))
    archive_single = manager.get_job_or_archived(int(archived["id"]), domain="claims")
    assert set(rows) == {int(active["id"]), int(archived["id"])}
    assert active_single is not None
    assert archive_single is not None
    assert rows[int(active["id"])]["payload"] == active_single["payload"]
    assert rows[int(active["id"])]["result"] == active_single["result"]
    assert rows[int(active["id"])]["archived"] is False
    assert rows[int(archived["id"])]["payload"] == archive_single["payload"]
    assert rows[int(archived["id"])]["result"] == archive_single["result"]
    assert rows[int(archived["id"])]["archived"] is True


def test_get_jobs_by_ids_postgres_chunks_queries_at_1000(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _create_job(manager)
    original_connect = manager._connect
    original_pg_cursor = manager._pg_cursor
    query_sizes: list[int] = []
    close_calls: list[bool] = []

    class _TrackingConnection:
        def __init__(self, connection: Any) -> None:
            self._connection = connection

        def close(self) -> None:
            close_calls.append(True)
            self._connection.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

    class _TrackingCursor:
        def __init__(self, cursor: Any) -> None:
            self._cursor = cursor

        def __enter__(self) -> "_TrackingCursor":
            self._cursor.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self._cursor.__exit__(*args)

        def execute(self, query: Any, params: Any = None) -> Any:
            if isinstance(query, str) and "FROM jobs WHERE id IN" in query:
                query_sizes.append(len(params) - 2)
            return self._cursor.execute(query, params)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._cursor, name)

    def _tracking_cursor(connection: Any) -> _TrackingCursor:
        return _TrackingCursor(original_pg_cursor(connection))

    def _tracking_connect() -> _TrackingConnection:
        return _TrackingConnection(original_connect())

    monkeypatch.setattr(manager, "_connect", _tracking_connect)
    monkeypatch.setattr(manager, "_pg_cursor", _tracking_cursor)
    ids = [int(job["id"]), *range(100_000, 101_000)]

    rows = manager.get_jobs_by_ids(
        ids,
        domain="claims",
        owner_user_id="1",
    )

    assert set(rows) == {int(job["id"])}
    assert query_sizes == [1000, 1]
    assert close_calls == [True]


def test_get_jobs_by_ids_postgres_prefers_newest_scoped_archive_row(
    jobs_pg_dsn: str,
) -> None:
    manager = _manager(jobs_pg_dsn)
    matching_id = 701
    foreign_owner_id = 702
    other_domain_id = 703
    older_archive_id = _insert_archived_row(
        manager,
        job_id=matching_id,
        payload={"version": "older"},
    )
    newer_archive_id = _insert_archived_row(
        manager,
        job_id=matching_id,
        payload={"version": "newer"},
    )
    _insert_archived_row(
        manager,
        job_id=foreign_owner_id,
        owner_user_id="2",
        payload={"scope": "foreign-owner"},
    )
    _insert_archived_row(
        manager,
        job_id=other_domain_id,
        domain="media",
        payload={"scope": "other-domain"},
    )
    assert newer_archive_id > older_archive_id

    assert manager.get_jobs_by_ids(
        [matching_id, foreign_owner_id, other_domain_id],
        domain="claims",
        owner_user_id="1",
        include_archived=False,
    ) == {}

    rows = manager.get_jobs_by_ids(
        [matching_id, foreign_owner_id, other_domain_id],
        domain="claims",
        owner_user_id="1",
        include_archived=True,
    )

    assert set(rows) == {matching_id}
    assert rows[matching_id]["archive_id"] == newer_archive_id
    assert rows[matching_id]["payload"] == {"version": "newer"}
    assert rows[matching_id]["archived"] is True


def test_get_jobs_by_ids_postgres_prefers_active_numeric_id_collision(
    jobs_pg_dsn: str,
) -> None:
    manager = _manager(jobs_pg_dsn)
    active = _create_job(manager, payload={"version": "active"})
    active_id = int(active["id"])
    _insert_archived_row(
        manager,
        job_id=active_id,
        payload={"version": "archived"},
    )

    rows = manager.get_jobs_by_ids(
        [active_id],
        domain="claims",
        owner_user_id="1",
        include_archived=True,
    )

    assert rows[active_id]["uuid"] == active["uuid"]
    assert rows[active_id]["payload"] == {"version": "active"}
    assert rows[active_id]["archived"] is False


def test_get_jobs_by_ids_postgres_rejects_invalid_values_without_connecting(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(jobs_pg_dsn)
    invalid_values: list[Any] = [
        None,
        1,
        "1",
        (1,),
        {1},
        [True],
        [0],
        [-1],
        [1.0],
        ["1"],
    ]
    for value in invalid_values:
        with pytest.raises(BadRequestError):
            manager.get_jobs_by_ids(value)

    def _unexpected_connect() -> None:
        raise AssertionError("empty batch read opened a connection")

    monkeypatch.setattr(manager, "_connect", _unexpected_connect)
    assert manager.get_jobs_by_ids([]) == {}


def test_find_job_by_batch_group_postgres_exact_scope_and_archive_fallback(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(jobs_pg_dsn)
    archived_group = "claims-analytics-export:" + "b" * 32
    archived = _archive_completed_job(
        manager,
        monkeypatch,
        batch_group=archived_group,
    )
    active_group = "claims-analytics-export:" + "a" * 32
    older = _create_job(manager, batch_group=active_group)
    newer = _create_job(manager, batch_group=active_group)
    original_connect = manager._connect
    close_calls: list[bool] = []

    class _TrackingConnection:
        def __init__(self, connection: Any) -> None:
            self._connection = connection

        def close(self) -> None:
            close_calls.append(True)
            self._connection.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

    def _tracking_connect() -> _TrackingConnection:
        return _TrackingConnection(original_connect())

    monkeypatch.setattr(manager, "_connect", _tracking_connect)

    found = manager.find_job_by_batch_group(
        batch_group=active_group,
        domain="claims",
        owner_user_id="1",
        job_type="claims_generate_analytics_export",
        include_archived=True,
    )

    assert found is not None
    assert int(found["id"]) == int(newer["id"])
    assert int(found["id"]) != int(older["id"])
    assert found["archived"] is False
    assert close_calls == [True]
    for mismatch in (
        {"owner_user_id": "2"},
        {"domain": "media"},
        {"job_type": "other_type"},
        {"batch_group": "claims-analytics-export:"},
    ):
        query = {
            "batch_group": active_group,
            "domain": "claims",
            "owner_user_id": "1",
            "job_type": "claims_generate_analytics_export",
        }
        query.update(mismatch)
        assert manager.find_job_by_batch_group(**query) is None

    assert manager.find_job_by_batch_group(
        batch_group=archived_group,
        domain="claims",
        owner_user_id="1",
        job_type="claims_generate_analytics_export",
    ) is None
    archived_found = manager.find_job_by_batch_group(
        batch_group=archived_group,
        domain="claims",
        owner_user_id="1",
        job_type="claims_generate_analytics_export",
        include_archived=True,
    )
    assert archived_found is not None
    assert int(archived_found["id"]) == int(archived["id"])
    assert archived_found["archived"] is True
