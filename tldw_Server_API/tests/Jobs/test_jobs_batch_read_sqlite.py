import base64
import json
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.Jobs.manager import JobManager


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
    domain: str = "claims",
    owner_user_id: str = "1",
    job_type: str = "claims_generate_analytics_export",
    batch_group: str | None = None,
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    preserve_id: bool = True,
) -> dict[str, Any]:
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    job = _create_job(
        manager,
        domain=domain,
        owner_user_id=owner_user_id,
        job_type=job_type,
        batch_group=batch_group,
        payload=payload,
    )
    acquired = manager.acquire_next_job(
        domain=domain,
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
    if preserve_id:
        _create_job(
            manager,
            domain="batch-read-keeper",
            owner_user_id="keeper",
            job_type=f"keeper-{job['id']}",
        )
    assert manager.prune_jobs(
        statuses=["completed"],
        older_than_days=0,
        domain=domain,
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
    batch_group: str | None = None,
    payload: dict[str, Any] | None = None,
) -> int:
    connection = manager._connect()
    try:
        with connection:
            cursor = connection.execute(
                "INSERT INTO jobs_archive "
                "(id, uuid, domain, queue, job_type, owner_user_id, "
                "batch_group, payload, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    f"archive-{job_id}-{domain}-{owner_user_id}-{payload}",
                    domain,
                    "default",
                    job_type,
                    owner_user_id,
                    batch_group,
                    json.dumps(payload or {}),
                    "completed",
                ),
            )
            return int(cursor.lastrowid)
    finally:
        connection.close()


@pytest.fixture
def manager(tmp_path: Any) -> JobManager:
    return JobManager(tmp_path / "jobs-batch-read.db")


def test_get_jobs_by_ids_scopes_active_and_archived_rows(
    manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archived = _archive_completed_job(
        manager,
        monkeypatch,
        job_type="claims_archived_export",
    )
    active = _create_job(manager, job_type="claims_active_export")
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
            int(archived["id"]),
        ],
        domain="claims",
        owner_user_id="1",
        include_archived=True,
    )

    assert set(rows) == {int(active["id"]), int(archived["id"])}
    assert rows[int(active["id"])]["job_type"] == "claims_active_export"
    assert rows[int(active["id"])]["archived"] is False
    assert rows[int(archived["id"])]["job_type"] == "claims_archived_export"
    assert rows[int(archived["id"])]["archived"] is True


def test_get_jobs_by_ids_applies_scope_to_archive_rows_and_honors_flag(
    manager: JobManager,
) -> None:
    matching_id = 501
    foreign_owner_id = 502
    other_domain_id = 503
    _insert_archived_row(manager, job_id=matching_id, payload={"scope": "match"})
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
    assert rows[matching_id]["payload"] == {"scope": "match"}
    assert rows[matching_id]["archived"] is True


@pytest.mark.parametrize(
    "job_ids",
    [
        None,
        1,
        "1",
        (1,),
        {1},
        {"id": 1},
        [True],
        [False],
        [0],
        [-1],
        [1.0],
        ["1"],
        [None],
    ],
)
def test_get_jobs_by_ids_rejects_malformed_ids(
    manager: JobManager,
    job_ids: Any,
) -> None:
    with pytest.raises(BadRequestError):
        manager.get_jobs_by_ids(job_ids)


def test_get_jobs_by_ids_empty_list_does_not_connect(
    manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_connect() -> None:
        raise AssertionError("empty batch read opened a connection")

    monkeypatch.setattr(manager, "_connect", _unexpected_connect)

    assert manager.get_jobs_by_ids([]) == {}


def test_get_jobs_by_ids_chunks_sqlite_queries_at_400(
    manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _create_job(manager)
    connection = manager._connect()
    query_sizes: list[int] = []
    close_calls: list[bool] = []

    class _TrackingConnection:
        def execute(self, query: str, params: tuple[Any, ...] = ()) -> Any:
            if "FROM jobs WHERE id IN" in query:
                query_sizes.append(len(params))
            return connection.execute(query, params)

        def close(self) -> None:
            close_calls.append(True)
            connection.close()

    monkeypatch.setattr(manager, "_connect", lambda: _TrackingConnection())
    ids = [int(job["id"]), *range(10_000, 10_400)]

    rows = manager.get_jobs_by_ids(ids)

    assert set(rows) == {int(job["id"])}
    assert query_sizes == [400, 1]
    assert close_calls == [True]


def test_get_jobs_by_ids_matches_single_row_decryption_for_active_and_archive(
    manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_ENCRYPT", "1")
    monkeypatch.setenv(
        "WORKFLOWS_ARTIFACT_ENC_KEY",
        base64.b64encode(b"0" * 32).decode("ascii"),
    )
    archived = _archive_completed_job(
        manager,
        monkeypatch,
        job_type="encrypted_archive",
        payload={"export_id": "archived-secret"},
        result={"artifact": "archived-result"},
    )
    active = _create_job(
        manager,
        job_type="encrypted_active",
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

    rows = manager.get_jobs_by_ids(
        [int(active["id"]), int(archived["id"])],
        domain="claims",
        owner_user_id="1",
        include_archived=True,
    )

    active_single = manager.get_job(int(active["id"]))
    archive_single = manager.get_job_or_archived(int(archived["id"]), domain="claims")
    assert active_single is not None
    assert archive_single is not None
    assert rows[int(active["id"])]["payload"] == active_single["payload"]
    assert rows[int(active["id"])]["result"] == active_single["result"]
    assert rows[int(archived["id"])]["payload"] == archive_single["payload"]
    assert rows[int(archived["id"])]["result"] == archive_single["result"]


def test_get_jobs_by_ids_prefers_active_row_on_reused_sqlite_id(
    manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archived = _archive_completed_job(
        manager,
        monkeypatch,
        job_type="collision",
        payload={"version": "archived"},
        preserve_id=False,
    )
    active = _create_job(
        manager,
        job_type="collision",
        payload={"version": "active"},
    )
    assert int(active["id"]) == int(archived["id"])

    rows = manager.get_jobs_by_ids(
        [int(active["id"])],
        domain="claims",
        owner_user_id="1",
        include_archived=True,
    )

    assert rows[int(active["id"])]["uuid"] == active["uuid"]
    assert rows[int(active["id"])]["payload"] == {"version": "active"}
    assert rows[int(active["id"])]["archived"] is False


def test_get_jobs_by_ids_prefers_newest_archive_row_for_duplicate_id(
    manager: JobManager,
) -> None:
    job_id = 701
    older_archive_id = _insert_archived_row(
        manager,
        job_id=job_id,
        payload={"version": "older"},
    )
    newer_archive_id = _insert_archived_row(
        manager,
        job_id=job_id,
        payload={"version": "newer"},
    )
    assert newer_archive_id > older_archive_id

    rows = manager.get_jobs_by_ids(
        [job_id],
        domain="claims",
        owner_user_id="1",
        include_archived=True,
    )

    assert rows[job_id]["archive_id"] == newer_archive_id
    assert rows[job_id]["payload"] == {"version": "newer"}
    assert rows[job_id]["archived"] is True


def test_find_job_by_batch_group_requires_exact_scope_and_returns_newest_active(
    manager: JobManager,
) -> None:
    batch_group = "claims-analytics-export:" + "a" * 32
    older = _create_job(manager, batch_group=batch_group)
    newer = _create_job(manager, batch_group=batch_group)

    found = manager.find_job_by_batch_group(
        batch_group=batch_group,
        domain="claims",
        owner_user_id="1",
        job_type="claims_generate_analytics_export",
        include_archived=True,
    )

    assert found is not None
    assert int(found["id"]) == int(newer["id"])
    assert int(found["id"]) != int(older["id"])
    assert found["archived"] is False
    assert manager.find_job_by_batch_group(
        batch_group=batch_group,
        domain="claims",
        owner_user_id="2",
        job_type="claims_generate_analytics_export",
    ) is None
    assert manager.find_job_by_batch_group(
        batch_group=batch_group,
        domain="media",
        owner_user_id="1",
        job_type="claims_generate_analytics_export",
    ) is None
    assert manager.find_job_by_batch_group(
        batch_group=batch_group,
        domain="claims",
        owner_user_id="1",
        job_type="other_type",
    ) is None
    assert manager.find_job_by_batch_group(
        batch_group="claims-analytics-export:",
        domain="claims",
        owner_user_id="1",
        job_type="claims_generate_analytics_export",
    ) is None


def test_find_job_by_batch_group_closes_sqlite_connection(
    manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_group = "claims-analytics-export:" + "c" * 32
    job = _create_job(manager, batch_group=batch_group)
    connection = manager._connect()
    close_calls: list[bool] = []

    class _TrackingConnection:
        def execute(self, query: str, params: tuple[Any, ...] = ()) -> Any:
            return connection.execute(query, params)

        def close(self) -> None:
            close_calls.append(True)
            connection.close()

    monkeypatch.setattr(manager, "_connect", lambda: _TrackingConnection())

    found = manager.find_job_by_batch_group(
        batch_group=batch_group,
        domain="claims",
        owner_user_id="1",
        job_type="claims_generate_analytics_export",
    )

    assert found is not None
    assert int(found["id"]) == int(job["id"])
    assert close_calls == [True]


def test_find_job_by_batch_group_repairs_missing_active_batch_group_once(
    manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = manager._connect()
    try:
        connection.execute("DROP INDEX IF EXISTS idx_jobs_batch_group")
        connection.execute("ALTER TABLE jobs DROP COLUMN batch_group")
        connection.commit()
    finally:
        connection.close()

    original_repair = manager._sqlite_ensure_batch_group
    repair_calls: list[bool] = []

    def _tracking_repair(connection: Any) -> bool:
        repair_calls.append(True)
        return original_repair(connection)

    monkeypatch.setattr(manager, "_sqlite_ensure_batch_group", _tracking_repair)

    assert manager.find_job_by_batch_group(
        batch_group="claims-analytics-export:legacy-active",
        domain="claims",
        owner_user_id="1",
        job_type="claims_generate_analytics_export",
    ) is None

    connection = manager._connect()
    try:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }
        indexes = {
            str(row[1])
            for row in connection.execute("PRAGMA index_list(jobs)").fetchall()
        }
    finally:
        connection.close()

    assert repair_calls == [True]
    assert "batch_group" in columns
    assert "idx_jobs_batch_group" in indexes


def test_find_job_by_batch_group_repairs_missing_archive_batch_group_once(
    manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = manager._connect()
    try:
        connection.execute(
            "DROP INDEX IF EXISTS idx_jobs_archive_batch_group_scope"
        )
        connection.execute("ALTER TABLE jobs_archive DROP COLUMN batch_group")
        connection.commit()
    finally:
        connection.close()

    original_repair = manager._sqlite_ensure_batch_group
    repair_calls: list[bool] = []

    def _tracking_repair(connection: Any) -> bool:
        repair_calls.append(True)
        return original_repair(connection)

    monkeypatch.setattr(manager, "_sqlite_ensure_batch_group", _tracking_repair)

    assert manager.find_job_by_batch_group(
        batch_group="claims-analytics-export:legacy-archive",
        domain="claims",
        owner_user_id="1",
        job_type="claims_generate_analytics_export",
        include_archived=True,
    ) is None

    connection = manager._connect()
    try:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(jobs_archive)"
            ).fetchall()
        }
        indexes = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA index_list(jobs_archive)"
            ).fetchall()
        }
    finally:
        connection.close()

    assert repair_calls == [True]
    assert "batch_group" in columns
    assert "idx_jobs_archive_batch_group_scope" in indexes


def test_find_job_by_batch_group_uses_archive_only_when_requested(
    manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_group = "claims-analytics-export:" + "b" * 32
    archived = _archive_completed_job(
        manager,
        monkeypatch,
        batch_group=batch_group,
    )

    assert manager.find_job_by_batch_group(
        batch_group=batch_group,
        domain="claims",
        owner_user_id="1",
        job_type="claims_generate_analytics_export",
    ) is None

    found = manager.find_job_by_batch_group(
        batch_group=batch_group,
        domain="claims",
        owner_user_id="1",
        job_type="claims_generate_analytics_export",
        include_archived=True,
    )

    assert found is not None
    assert int(found["id"]) == int(archived["id"])
    assert found["archived"] is True
