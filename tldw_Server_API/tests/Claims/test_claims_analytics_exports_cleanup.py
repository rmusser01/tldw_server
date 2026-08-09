from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_analytics_exports as exports
from tldw_Server_API.app.core.Claims_Extraction.claims_analytics_exports import (
    cleanup_export_artifacts,
    hydrate_job_statuses,
    reconcile_export_artifacts,
)

NOW = datetime(2026, 8, 8, 12, tzinfo=timezone.utc)


def _timestamp(seconds_ago: float) -> str:
    value = NOW - timedelta(seconds=seconds_ago)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _artifact(
    export_number: int,
    *,
    status: str = "queued",
    job_id: Any = None,
    created_seconds_ago: float = 600,
    updated_seconds_ago: float | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    return {
        "export_id": f"{export_number:032x}",
        "user_id": "7",
        "format": "json",
        "status": status,
        "job_id": job_id,
        "error_code": error_code,
        "created_at": _timestamp(created_seconds_ago),
        "updated_at": _timestamp(created_seconds_ago if updated_seconds_ago is None else updated_seconds_ago),
    }


class MaintenanceDB:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = {row["export_id"]: dict(row) for row in rows}
        self.list_calls: list[dict[str, Any]] = []
        self.attach_calls: list[dict[str, Any]] = []
        self.transition_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.before_delete: Any = None

    def list_claims_analytics_exports_for_maintenance(
        self,
        *,
        user_id: str,
        limit: int,
        statuses: tuple[str, ...] | None = None,
        job_id_missing: bool | None = None,
        updated_before: str | None = None,
        export_id_after: str | None = None,
        export_id_at_or_before: str | None = None,
    ) -> list[dict[str, Any]]:
        self.list_calls.append(
            {
                "user_id": user_id,
                "limit": limit,
                "statuses": statuses,
                "job_id_missing": job_id_missing,
                "updated_before": updated_before,
                "export_id_after": export_id_after,
                "export_id_at_or_before": export_id_at_or_before,
            }
        )
        rows = [row for row in self.rows.values() if row["user_id"] == user_id]
        if statuses is not None:
            rows = [row for row in rows if row["status"] in statuses]
        if job_id_missing is True:
            rows = [row for row in rows if row["job_id"] is None]
        elif job_id_missing is False:
            rows = [row for row in rows if row["job_id"] is not None]
        if updated_before is not None:
            rows = [row for row in rows if row["updated_at"] < updated_before]
        if export_id_after is not None:
            rows = [row for row in rows if row["export_id"] > export_id_after]
        if export_id_at_or_before is not None:
            rows = [row for row in rows if row["export_id"] <= export_id_at_or_before]
        if export_id_after is not None or export_id_at_or_before is not None:
            rows.sort(key=lambda row: row["export_id"])
        else:
            rows.sort(key=lambda row: (row["updated_at"], row["export_id"]))
        return [dict(row) for row in rows[:limit]]

    def attach_claims_analytics_export_job(self, *, export_id: str, user_id: str, job_id: int) -> bool:
        self.attach_calls.append({"export_id": export_id, "user_id": user_id, "job_id": job_id})
        row = self.rows.get(export_id)
        if row is None or row["user_id"] != user_id or row["job_id"] not in (None, job_id):
            return False
        row["job_id"] = job_id
        return True

    def transition_claims_analytics_export_status(self, **values: Any) -> bool:
        self.transition_calls.append(dict(values))
        row = self.rows.get(values["export_id"])
        if row is None or row["user_id"] != values["user_id"] or row["status"] not in values["from_statuses"]:
            return False
        row["status"] = values["to_status"]
        row["error_code"] = values.get("error_code")
        row["error_message"] = values.get("error_message")
        return True

    def get_claims_analytics_export(self, export_id: str, *, user_id: str) -> dict[str, Any]:
        row = self.rows.get(export_id)
        return dict(row) if row is not None and row["user_id"] == user_id else {}

    def delete_claims_analytics_exports(self, *, user_id: str, export_ids: list[str], updated_before: str) -> int:
        self.delete_calls.append(
            {
                "user_id": user_id,
                "export_ids": list(export_ids),
                "updated_before": updated_before,
            }
        )
        if self.before_delete is not None:
            self.before_delete(self.rows)
        deleted = 0
        for export_id in export_ids:
            row = self.rows.get(export_id)
            if row is not None and row["user_id"] == user_id and row["updated_at"] < updated_before:
                del self.rows[export_id]
                deleted += 1
        return deleted


class FakeJobManager:
    def __init__(
        self,
        *,
        jobs_by_id: dict[int, dict[str, Any]] | None = None,
        groups: dict[str, dict[str, Any] | None | BaseException] | None = None,
        batch_error: BaseException | None = None,
    ) -> None:
        self.jobs_by_id = {
            job_id: {
                "id": job_id,
                "domain": "claims",
                "owner_user_id": "7",
                "job_type": "claims_generate_analytics_export",
                **job,
            }
            for job_id, job in (jobs_by_id or {}).items()
        }
        self.groups = groups or {}
        self.batch_error = batch_error
        self.batch_calls: list[dict[str, Any]] = []
        self.group_calls: list[dict[str, Any]] = []

    def get_jobs_by_ids(self, job_ids: list[int], **scope: Any) -> dict[int, dict[str, Any]]:
        self.batch_calls.append({"job_ids": list(job_ids), **scope})
        if self.batch_error is not None:
            raise self.batch_error
        return {job_id: self.jobs_by_id[job_id] for job_id in job_ids if job_id in self.jobs_by_id}

    def find_job_by_batch_group(self, **query: Any) -> dict[str, Any] | None:
        self.group_calls.append(dict(query))
        result = self.groups.get(query["batch_group"])
        if isinstance(result, BaseException):
            raise result
        return result


def _exact_job(export_id: str, job_id: int, *, archived: bool = False) -> dict[str, Any]:
    return {
        "id": job_id,
        "domain": "claims",
        "owner_user_id": "7",
        "job_type": "claims_generate_analytics_export",
        "batch_group": f"claims-analytics-export:{export_id}",
        "archived": archived,
    }


def test_hydrate_job_statuses_batches_unique_valid_ids_once_without_payload_access() -> None:
    rows = [
        {"job_id": 4},
        {"job_id": 4},
        {"job_id": 7},
        {"job_id": None},
        {"job_id": True},
        {"job_id": 0},
    ]
    manager = FakeJobManager(
        jobs_by_id={
            4: {"id": 4, "status": "processing", "result": object()},
        }
    )

    statuses = hydrate_job_statuses(rows, owner_user_id="7", job_manager=manager)

    assert statuses == {4: "processing", 7: None}
    assert manager.batch_calls == [
        {
            "job_ids": [4, 7],
            "domain": "claims",
            "owner_user_id": "7",
            "include_archived": True,
        }
    ]
    assert rows == [
        {"job_id": 4},
        {"job_id": 4},
        {"job_id": 7},
        {"job_id": None},
        {"job_id": True},
        {"job_id": 0},
    ]


def test_hydrate_job_statuses_jobs_exception_returns_nulls_without_mutation() -> None:
    rows = [{"job_id": 9, "status": "ready"}, {"job_id": 10, "status": "failed"}]
    before = [dict(row) for row in rows]
    manager = FakeJobManager(batch_error=RuntimeError("jobs unavailable with secret"))

    assert hydrate_job_statuses(rows, owner_user_id="7", job_manager=manager) == {
        9: None,
        10: None,
    }
    assert rows == before
    assert len(manager.batch_calls) == 1


def test_reconcile_repairs_exact_active_and_archived_jobs_before_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(exports.settings, "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC", 300)
    active = _artifact(1, created_seconds_ago=1)
    archived = _artifact(2, created_seconds_ago=299)
    db = MaintenanceDB([active, archived])
    manager = FakeJobManager(
        groups={
            f"claims-analytics-export:{active['export_id']}": _exact_job(active["export_id"], 41),
            f"claims-analytics-export:{archived['export_id']}": _exact_job(archived["export_id"], 42, archived=True),
        }
    )

    result = reconcile_export_artifacts(
        db,
        owner_user_id="7",
        job_manager=manager,
        now=NOW,
        limit=500,
    )

    assert result == {"examined": 2, "repaired": 2, "failed": 0, "unchanged": 0}
    assert db.list_calls == [
        {
            "user_id": "7",
            "limit": 100,
            "statuses": ("queued",),
            "job_id_missing": True,
            "updated_before": None,
            "export_id_after": None,
            "export_id_at_or_before": None,
        }
    ]
    assert {db.rows[active["export_id"]]["job_id"], db.rows[archived["export_id"]]["job_id"]} == {41, 42}
    assert all(
        call
        == {
            "batch_group": f"claims-analytics-export:{call['batch_group'].split(':', 1)[1]}",
            "domain": "claims",
            "owner_user_id": "7",
            "job_type": "claims_generate_analytics_export",
            "include_archived": True,
        }
        for call in manager.group_calls
    )


@pytest.mark.parametrize("age_seconds", [299.999, 300])
def test_reconcile_marks_proven_orphan_only_when_grace_has_elapsed(
    monkeypatch: pytest.MonkeyPatch,
    age_seconds: float,
) -> None:
    monkeypatch.setitem(exports.settings, "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC", 300)
    row = _artifact(3, created_seconds_ago=age_seconds)
    db = MaintenanceDB([row])

    result = reconcile_export_artifacts(
        db,
        owner_user_id="7",
        job_manager=FakeJobManager(),
        now=NOW,
    )

    expected_failed = int(age_seconds >= 300)
    assert result == {
        "examined": 1,
        "repaired": 0,
        "failed": expected_failed,
        "unchanged": 1 - expected_failed,
    }
    stored = db.rows[row["export_id"]]
    assert stored["status"] == ("failed" if expected_failed else "queued")
    assert stored.get("error_code") == ("claims_export_enqueue_failed" if expected_failed else None)


@pytest.mark.parametrize(
    "returned_job",
    [
        lambda export_id: {
            **_exact_job(export_id, 51),
            "batch_group": f"claims-analytics-export:{export_id}-prefix",
        },
        lambda export_id: {**_exact_job(export_id, 52), "owner_user_id": "8"},
        lambda export_id: {**_exact_job(export_id, 53), "domain": "other"},
        lambda export_id: {**_exact_job(export_id, 54), "id": True},
    ],
)
def test_reconcile_ignores_nonexact_or_wrong_scope_matches(returned_job: Any) -> None:
    row = _artifact(4, created_seconds_ago=600)
    group = f"claims-analytics-export:{row['export_id']}"
    db = MaintenanceDB([row])
    manager = FakeJobManager(groups={group: returned_job(row["export_id"])})

    result = reconcile_export_artifacts(db, owner_user_id="7", job_manager=manager, now=NOW)

    assert result == {"examined": 1, "repaired": 0, "failed": 0, "unchanged": 1}
    assert db.rows[row["export_id"]]["status"] == "queued"
    assert db.rows[row["export_id"]]["job_id"] is None


def test_reconcile_rejects_type_only_job_row_shape() -> None:
    row = _artifact(7, created_seconds_ago=1)
    group = f"claims-analytics-export:{row['export_id']}"
    type_only = _exact_job(row["export_id"], 55)
    type_only["type"] = type_only.pop("job_type")
    db = MaintenanceDB([row])

    result = reconcile_export_artifacts(
        db,
        owner_user_id="7",
        job_manager=FakeJobManager(groups={group: type_only}),
        now=NOW,
    )

    assert result == {"examined": 1, "repaired": 0, "failed": 0, "unchanged": 1}
    assert db.rows[row["export_id"]]["job_id"] is None


def test_reconcile_jobs_outage_preserves_artifact_and_zero_grace_is_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(exports.settings, "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC", 0)
    outage = _artifact(5, created_seconds_ago=600)
    proven = _artifact(6, created_seconds_ago=0)
    db = MaintenanceDB([outage, proven])
    manager = FakeJobManager(groups={f"claims-analytics-export:{outage['export_id']}": RuntimeError("jobs down")})

    result = reconcile_export_artifacts(db, owner_user_id="7", job_manager=manager, now=NOW)

    assert result == {"examined": 2, "repaired": 0, "failed": 1, "unchanged": 1}
    assert db.rows[outage["export_id"]]["status"] == "queued"
    assert db.rows[proven["export_id"]]["status"] == "failed"


def test_reconcile_candidate_filters_prevent_attached_rows_from_consuming_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(exports.settings, "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC", 300)
    irrelevant = [
        _artifact(
            60 + index,
            status="queued",
            job_id=160 + index,
            created_seconds_ago=7200,
            updated_seconds_ago=7200,
        )
        for index in range(3)
    ]
    orphan = _artifact(
        63,
        status="queued",
        created_seconds_ago=600,
        updated_seconds_ago=600,
    )
    db = MaintenanceDB([*irrelevant, orphan])

    result = reconcile_export_artifacts(
        db,
        owner_user_id="7",
        job_manager=FakeJobManager(),
        now=NOW,
        limit=2,
    )

    assert result == {"examined": 1, "repaired": 0, "failed": 1, "unchanged": 0}
    assert db.rows[orphan["export_id"]]["status"] == "failed"
    assert all(db.rows[row["export_id"]]["status"] == "queued" for row in irrelevant)


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled", "quarantined"])
def test_cleanup_deletes_old_ready_and_terminal_failed_rows(terminal: str) -> None:
    ready = _artifact(10, status="ready", updated_seconds_ago=3601)
    failed = _artifact(11, status="failed", job_id=81, updated_seconds_ago=3601)
    db = MaintenanceDB([ready, failed])
    manager = FakeJobManager(jobs_by_id={81: {"id": 81, "status": terminal}})

    deleted = cleanup_export_artifacts(
        db,
        owner_user_id="7",
        job_manager=manager,
        now=NOW,
        retention_hours=1,
    )

    cutoff = _timestamp(3600)
    assert deleted == 2
    assert db.rows == {}
    assert manager.batch_calls == [
        {
            "job_ids": [81],
            "domain": "claims",
            "owner_user_id": "7",
            "include_archived": True,
        }
    ]
    assert db.delete_calls == [
        {
            "user_id": "7",
            "export_ids": [ready["export_id"], failed["export_id"]],
            "updated_before": cutoff,
        }
    ]
    anchor = exports._cleanup_rotation_anchor(owner_user_id="7", now=NOW)
    assert db.list_calls == [
        {
            "user_id": "7",
            "limit": 50,
            "statuses": ("ready",),
            "job_id_missing": None,
            "updated_before": cutoff,
            "export_id_after": None,
            "export_id_at_or_before": None,
        },
        {
            "user_id": "7",
            "limit": 50,
            "statuses": ("failed",),
            "job_id_missing": None,
            "updated_before": cutoff,
            "export_id_after": anchor,
            "export_id_at_or_before": None,
        },
        {
            "user_id": "7",
            "limit": 50,
            "statuses": ("failed",),
            "job_id_missing": None,
            "updated_before": cutoff,
            "export_id_after": None,
            "export_id_at_or_before": anchor,
        },
    ]


def test_cleanup_candidate_filters_prevent_active_rows_from_consuming_limit() -> None:
    irrelevant = [
        _artifact(
            70 + index,
            status="queued" if index % 2 == 0 else "processing",
            updated_seconds_ago=7200,
        )
        for index in range(3)
    ]
    ready = _artifact(73, status="ready", updated_seconds_ago=3601)
    db = MaintenanceDB([*irrelevant, ready])

    deleted = cleanup_export_artifacts(
        db,
        owner_user_id="7",
        job_manager=FakeJobManager(),
        now=NOW,
        retention_hours=1,
        limit=2,
    )

    assert deleted == 1
    assert ready["export_id"] not in db.rows
    assert all(row["export_id"] in db.rows for row in irrelevant)


def test_cleanup_rotates_failed_page_past_older_uncertain_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = f"{50:032x}"
    monkeypatch.setattr(exports, "_cleanup_rotation_anchor", lambda **_: anchor)
    uncertain = [
        _artifact(
            index,
            status="failed",
            job_id=200 + index,
            updated_seconds_ago=7200,
        )
        for index in (1, 2, 3)
    ]
    terminal = _artifact(
        100,
        status="failed",
        job_id=300,
        updated_seconds_ago=3601,
    )
    db = MaintenanceDB([*uncertain, terminal])
    manager = FakeJobManager(
        jobs_by_id={
            **{row["job_id"]: {"status": "processing"} for row in uncertain},
            300: {"status": "completed"},
        }
    )

    deleted = cleanup_export_artifacts(
        db,
        owner_user_id="7",
        job_manager=manager,
        now=NOW,
        retention_hours=1,
        limit=4,
    )

    assert deleted == 1
    assert terminal["export_id"] not in db.rows
    assert all(row["export_id"] in db.rows for row in uncertain)
    assert manager.batch_calls[0]["job_ids"] == [300, 201]
    failed_calls = [call for call in db.list_calls if call["statuses"] == ("failed",)]
    assert [call["export_id_after"] for call in failed_calls] == [anchor, None]
    assert [call["export_id_at_or_before"] for call in failed_calls] == [None, anchor]


def test_cleanup_preserves_cutoff_active_and_status_uncertainty() -> None:
    rows = [
        _artifact(20, status="ready", updated_seconds_ago=3600),
        _artifact(21, status="queued", updated_seconds_ago=7200),
        _artifact(22, status="processing", updated_seconds_ago=7200),
        _artifact(23, status="failed", job_id=91, updated_seconds_ago=7200),
        _artifact(24, status="failed", job_id=92, updated_seconds_ago=7200),
        _artifact(25, status="failed", job_id=93, updated_seconds_ago=7200),
    ]
    db = MaintenanceDB(rows)
    manager = FakeJobManager(
        jobs_by_id={
            91: {"id": 91, "status": "queued"},
            92: {"id": 92, "status": "processing"},
            93: {"id": 93, "status": "retrying"},
        }
    )

    assert (
        cleanup_export_artifacts(
            db,
            owner_user_id="7",
            job_manager=manager,
            now=NOW,
            retention_hours=1,
        )
        == 0
    )
    assert set(db.rows) == {row["export_id"] for row in rows}
    assert len(manager.batch_calls) == 1
    assert db.delete_calls == []


def test_cleanup_requires_successful_absence_and_grace_for_pruned_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(exports.settings, "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC", 300)
    elapsed = _artifact(30, status="failed", job_id=101, updated_seconds_ago=601)
    exact_grace = _artifact(31, status="failed", job_id=102, updated_seconds_ago=300)
    db = MaintenanceDB([elapsed, exact_grace])

    deleted = cleanup_export_artifacts(
        db,
        owner_user_id="7",
        job_manager=FakeJobManager(),
        now=NOW,
        retention_hours=1 / 12,
    )

    assert deleted == 1
    assert elapsed["export_id"] not in db.rows
    assert exact_grace["export_id"] in db.rows


def test_cleanup_reconciled_failed_without_job_waits_retention_plus_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(exports.settings, "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC", 300)
    old = _artifact(
        40,
        status="failed",
        updated_seconds_ago=3901,
        error_code="claims_export_enqueue_failed",
    )
    exact = _artifact(
        41,
        status="failed",
        updated_seconds_ago=3900,
        error_code="claims_export_enqueue_failed",
    )
    db = MaintenanceDB([old, exact])
    manager = FakeJobManager()

    deleted = cleanup_export_artifacts(
        db,
        owner_user_id="7",
        job_manager=manager,
        now=NOW,
        retention_hours=1,
    )

    assert deleted == 1
    assert old["export_id"] not in db.rows
    assert exact["export_id"] in db.rows
    assert manager.batch_calls == []
    assert manager.group_calls == [
        {
            "batch_group": f"claims-analytics-export:{old['export_id']}",
            "domain": "claims",
            "owner_user_id": "7",
            "job_type": "claims_generate_analytics_export",
            "include_archived": True,
        }
    ]


@pytest.mark.parametrize("archived", [False, True])
def test_cleanup_preserves_enqueue_failed_without_job_when_exact_job_exists(
    monkeypatch: pytest.MonkeyPatch,
    archived: bool,
) -> None:
    monkeypatch.setitem(exports.settings, "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC", 300)
    row = _artifact(
        43,
        status="failed",
        updated_seconds_ago=7200,
        error_code="claims_export_enqueue_failed",
    )
    batch_group = f"claims-analytics-export:{row['export_id']}"
    db = MaintenanceDB([row])
    manager = FakeJobManager(groups={batch_group: _exact_job(row["export_id"], 143, archived=archived)})

    deleted = cleanup_export_artifacts(
        db,
        owner_user_id="7",
        job_manager=manager,
        now=NOW,
        retention_hours=1,
    )

    assert deleted == 0
    assert row["export_id"] in db.rows
    assert len(manager.group_calls) == 1
    assert manager.group_calls[0]["include_archived"] is True
    assert db.delete_calls == []


def test_cleanup_preserves_enqueue_failed_without_job_on_malformed_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(exports.settings, "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC", 300)
    row = _artifact(
        44,
        status="failed",
        updated_seconds_ago=7200,
        error_code="claims_export_enqueue_failed",
    )
    batch_group = f"claims-analytics-export:{row['export_id']}"
    db = MaintenanceDB([row])
    manager = FakeJobManager(groups={batch_group: {**_exact_job(row["export_id"], 144), "owner_user_id": "8"}})

    assert (
        cleanup_export_artifacts(
            db,
            owner_user_id="7",
            job_manager=manager,
            now=NOW,
            retention_hours=1,
        )
        == 0
    )
    assert row["export_id"] in db.rows
    assert len(manager.group_calls) == 1
    assert db.delete_calls == []


def test_cleanup_preserves_enqueue_failed_without_job_during_jobs_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(exports.settings, "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC", 300)
    row = _artifact(
        45,
        status="failed",
        updated_seconds_ago=7200,
        error_code="claims_export_enqueue_failed",
    )
    batch_group = f"claims-analytics-export:{row['export_id']}"
    db = MaintenanceDB([row])
    manager = FakeJobManager(groups={batch_group: RuntimeError("jobs outage secret")})

    assert (
        cleanup_export_artifacts(
            db,
            owner_user_id="7",
            job_manager=manager,
            now=NOW,
            retention_hours=1,
        )
        == 0
    )
    assert row["export_id"] in db.rows
    assert len(manager.group_calls) == 1
    assert db.delete_calls == []


@pytest.mark.parametrize(
    "error_code",
    [None, "claims_export_serialization_failed", "claims_export_storage_unavailable"],
)
def test_cleanup_preserves_unrelated_failed_without_job_after_retention_and_grace(
    monkeypatch: pytest.MonkeyPatch,
    error_code: str | None,
) -> None:
    monkeypatch.setitem(exports.settings, "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC", 300)
    unrelated = _artifact(
        42,
        status="failed",
        updated_seconds_ago=7200,
        error_code=error_code,
    )
    db = MaintenanceDB([unrelated])
    manager = FakeJobManager(batch_error=RuntimeError("Jobs should not be called"))

    deleted = cleanup_export_artifacts(
        db,
        owner_user_id="7",
        job_manager=manager,
        now=NOW,
        retention_hours=1,
    )

    assert deleted == 0
    assert unrelated["export_id"] in db.rows
    assert manager.batch_calls == []
    assert manager.group_calls == []
    assert db.delete_calls == []


def test_cleanup_jobs_outage_deletes_ready_but_preserves_uncertain_failed() -> None:
    ready = _artifact(50, status="ready", updated_seconds_ago=7200)
    failed = _artifact(51, status="failed", job_id=111, updated_seconds_ago=7200)
    db = MaintenanceDB([ready, failed])
    manager = FakeJobManager(batch_error=RuntimeError("jobs outage secret"))

    deleted = cleanup_export_artifacts(
        db,
        owner_user_id="7",
        job_manager=manager,
        now=NOW,
        retention_hours=1,
    )

    assert deleted == 1
    assert ready["export_id"] not in db.rows
    assert failed["export_id"] in db.rows
    assert len(manager.batch_calls) == 1


def test_cleanup_ignores_terminal_job_row_outside_requested_scope() -> None:
    failed = _artifact(52, status="failed", job_id=112, updated_seconds_ago=7200)
    db = MaintenanceDB([failed])
    manager = FakeJobManager(
        jobs_by_id={
            112: {
                "id": 112,
                "status": "completed",
                "domain": "claims",
                "owner_user_id": "8",
                "job_type": "claims_generate_analytics_export",
            }
        }
    )

    assert (
        cleanup_export_artifacts(
            db,
            owner_user_id="7",
            job_manager=manager,
            now=NOW,
            retention_hours=1,
        )
        == 0
    )
    assert failed["export_id"] in db.rows


def test_cleanup_rejects_type_only_terminal_job_row_shape() -> None:
    failed = _artifact(53, status="failed", job_id=113, updated_seconds_ago=7200)
    db = MaintenanceDB([failed])
    type_only = {
        "id": 113,
        "status": "completed",
        "domain": "claims",
        "owner_user_id": "7",
        "job_type": None,
        "type": "claims_generate_analytics_export",
    }

    assert (
        cleanup_export_artifacts(
            db,
            owner_user_id="7",
            job_manager=FakeJobManager(jobs_by_id={113: type_only}),
            now=NOW,
            retention_hours=1,
        )
        == 0
    )
    assert failed["export_id"] in db.rows


def test_cleanup_delete_cutoff_prevents_status_blind_race_deletion() -> None:
    failed = _artifact(60, status="failed", job_id=121, updated_seconds_ago=7200)
    db = MaintenanceDB([failed])
    manager = FakeJobManager(jobs_by_id={121: {"id": 121, "status": "completed"}})

    def retry_started(rows: dict[str, dict[str, Any]]) -> None:
        rows[failed["export_id"]].update(status="processing", updated_at=_timestamp(0))

    db.before_delete = retry_started

    assert (
        cleanup_export_artifacts(
            db,
            owner_user_id="7",
            job_manager=manager,
            now=NOW,
            retention_hours=1,
        )
        == 0
    )
    assert db.rows[failed["export_id"]]["status"] == "processing"
