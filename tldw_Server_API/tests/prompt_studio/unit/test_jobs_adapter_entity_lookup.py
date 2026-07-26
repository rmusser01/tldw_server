from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.jobs_adapter import (
    PromptStudioJobsAdapter,
)

pytestmark = pytest.mark.unit


def _job(
    job_id: int,
    *,
    entity_id: int,
    owner: str | None = "alice",
    domain: str = "prompt_studio",
    job_type: str = "optimization",
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": job_id,
        "uuid": f"job-{job_id}",
        "domain": domain,
        "job_type": job_type,
        "owner_user_id": owner,
        "status": status,
        "payload": {"optimization_id": entity_id, "entity_id": entity_id},
        "created_at": datetime(2026, 7, 16, tzinfo=timezone.utc)
        + timedelta(seconds=job_id),
    }


class _CursorJobManager:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        honor_filters: bool = True,
        inject_after_first_page: dict[str, Any] | None = None,
        fail_on_call: int | None = None,
    ) -> None:
        self.rows = list(rows)
        self.honor_filters = honor_filters
        self.inject_after_first_page = inject_after_first_page
        self.fail_on_call = fail_on_call
        self.calls: list[dict[str, Any]] = []

    def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(dict(kwargs))
        if self.fail_on_call == len(self.calls):
            raise ConnectionError("jobs lookup unavailable")

        rows = list(self.rows)
        if self.honor_filters:
            for key in ("domain", "owner_user_id", "job_type"):
                requested = kwargs.get(key)
                if requested is not None:
                    rows = [row for row in rows if str(row.get(key)) == str(requested)]

        created_before = kwargs.get("created_before")
        before_id = kwargs.get("before_id")
        if created_before is not None:
            rows = [
                row
                for row in rows
                if row["created_at"] < created_before
                or (
                    row["created_at"] == created_before
                    and int(row["id"]) < int(before_id)
                )
            ]

        rows.sort(key=lambda row: (row["created_at"], int(row["id"])), reverse=True)
        page = rows[: int(kwargs["limit"])]
        if len(self.calls) == 1 and self.inject_after_first_page is not None:
            self.rows.append(self.inject_after_first_page)
        return [dict(row) for row in page]


def _adapter(job_manager: _CursorJobManager) -> PromptStudioJobsAdapter:
    adapter = object.__new__(PromptStudioJobsAdapter)
    adapter._backend = "core"
    adapter._jm = job_manager
    return adapter


def test_latest_entity_job_pages_past_newer_jobs_with_a_stable_cursor() -> None:
    target = _job(1, entity_id=42, status="processing")
    newer_jobs = [_job(job_id, entity_id=1000 + job_id) for job_id in range(2, 103)]
    manager = _CursorJobManager(
        [target, *newer_jobs],
        inject_after_first_page=_job(103, entity_id=1103),
    )

    found = _adapter(manager).get_latest_job_for_entity(
        db=object(),
        user_id="alice",
        job_type="optimization",
        entity_id=42,
    )

    assert found is not None
    assert found["id"] == "job-1"
    assert found["status"] == "processing"
    assert len(manager.calls) == 2
    assert manager.calls[1]["before_id"] == 3
    assert manager.calls[1]["created_before"] == newer_jobs[1]["created_at"]


def test_latest_entity_job_revalidates_exact_tenant_domain_and_job_type() -> None:
    manager = _CursorJobManager(
        [
            _job(1, entity_id=42),
            _job(2, entity_id=42, owner="bob"),
            _job(3, entity_id=42, job_type="evaluation"),
            _job(4, entity_id=42, domain="other"),
            _job(5, entity_id=42, owner=None),
        ],
        honor_filters=False,
    )

    found = _adapter(manager).get_latest_job_for_entity(
        db=object(),
        user_id="alice",
        job_type="optimization",
        entity_id=42,
    )

    assert found is not None
    assert found["id"] == "job-1"
    assert manager.calls[0]["domain"] == "prompt_studio"
    assert manager.calls[0]["owner_user_id"] == "alice"
    assert manager.calls[0]["job_type"] == "optimization"


def test_list_jobs_revalidates_exact_tenant_domain_and_job_type() -> None:
    manager = _CursorJobManager(
        [
            _job(1, entity_id=42),
            _job(2, entity_id=42, owner="bob"),
            _job(3, entity_id=42, job_type="evaluation"),
            _job(4, entity_id=42, domain="other"),
            _job(5, entity_id=42, owner=None),
        ],
        honor_filters=False,
    )

    jobs = _adapter(manager).list_jobs(
        db=object(),
        user_id="alice",
        job_type="optimization",
    )

    assert [job["id"] for job in jobs] == ["job-1"]
    assert manager.calls[0]["domain"] == "prompt_studio"
    assert manager.calls[0]["owner_user_id"] == "alice"
    assert manager.calls[0]["job_type"] == "optimization"


def test_latest_entity_job_does_not_report_absence_after_a_partial_lookup() -> None:
    target = _job(1, entity_id=42, status="processing")
    newer_jobs = [_job(job_id, entity_id=1000 + job_id) for job_id in range(2, 102)]
    manager = _CursorJobManager(
        [target, *newer_jobs],
        fail_on_call=2,
    )

    with pytest.raises(ConnectionError, match="jobs lookup unavailable"):
        _adapter(manager).get_latest_job_for_entity(
            db=object(),
            user_id="alice",
            job_type="optimization",
            entity_id=42,
        )


@pytest.mark.parametrize("user_id", [None, "", "   "])
def test_public_adapter_methods_reject_missing_tenant_without_querying(
    user_id: str | None,
) -> None:
    class _MustNotQuery:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"missing tenant must not reach JobManager.{name}")

    adapter = _adapter(_MustNotQuery())  # type: ignore[arg-type]
    calls = [
        lambda: adapter.get_job(
            "job-1",
            db=object(),
            user_id=user_id,
            job_type="optimization",
        ),
        lambda: adapter.list_jobs(
            db=object(),
            user_id=user_id,
            job_type="optimization",
        ),
        lambda: adapter.get_latest_job_for_entity(
            db=object(),
            user_id=user_id,
            job_type="optimization",
            entity_id=42,
        ),
        lambda: adapter.list_jobs_for_entity(
            db=object(),
            user_id=user_id,
            job_type="optimization",
            entity_id=42,
        ),
        lambda: adapter.create_job(
            user_id=user_id,
            job_type="optimization",
            entity_id=42,
            payload={},
        ),
        lambda: adapter.cancel_job(
            "job-1",
            user_id=user_id,
            job_type="optimization",
        ),
    ]

    for call in calls:
        with pytest.raises(ValueError, match="owner is required"):
            call()


@pytest.mark.parametrize("optimization_uuid", [None, "", "   ", 17])
def test_create_optimization_job_rejects_invalid_row_identity_before_enqueue(
    optimization_uuid: object,
) -> None:
    class _MustNotCreate:
        def create_job(self, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("invalid optimization identity reached core Jobs")

    payload: dict[str, Any] = {"optimization_id": 42}
    if optimization_uuid is not None:
        payload["optimization_uuid"] = optimization_uuid

    with pytest.raises(ValueError, match="optimization_uuid"):
        _adapter(_MustNotCreate()).create_job(  # type: ignore[arg-type]
            user_id="alice",
            job_type="optimization",
            entity_id=42,
            payload=payload,
        )


def test_create_optimization_job_preserves_valid_row_identity() -> None:
    created: list[dict[str, Any]] = []

    class _RecordingManager:
        def create_job(self, **kwargs: Any) -> dict[str, Any]:
            created.append(dict(kwargs))
            return {"id": 1, "uuid": "job-1", **kwargs}

    result = _adapter(_RecordingManager()).create_job(  # type: ignore[arg-type]
        user_id="alice",
        job_type="optimization",
        entity_id=42,
        payload={
            "optimization_id": 42,
            "optimization_uuid": "  optimization-42  ",
        },
    )

    assert result["payload"]["optimization_uuid"] == "optimization-42"
    assert created[0]["payload"]["optimization_uuid"] == "optimization-42"


def test_cancel_job_does_not_cancel_reused_id_after_lookup_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager(tmp_path / "cancel-id-reuse.sqlite")
    stale_job = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload={"optimization_id": 42},
        owner_user_id="alice",
    )
    original_cancel = manager.cancel_job
    replacement: dict[str, Any] = {}

    def _replace_before_cancel(job_id: int, **kwargs: Any) -> bool:
        conn = sqlite3.connect(manager.db_path)
        try:
            conn.execute("DELETE FROM jobs WHERE id = ?", (int(job_id),))
            conn.commit()
        finally:
            conn.close()
        replacement.update(
            manager.create_job(
                domain="prompt_studio",
                queue="default",
                job_type="optimization",
                payload={"optimization_id": 99},
                owner_user_id="alice",
            )
        )
        assert int(replacement["id"]) == int(stale_job["id"])
        return original_cancel(job_id, **kwargs)

    monkeypatch.setattr(manager, "cancel_job", _replace_before_cancel)
    adapter = _adapter(manager)  # type: ignore[arg-type]

    assert adapter.cancel_job(
        str(stale_job["uuid"]),
        user_id="alice",
        reason="stale request",
        job_type="optimization",
    ) is False
    assert manager.get_job(int(replacement["id"]))["status"] == "queued"
