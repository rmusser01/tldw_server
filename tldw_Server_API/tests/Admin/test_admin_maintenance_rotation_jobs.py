from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


@pytest.fixture()
async def maintenance_rotation_repo(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.maintenance_rotation_runs_repo import (
        AuthnzMaintenanceRotationRunsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users_maintenance_rotation.db"
    jobs_db_path = tmp_path / "jobs_maintenance_rotation.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = AuthnzMaintenanceRotationRunsRepo(pool)
    await repo.ensure_schema()
    return repo


async def _create_queued_run(repo, *, mode: str):
    run = await repo.create_run(
        mode=mode,
        domain="jobs",
        queue="default",
        job_type="encryption_rotation",
        fields_json='["payload","result"]',
        limit=125,
        requested_by_user_id=5,
        requested_by_label="ops-admin@example.com",
        confirmation_recorded=(mode == "execute"),
        scope_summary="domain=jobs, queue=default, job_type=encryption_rotation, fields=payload,result, limit=125",
        key_source="env:jobs_crypto_rotate",
    )
    return run, {"id": f"job-{mode}", "payload": {"run_id": str(run["id"])}}  # type: ignore[dict-item]


@pytest.mark.asyncio
async def test_handle_maintenance_rotation_job_completes_dry_run(
    maintenance_rotation_repo,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.admin_maintenance_rotation_jobs_worker import (
        handle_maintenance_rotation_job,
    )

    monkeypatch.setenv("JOBS_CRYPTO_ROTATE_OLD_KEY", "old-key-material")
    monkeypatch.setenv("JOBS_CRYPTO_ROTATE_NEW_KEY", "new-key-material")

    run, job = await _create_queued_run(maintenance_rotation_repo, mode="dry_run")
    captured: dict[str, object] = {}

    def _fake_rotate_encryption_keys(**kwargs):
        captured.update(kwargs)
        return 17

    result = await handle_maintenance_rotation_job(
        job,
        repo=maintenance_rotation_repo,
        rotate_encryption_keys_fn=_fake_rotate_encryption_keys,
    )

    assert result["status"] == "complete"
    assert result["run_id"] == str(run["id"])
    assert captured["domain"] == "jobs"
    assert captured["queue"] == "default"
    assert captured["job_type"] == "encryption_rotation"
    assert captured["old_key_b64"] == "old-key-material"
    assert captured["new_key_b64"] == "new-key-material"
    assert captured["fields"] == ["payload", "result"]
    assert captured["limit"] == 125
    assert captured["dry_run"] is True

    stored = await maintenance_rotation_repo.get_run(str(run["id"]))
    assert stored is not None
    assert stored["status"] == "complete"
    assert stored["job_id"] == "job-dry_run"
    assert stored["affected_count"] == 17
    assert stored["started_at"] is not None
    assert stored["completed_at"] is not None


@pytest.mark.asyncio
async def test_handle_maintenance_rotation_job_completes_execute_mode(
    maintenance_rotation_repo,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.admin_maintenance_rotation_jobs_worker import (
        handle_maintenance_rotation_job,
    )

    monkeypatch.setenv("JOBS_CRYPTO_ROTATE_OLD_KEY", "old-key-material")
    monkeypatch.setenv("JOBS_CRYPTO_ROTATE_NEW_KEY", "new-key-material")

    run, job = await _create_queued_run(maintenance_rotation_repo, mode="execute")
    captured: dict[str, object] = {}

    def _fake_rotate_encryption_keys(**kwargs):
        captured.update(kwargs)
        return 9

    result = await handle_maintenance_rotation_job(
        job,
        repo=maintenance_rotation_repo,
        rotate_encryption_keys_fn=_fake_rotate_encryption_keys,
    )

    assert result["status"] == "complete"
    assert captured["dry_run"] is False

    stored = await maintenance_rotation_repo.get_run(str(run["id"]))
    assert stored is not None
    assert stored["status"] == "complete"
    assert stored["affected_count"] == 9


@pytest.mark.asyncio
async def test_handle_maintenance_rotation_job_marks_failure(
    maintenance_rotation_repo,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services.admin_maintenance_rotation_jobs_worker import (
        handle_maintenance_rotation_job,
    )

    monkeypatch.setenv("JOBS_CRYPTO_ROTATE_OLD_KEY", "old-key-material")
    monkeypatch.setenv("JOBS_CRYPTO_ROTATE_NEW_KEY", "new-key-material")

    run, job = await _create_queued_run(maintenance_rotation_repo, mode="execute")

    def _failing_rotate_encryption_keys(**kwargs):
        raise RuntimeError("rotation exploded")

    with pytest.raises(RuntimeError, match="rotation exploded"):
        await handle_maintenance_rotation_job(
            job,
            repo=maintenance_rotation_repo,
            rotate_encryption_keys_fn=_failing_rotate_encryption_keys,
        )

    stored = await maintenance_rotation_repo.get_run(str(run["id"]))
    assert stored is not None
    assert stored["status"] == "failed"
    assert stored["job_id"] == "job-execute"
    assert stored["error_message"] == "rotation exploded"
    assert stored["completed_at"] is not None


@pytest.mark.asyncio
async def test_run_admin_maintenance_rotation_jobs_worker_stops_sdk_when_stop_event_is_set(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services import admin_maintenance_rotation_jobs_worker as worker

    stop_event = asyncio.Event()
    observed = {"stopped": False}

    class _FakeWorkerSDK:
        def __init__(self, _jm, _cfg) -> None:
            pass

        def stop(self) -> None:
            observed["stopped"] = True

        async def run(self, *, handler) -> None:
            while not observed["stopped"]:
                await asyncio.sleep(0)

    monkeypatch.setattr(worker, "WorkerSDK", _FakeWorkerSDK)

    task = asyncio.create_task(worker.run_admin_maintenance_rotation_jobs_worker(stop_event))
    await asyncio.sleep(0)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1)

    assert observed["stopped"] is True


@pytest.mark.asyncio
async def test_start_admin_maintenance_rotation_jobs_worker_forwards_caller_owned_stop_event(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services import admin_maintenance_rotation_jobs_worker as worker

    observed: dict[str, object] = {}
    stop_event = asyncio.Event()

    async def _fake_run(stop_event_arg: asyncio.Event | None = None) -> None:
        observed["stop_event"] = stop_event_arg
        await stop_event_arg.wait()

    monkeypatch.setenv("ADMIN_MAINTENANCE_ROTATION_JOBS_WORKER_ENABLED", "1")
    monkeypatch.setattr(worker, "run_admin_maintenance_rotation_jobs_worker", _fake_run)

    task = await worker.start_admin_maintenance_rotation_jobs_worker(stop_event=stop_event)
    assert task is not None
    await asyncio.sleep(0)

    assert observed["stop_event"] is stop_event

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)
