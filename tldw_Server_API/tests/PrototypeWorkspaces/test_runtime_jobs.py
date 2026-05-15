"""Task 4 coverage for prototype runtime job orchestration."""
from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _load_attr(module: Any, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"Module {module.__name__} does not define any of: {', '.join(names)}")


@pytest.fixture
def prototype_jobs(repo):
    module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.jobs")
    jobs_cls = _load_attr(module, "PrototypeWorkspaceJobs", "PrototypeRuntimeJobs")
    return jobs_cls(repo=repo)


class _PromoteService:
    def __init__(self) -> None:
        self.called = False

    async def promote_candidate(self, **_kwargs: Any) -> dict[str, Any]:
        self.called = True
        return {"status": "promoted"}


class _FailedPromotionService:
    async def promote_candidate(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "failed",
            "failure_code": "publish_validation_failed",
            "canonical_snapshot_id": "snap_original",
            "retryable": True,
        }


class _RetryablePreviewFailureService:
    async def boot_preview(self, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("preview runtime temporarily unavailable")


class _NoopService:
    pass


async def _seed_branch_workspace(repo, prototype_db):
    workspace = await repo.create_workspace(
        owner_user_id=1,
        title="Task 4 Runtime",
        creation_source="prompt",
        runtime_policy={"external_collaborator_profile": "locked_collab"},
        share_policy={"allow_browser_session_resume": True},
    )
    canonical = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_canonical",
        created_by_user_id=1,
    )
    prototype_db.execute(
        """
        UPDATE prototype_workspaces
        SET canonical_snapshot_id = ?, last_known_good_snapshot_id = ?, canonical_preview_status = ?, publish_validation_status = ?
        WHERE id = ?
        """,
        ("snap_canonical", "snap_canonical", "ready", "validated", workspace["id"]),
    )
    prototype_db.commit()
    return workspace, canonical


def test_jobs_module_exports_expected_task_types() -> None:
    module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.jobs")
    assert {
        "branch_session_bootstrap",
        "preview_boot",
        "snapshot_save",
        "publish_validate_and_promote",
    }.issubset(set(getattr(module, "PROTOTYPE_JOB_TYPES", set())))


def test_default_jobs_manager_is_shared_for_same_environment(repo, tmp_path, monkeypatch) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.jobs")
    jobs_cls = _load_attr(module, "PrototypeWorkspaceJobs", "PrototypeRuntimeJobs")
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "prototype_jobs.db"))
    monkeypatch.setattr(module, "_DEFAULT_JOBS_MANAGER", None, raising=False)
    monkeypatch.setattr(module, "_DEFAULT_JOBS_MANAGER_KEY", None, raising=False)

    first = jobs_cls(repo=repo)
    second = jobs_cls(repo=repo)

    assert first._jobs_manager is second._jobs_manager


@pytest.mark.asyncio
async def test_publish_job_requires_reviewer_user_id_before_service_call() -> None:
    worker_module = importlib.import_module(
        "tldw_Server_API.app.core.Prototype_Workspaces.jobs_worker"
    )
    service = _PromoteService()

    with pytest.raises(ValueError, match="reviewer_user_id is required") as exc_info:
        await worker_module.handle_prototype_job(
            {
                "job_type": "publish_validate_and_promote",
                "payload": {
                    "prototype_workspace_id": "pw_1",
                    "candidate_snapshot_id": "snap_candidate",
                },
            },
            service=service,
        )

    assert service.called is False
    assert getattr(exc_info.value, "retryable", True) is False
    assert getattr(exc_info.value, "failure_code", None) == "invalid_job_payload"


@pytest.mark.asyncio
async def test_preview_boot_transient_runtime_failure_is_retryable_for_worker() -> None:
    worker_module = importlib.import_module(
        "tldw_Server_API.app.core.Prototype_Workspaces.jobs_worker"
    )

    with pytest.raises(RuntimeError, match="temporarily unavailable") as exc_info:
        await worker_module.handle_prototype_job(
            {
                "job_type": "preview_boot",
                "payload": {
                    "prototype_workspace_id": "pw_1",
                    "prototype_session_id": "session_1",
                    "snapshot_id": "snap_1",
                    "runtime_target_url": "http://127.0.0.1:9101",
                },
            },
            service=_RetryablePreviewFailureService(),
        )

    assert getattr(exc_info.value, "retryable", False) is True
    assert getattr(exc_info.value, "failure_code", None) == "runtime_retryable"


@pytest.mark.asyncio
async def test_publish_validation_failure_returns_terminal_job_result() -> None:
    worker_module = importlib.import_module(
        "tldw_Server_API.app.core.Prototype_Workspaces.jobs_worker"
    )

    result = await worker_module.handle_prototype_job(
        {
            "job_type": "publish_validate_and_promote",
            "payload": {
                "prototype_workspace_id": "pw_1",
                "candidate_snapshot_id": "snap_candidate",
                "reviewer_user_id": 7,
                "review_baseline_snapshot_id": "snap_original",
            },
        },
        service=_FailedPromotionService(),
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "publish_validation_failed"
    assert result["canonical_snapshot_id"] == "snap_original"
    assert result["job_type"] == "publish_validate_and_promote"
    assert result["retryable"] is False


@pytest.mark.asyncio
async def test_runtime_jobs_worker_stops_when_stop_event_is_set(monkeypatch) -> None:
    worker_module = importlib.import_module(
        "tldw_Server_API.app.core.Prototype_Workspaces.jobs_worker"
    )
    captured: dict[str, Any] = {}

    class FakeWorkerSDK:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.stop_called = False
            self._stopped = asyncio.Event()
            captured["sdk"] = self

        def stop(self) -> None:
            self.stop_called = True
            self._stopped.set()

        async def run(self, **_kwargs: Any) -> None:
            await asyncio.wait_for(self._stopped.wait(), timeout=1)

    monkeypatch.setattr(worker_module, "WorkerSDK", FakeWorkerSDK)
    monkeypatch.setattr(worker_module, "_jobs_manager", lambda: object())
    stop_event = asyncio.Event()
    worker_task = asyncio.create_task(
        worker_module.run_prototype_jobs_worker(service=_NoopService(), stop_event=stop_event)
    )

    await asyncio.sleep(0)
    stop_event.set()
    await asyncio.wait_for(worker_task, timeout=1)

    assert captured["sdk"].stop_called is True


@pytest.mark.asyncio
async def test_branch_session_bootstrap_is_idempotent_for_same_request_nonce(
    repo,
    prototype_db,
    prototype_jobs,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)

    first = await prototype_jobs.enqueue_branch_session_bootstrap(
        prototype_workspace_id=workspace["id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
        request_nonce="retry-token",
    )
    second = await prototype_jobs.enqueue_branch_session_bootstrap(
        prototype_workspace_id=workspace["id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
        request_nonce="retry-token",
    )

    assert first["id"] == second["id"]
    assert first["job_type"] == "branch_session_bootstrap"
    assert first["idempotency_key"] == second["idempotency_key"]
    assert first["payload"]["prototype_workspace_id"] == workspace["id"]
    assert first["payload"]["base_snapshot_id"] == canonical["snapshot_id"]
    assert first["payload"]["request_nonce"] == "retry-token"


@pytest.mark.asyncio
async def test_branch_session_bootstrap_does_not_reuse_expired_session(
    repo,
    prototype_db,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)
    expired = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    existing = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=canonical["snapshot_id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
        expires_at=expired,
    )

    service_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(service_module, "PrototypeWorkspaceService", "PrototypePromotionService")
    service = service_cls(repo=repo)

    result = await service.create_or_reuse_branch_session(
        prototype_workspace_id=workspace["id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
        request_nonce="fresh-after-expiry",
    )

    assert result["created"] is True
    assert result["session"]["id"] != existing["id"]


@pytest.mark.asyncio
async def test_preview_boot_idempotency_distinguishes_runtime_target_url(
    repo,
    prototype_db,
    prototype_jobs,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)
    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=canonical["snapshot_id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
    )

    first = await prototype_jobs.enqueue_preview_boot(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id=canonical["snapshot_id"],
        runtime_target_url="http://127.0.0.1:9101",
    )
    second = await prototype_jobs.enqueue_preview_boot(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id=canonical["snapshot_id"],
        runtime_target_url="http://127.0.0.1:9102",
    )

    assert first["id"] != second["id"]
    assert first["idempotency_key"] != second["idempotency_key"]
    assert first["payload"]["metadata"]["runtime_profile_version"] == "v1"


@pytest.mark.asyncio
async def test_preview_boot_idempotency_distinguishes_runtime_profile_version(
    repo,
    prototype_db,
    prototype_jobs,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)
    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=canonical["snapshot_id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
    )

    first = await prototype_jobs.enqueue_preview_boot(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id=canonical["snapshot_id"],
        runtime_target_url="http://127.0.0.1:9101",
        runtime_profile_version="v1",
    )
    second = await prototype_jobs.enqueue_preview_boot(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id=canonical["snapshot_id"],
        runtime_target_url="http://127.0.0.1:9101",
        runtime_profile_version="v2",
    )

    assert first["id"] != second["id"]
    assert first["idempotency_key"] != second["idempotency_key"]
    assert second["payload"]["metadata"]["runtime_profile_version"] == "v2"


@pytest.mark.asyncio
async def test_revoked_external_collaborator_cannot_reuse_branch_session(
    repo,
    prototype_db,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)
    actor = await repo.create_shared_actor(
        prototype_workspace_id=workspace["id"],
        share_link_id=51,
        display_name="Revoked Stakeholder",
        runtime_policy_profile="locked_collab",
    )
    await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=canonical["snapshot_id"],
        actor_type="external_collaborator",
        actor_shared_actor_id=actor["id"],
        share_link_id=51,
    )
    await repo.revoke_shared_actor(
        actor["id"],
        revoked_at=datetime.now(timezone.utc).isoformat(),
    )

    service_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(service_module, "PrototypeWorkspaceService", "PrototypePromotionService")
    service = service_cls(repo=repo)

    with pytest.raises(RuntimeError, match="revoked") as exc_info:
        await service.create_or_reuse_branch_session(
            prototype_workspace_id=workspace["id"],
            actor_type="external_collaborator",
            actor_shared_actor_id=actor["id"],
            request_nonce="after-revoke",
        )

    assert getattr(exc_info.value, "retryable", True) is False
    assert getattr(exc_info.value, "failure_code", None) == "runtime_terminal"


@pytest.mark.asyncio
async def test_expired_session_cannot_save_snapshot(
    repo,
    prototype_db,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)
    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=canonical["snapshot_id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
        expires_at=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
    )

    service_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(service_module, "PrototypeWorkspaceService", "PrototypePromotionService")
    service = service_cls(repo=repo)

    with pytest.raises(RuntimeError, match="expired") as exc_info:
        await service.save_session_snapshot(
            prototype_session_id=session["id"],
            snapshot_id="snap_should_not_save",
        )

    assert getattr(exc_info.value, "retryable", True) is False
    assert getattr(exc_info.value, "failure_code", None) == "runtime_terminal"


@pytest.mark.asyncio
async def test_save_session_snapshot_reuses_existing_snapshot_for_same_snapshot_id(
    repo,
    prototype_db,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)
    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=canonical["snapshot_id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
    )
    service_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(service_module, "PrototypeWorkspaceService", "PrototypePromotionService")
    service = service_cls(repo=repo)

    first = await service.save_session_snapshot(
        prototype_session_id=session["id"],
        snapshot_id="snap_retry_safe",
        storage_ref="prototype://retry-safe",
        diff_summary={"step": 1},
    )
    second = await service.save_session_snapshot(
        prototype_session_id=session["id"],
        snapshot_id="snap_retry_safe",
        storage_ref="prototype://retry-safe",
        diff_summary={"step": 1},
    )
    rows = prototype_db.execute(
        "SELECT COUNT(*) FROM prototype_snapshots WHERE id = ?",
        ("snap_retry_safe",),
    ).fetchone()
    refreshed_session = await repo.get_session(session["id"])

    assert first["snapshot_id"] == "snap_retry_safe"
    assert second["snapshot_id"] == "snap_retry_safe"
    assert rows[0] == 1
    assert refreshed_session["last_saved_snapshot_id"] == "snap_retry_safe"


@pytest.mark.asyncio
async def test_archived_workspace_blocks_branch_session_creation(
    repo,
    prototype_db,
):
    workspace, _canonical = await _seed_branch_workspace(repo, prototype_db)
    prototype_db.execute(
        "UPDATE prototype_workspaces SET archived_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), workspace["id"]),
    )
    prototype_db.commit()

    service_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(service_module, "PrototypeWorkspaceService", "PrototypePromotionService")
    service = service_cls(repo=repo)

    with pytest.raises(RuntimeError, match="archived"):
        await service.create_or_reuse_branch_session(
            prototype_workspace_id=workspace["id"],
            actor_type="internal_collaborator",
            actor_user_id=2,
            request_nonce="archived-block",
        )


@pytest.mark.asyncio
async def test_archived_workspace_blocks_snapshot_save(
    repo,
    prototype_db,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)
    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=canonical["snapshot_id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
    )
    prototype_db.execute(
        "UPDATE prototype_workspaces SET archived_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), workspace["id"]),
    )
    prototype_db.commit()

    service_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(service_module, "PrototypeWorkspaceService", "PrototypePromotionService")
    service = service_cls(repo=repo)

    with pytest.raises(RuntimeError, match="archived"):
        await service.save_session_snapshot(
            prototype_session_id=session["id"],
            snapshot_id="snap_archived_block",
        )
