"""Worker boundary tests for durable shared Workspace clone Jobs."""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Sharing import (
    shared_workspace_clone_jobs_worker as clone_worker_module,
)
from tldw_Server_API.app.core.Sharing.clone_models import (
    CloneCancelled,
    CloneCopyCounts,
    CloneRetrievalReadiness,
    WorkspaceCloneResult,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
    SharedWorkspaceAccessContext,
    SharedWorkspaceNotFound,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_clone_jobs_worker import (
    SharedWorkspaceCloneJobError,
    SharedWorkspaceCloneRuntime,
    _build_default_runtime,
    _build_worker_config,
    handle_shared_workspace_clone_job,
    run_shared_workspace_clone_jobs_worker,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_clone_operations import (
    build_clone_admission_command,
    target_workspace_id,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("value", ["not-a-number", "nan", "inf"])
def test_worker_config_uses_safe_completion_timeout_fallback(
    monkeypatch,
    value: str,
) -> None:
    monkeypatch.setenv("SHARED_WORKSPACE_CLONE_COMPLETION_TIMEOUT_SECONDS", value)

    config = _build_worker_config()

    assert config.completion_callback_timeout_seconds == 30.0


@pytest.mark.asyncio
async def test_default_runtime_uses_unified_audit_and_authoritative_share_repo(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps, DB_Deps
    from tldw_Server_API.app.core.AuthNZ import database
    from tldw_Server_API.app.core.AuthNZ.repos import shared_workspace_repo, users_repo

    pool = MagicMock(name="pool")
    share_repo = MagicMock(name="share_repo")
    users = MagicMock(name="users")
    access_service = MagicMock(name="access_service")
    audit_service = MagicMock(name="audit_service")
    share_repo_factory = MagicMock(return_value=share_repo)
    users_repo_factory = MagicMock(return_value=users)
    access_service_factory = MagicMock(return_value=access_service)
    audit_service_factory = MagicMock(return_value=audit_service)
    load_chacha = AsyncMock()
    media_session_factory = MagicMock()

    monkeypatch.setattr(database, "get_db_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(
        shared_workspace_repo,
        "SharedWorkspaceRepo",
        share_repo_factory,
    )
    monkeypatch.setattr(users_repo, "AuthnzUsersRepo", users_repo_factory)
    monkeypatch.setattr(
        ChaCha_Notes_DB_Deps,
        "get_chacha_db_for_owner",
        load_chacha,
    )
    monkeypatch.setattr(
        DB_Deps,
        "managed_media_db_for_owner",
        media_session_factory,
    )
    monkeypatch.setattr(
        clone_worker_module,
        "SharedWorkspaceAccessService",
        access_service_factory,
    )
    monkeypatch.setattr(
        clone_worker_module,
        "ShareAuditService",
        audit_service_factory,
    )

    runtime = await _build_default_runtime(jobs=MagicMock())

    audit_service_factory.assert_called_once_with()
    assert runtime.share_repo is share_repo
    assert runtime.audit_service is audit_service


@pytest.mark.asyncio
async def test_worker_shutdown_stops_owned_audit_service(monkeypatch) -> None:
    jobs = MagicMock(name="jobs")
    audit_service = MagicMock(name="audit_service")
    audit_service.stop = AsyncMock()
    runtime = SharedWorkspaceCloneRuntime(
        jobs=jobs,
        access_service=MagicMock(),
        load_chacha_db=AsyncMock(),
        media_session_factory=MagicMock(),
        audit_service=audit_service,
    )

    async def _build_runtime(*, jobs: Any, stop_event: asyncio.Event):
        runtime.jobs = jobs
        runtime.stop_event = stop_event
        return runtime

    class _WorkerSDK:
        def __init__(self, *_args, **_kwargs) -> None:
            self.stop = MagicMock()

        async def run(self, **_kwargs) -> None:
            assert runtime.stop_event is not None
            runtime.stop_event.set()

    monkeypatch.setattr(clone_worker_module, "jobs_manager_from_env", lambda: jobs)
    monkeypatch.setattr(clone_worker_module, "_build_default_runtime", _build_runtime)
    monkeypatch.setattr(clone_worker_module, "WorkerSDK", _WorkerSDK)

    await run_shared_workspace_clone_jobs_worker()

    audit_service.stop.assert_awaited_once_with()


def _context(*, allow_clone: bool = True) -> SharedWorkspaceAccessContext:
    return SharedWorkspaceAccessContext(
        share_id=42,
        workspace_id="workspace-alpha",
        owner_user_id=7,
        recipient_user_id=9,
        share_scope_type="team",
        share_scope_id=11,
        access_level="view_chat",
        allow_clone=allow_clone,
        owner_display_name="Research owner",
        shared_at="2026-08-20T18:00:00+00:00",
        workspace={
            "id": "workspace-alpha",
            "name": "Evidence review",
            "description": "Review set",
        },
        policy_actions={},
    )


class _AccessService:
    def __init__(self, contexts: list[Any] | None = None) -> None:
        self.contexts = contexts or [_context()]
        self.calls: list[tuple[int, int, int]] = []

    async def resolve(self, *, share_id: int, recipient_user_id: int):
        self.calls.append((share_id, recipient_user_id, threading.get_ident()))
        value = self.contexts.pop(0) if len(self.contexts) > 1 else self.contexts[0]
        if isinstance(value, Exception):
            raise value
        return value


class _MediaSessions:
    def __init__(self) -> None:
        self.source = MagicMock(name="source_media")
        self.target = MagicMock(name="target_media")
        self.events: list[tuple[str, int, int]] = []

    @contextmanager
    def open(self, owner_user_id: int):
        self.events.append(("open", owner_user_id, threading.get_ident()))
        try:
            yield self.source if owner_user_id == 7 else self.target
        finally:
            self.events.append(("close", owner_user_id, threading.get_ident()))


class _CloneService:
    request = None
    worker_thread_id: int | None = None
    cancel_result: bool | None = None

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def clone_workspace(self, request, *, should_cancel, on_progress):
        type(self).request = request
        type(self).worker_thread_id = threading.get_ident()
        type(self).cancel_result = should_cancel()
        on_progress("preparing", 0.2)
        on_progress("finalizing", 1.0)
        return WorkspaceCloneResult(
            workspace_id=request.target_workspace_id,
            name=request.name,
            outcome="complete",
            publication_confirmed=False,
            counts=CloneCopyCounts.empty(),
            readiness=CloneRetrievalReadiness(
                text_search="ready",
                citations="ready",
                vector_search="needs_indexing",
            ),
        )


class _CancellingCloneService(_CloneService):
    def clone_workspace(self, request, *, should_cancel, on_progress):
        assert should_cancel() is True
        raise CloneCancelled(cleanup_state="complete")


@pytest.fixture
def acquired_job(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    monkeypatch.setenv("JOBS_METRICS_GAUGES_ENABLED", "false")
    manager = JobManager(tmp_path / "clone-worker-jobs.db")
    admission = manager.admit_idempotent_operation(
        build_clone_admission_command(
            share_id=42,
            recipient_user_id=9,
            requested_name=None,
            idempotency_key="clone-worker-test-0001",
        )
    )
    job = manager.acquire_next_job(
        domain="sharing",
        queue="workspace-clone",
        worker_id="clone-worker-test",
        lease_seconds=60,
        job_type="workspace_clone",
    )
    assert job is not None
    assert job["uuid"] == admission.job["uuid"]
    return manager, job


def _runtime(
    manager: JobManager,
    *,
    access_service: _AccessService | None = None,
    clone_service_factory: type[_CloneService] = _CloneService,
):
    source_chacha = MagicMock(name="source_chacha")
    target_chacha = MagicMock(name="target_chacha")
    close_threads: list[tuple[int, int]] = []
    source_chacha.close_connection.side_effect = lambda: close_threads.append((7, threading.get_ident()))
    target_chacha.close_connection.side_effect = lambda: close_threads.append((9, threading.get_ident()))
    target_chacha.list_clone_targets_for_reconciliation.side_effect = lambda *, operation_ids, limit: [
        {
            "id": target_workspace_id(operation_ids[0]),
            "system_operation_id": operation_ids[0],
            "system_operation_kind": "shared_workspace_clone",
            "system_operation_state": "publication_pending",
            "deleted": 0,
        }
    ]

    async def _load_chacha(owner_user_id: int):
        return source_chacha if owner_user_id == 7 else target_chacha

    media = _MediaSessions()
    runtime = SharedWorkspaceCloneRuntime(
        jobs=manager,
        access_service=access_service or _AccessService(),
        load_chacha_db=_load_chacha,
        media_session_factory=media.open,
        clone_service_factory=clone_service_factory,
        authorization_timeout_seconds=1.0,
    )
    return runtime, source_chacha, target_chacha, media, close_threads


@pytest.mark.asyncio
async def test_handler_rejects_malformed_scope_before_access_or_database_load(
    acquired_job,
) -> None:
    manager, job = acquired_job
    access = _AccessService()
    loaded: list[int] = []

    async def _load(owner_user_id: int):
        loaded.append(owner_user_id)
        return MagicMock()

    runtime = SharedWorkspaceCloneRuntime(
        jobs=manager,
        access_service=access,
        load_chacha_db=_load,
        media_session_factory=MagicMock(),
    )

    with pytest.raises(SharedWorkspaceCloneJobError) as raised:
        await handle_shared_workspace_clone_job(
            {**job, "batch_group": "share:99"},
            runtime=runtime,
        )

    assert raised.value.failure_code == "clone_validation_failed"
    assert access.calls == []
    assert loaded == []


@pytest.mark.asyncio
async def test_handler_rejects_tampered_payload_before_access_or_database_load(
    acquired_job,
) -> None:
    manager, job = acquired_job
    access = _AccessService()
    loaded: list[int] = []

    async def _load(owner_user_id: int):
        loaded.append(owner_user_id)
        return MagicMock()

    runtime = SharedWorkspaceCloneRuntime(
        jobs=manager,
        access_service=access,
        load_chacha_db=_load,
        media_session_factory=MagicMock(),
    )
    tampered = {
        **job,
        "payload": {**job["payload"], "requested_name": "Changed after admission"},
    }

    with pytest.raises(SharedWorkspaceCloneJobError) as raised:
        await handle_shared_workspace_clone_job(tampered, runtime=runtime)

    assert raised.value.failure_code == "clone_validation_failed"
    assert access.calls == []
    assert loaded == []


@pytest.mark.asyncio
async def test_handler_authorizes_before_loading_owner_databases(acquired_job) -> None:
    manager, job = acquired_job
    access = _AccessService([_context(allow_clone=False)])
    loaded: list[int] = []

    async def _load(owner_user_id: int):
        loaded.append(owner_user_id)
        return MagicMock()

    runtime = SharedWorkspaceCloneRuntime(
        jobs=manager,
        access_service=access,
        load_chacha_db=_load,
        media_session_factory=MagicMock(),
    )

    with pytest.raises(SharedWorkspaceCloneJobError) as raised:
        await handle_shared_workspace_clone_job(job, runtime=runtime)

    assert raised.value.failure_code == "clone_permission_removed"
    assert raised.value.cleanup_state == "complete"
    assert loaded == []


@pytest.mark.asyncio
async def test_handler_maps_database_loader_failure_to_bounded_error(acquired_job) -> None:
    manager, job = acquired_job

    async def _load(_owner_user_id: int):
        raise RuntimeError("database credential leaked by driver")

    runtime = SharedWorkspaceCloneRuntime(
        jobs=manager,
        access_service=_AccessService(),
        load_chacha_db=_load,
        media_session_factory=MagicMock(),
    )

    with pytest.raises(SharedWorkspaceCloneJobError) as raised:
        await handle_shared_workspace_clone_job(job, runtime=runtime)

    assert raised.value.failure_code == "clone_persistence_failed"
    assert raised.value.cleanup_state == "complete"
    assert str(raised.value) == "clone_persistence_failed"


@pytest.mark.asyncio
async def test_handler_maps_request_construction_failure_to_bounded_error(
    acquired_job,
) -> None:
    manager, job = acquired_job
    invalid_context = replace(
        _context(),
        workspace_id="workspace-ü",
        workspace={"id": "workspace-ü", "name": "Evidence review"},
    )
    runtime, *_rest = _runtime(
        manager,
        access_service=_AccessService([invalid_context]),
    )

    with pytest.raises(SharedWorkspaceCloneJobError) as raised:
        await handle_shared_workspace_clone_job(job, runtime=runtime)

    assert raised.value.failure_code == "clone_validation_failed"
    assert raised.value.cleanup_state == "complete"
    assert str(raised.value) == "clone_validation_failed"


@pytest.mark.asyncio
async def test_handler_runs_clone_and_content_connections_in_worker_thread(
    acquired_job,
) -> None:
    manager, job = acquired_job
    event_loop_thread = threading.get_ident()
    runtime, _source, _target, media, close_threads = _runtime(manager)

    result = await handle_shared_workspace_clone_job(job, runtime=runtime)

    assert result["schema_version"] == 1
    assert result["publication_confirmed"] is False
    assert result["workspace_id"] == target_workspace_id(job["uuid"])
    assert _CloneService.request.source_workspace_id == "workspace-alpha"
    assert _CloneService.request.name == "Evidence review"
    assert _CloneService.request.request_fingerprint == job["payload"]["request_fingerprint"]
    assert _CloneService.worker_thread_id != event_loop_thread
    assert _CloneService.cancel_result is False
    assert {event[:2] for event in media.events} == {
        ("open", 7),
        ("close", 7),
        ("open", 9),
        ("close", 9),
    }
    assert all(thread_id == _CloneService.worker_thread_id for _, _, thread_id in media.events)
    assert {owner for owner, _thread_id in close_threads} == {7, 9}
    assert all(thread_id == _CloneService.worker_thread_id for _, thread_id in close_threads)
    assert runtime.progress.snapshot() == {
        "progress_percent": 100.0,
        "progress_message": "finalizing",
    }


@pytest.mark.asyncio
async def test_thread_bridge_rechecks_access_on_event_loop(acquired_job) -> None:
    manager, job = acquired_job
    event_loop_thread = threading.get_ident()
    access = _AccessService([_context(), _context()])
    runtime, *_rest = _runtime(manager, access_service=access)

    await handle_shared_workspace_clone_job(job, runtime=runtime)

    assert len(access.calls) == 2
    assert {thread_id for _share, _recipient, thread_id in access.calls} == {event_loop_thread}


@pytest.mark.asyncio
async def test_revocation_during_copy_becomes_bounded_nonretryable_failure(
    acquired_job,
) -> None:
    manager, job = acquired_job
    access = _AccessService([_context(), SharedWorkspaceNotFound()])
    runtime, *_rest = _runtime(
        manager,
        access_service=access,
        clone_service_factory=_CancellingCloneService,
    )

    with pytest.raises(SharedWorkspaceCloneJobError) as raised:
        await handle_shared_workspace_clone_job(job, runtime=runtime)

    assert raised.value.failure_code == "clone_access_revoked"
    assert raised.value.cleanup_state == "complete"
    assert raised.value.retryable is False
    assert str(raised.value) == "clone_access_revoked"


@pytest.mark.asyncio
async def test_handler_rejects_clone_result_for_another_target(acquired_job) -> None:
    manager, job = acquired_job

    class _WrongTarget(_CloneService):
        def clone_workspace(self, request, *, should_cancel, on_progress):
            result = super().clone_workspace(
                request,
                should_cancel=should_cancel,
                on_progress=on_progress,
            )
            return replace(result, workspace_id="workspace-other")

    runtime, *_rest = _runtime(manager, clone_service_factory=_WrongTarget)

    with pytest.raises(SharedWorkspaceCloneJobError) as raised:
        await handle_shared_workspace_clone_job(job, runtime=runtime)

    assert raised.value.failure_code == "clone_validation_failed"
