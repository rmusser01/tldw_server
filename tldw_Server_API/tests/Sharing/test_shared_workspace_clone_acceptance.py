"""End-to-end SQLite/PostgreSQL Jobs acceptance for shared Workspace clones."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import sharing
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.repositories.clone_snapshot_repository import (
    OperationOwnedMediaPublicationState,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import SLIDES_ARCHIVE_EXACT_FIELDS
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
    SharedWorkspaceAccessContext,
    SharedWorkspaceNotFound,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_clone_jobs_worker import (
    CloneFinalizationOutcome,
    SharedWorkspaceCloneRuntime,
    cleanup_shared_workspace_clone,
    finalize_shared_workspace_clone,
    handle_shared_workspace_clone_job,
    reconcile_shared_workspace_clone_jobs,
)

pytestmark = pytest.mark.integration

_IDEMPOTENCY_KEY = "clone-acceptance-request-0001"
_SOURCE_WORKSPACE_ID = "workspace-alpha"


def _context() -> SharedWorkspaceAccessContext:
    return SharedWorkspaceAccessContext(
        share_id=42,
        workspace_id=_SOURCE_WORKSPACE_ID,
        owner_user_id=7,
        recipient_user_id=9,
        share_scope_type="team",
        share_scope_id=11,
        access_level="view_chat",
        allow_clone=True,
        owner_display_name="Research owner",
        shared_at="2026-08-20T18:00:00+00:00",
        workspace={
            "id": _SOURCE_WORKSPACE_ID,
            "name": "Evidence review",
            "description": "Review set",
        },
        policy_actions={
            "clone_workspace": {"allowed": True, "reason_code": None},
        },
    )


class _AccessService:
    def __init__(self) -> None:
        self.error: Exception | None = None

    async def resolve(self, *, share_id: int, recipient_user_id: int):
        if self.error is not None:
            raise self.error
        context = _context()
        assert (share_id, recipient_user_id) == (
            context.share_id,
            context.recipient_user_id,
        )
        return context

    async def resolve_clone(self, *, share_id: int, recipient_user_id: int):
        return await self.resolve(
            share_id=share_id,
            recipient_user_id=recipient_user_id,
        )


class _ShareRepo:
    async def get_share(self, share_id: int) -> dict[str, Any] | None:
        assert share_id == 42
        return {
            "id": 42,
            "workspace_id": _SOURCE_WORKSPACE_ID,
            "owner_user_id": 7,
            "revoked_at": None,
        }


@dataclass
class _Harness:
    client: TestClient
    app: FastAPI
    jobs: JobManager
    access: _AccessService
    source: CharactersRAGDB
    target: CharactersRAGDB
    runtime: SharedWorkspaceCloneRuntime
    principal: dict[str, Any]
    jobs_dsn: str | None


@pytest.fixture(
    params=(
        pytest.param("sqlite", id="sqlite"),
        pytest.param("postgres", id="postgres", marks=pytest.mark.pg_jobs),
    )
)
def clone_harness(request, monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("JOBS_METRICS_GAUGES_ENABLED", "false")
    monkeypatch.setenv("JOBS_DISABLE_LEASE_ENFORCEMENT", "true")
    jobs_dsn: str | None = None
    if request.param == "postgres":
        pg_temp_db = request.getfixturevalue("pg_temp_db")
        jobs_dsn = str(pg_temp_db["dsn"])
        from tldw_Server_API.app.core.Jobs.pg_migrations import (
            ensure_job_counters_pg,
            ensure_jobs_tables_pg,
        )

        ensure_jobs_tables_pg(jobs_dsn)
        ensure_job_counters_pg(jobs_dsn)
        jobs = JobManager(None, backend="postgres", db_url=jobs_dsn)
    else:
        jobs = JobManager(tmp_path / "clone-acceptance-jobs.db")
    source = CharactersRAGDB(
        str(tmp_path / "clone-source.db"),
        client_id="owner-7",
    )
    target = CharactersRAGDB(
        str(tmp_path / "clone-target.db"),
        client_id="recipient-9",
    )
    source.upsert_workspace(
        _SOURCE_WORKSPACE_ID,
        "Evidence review",
        description="Review set",
    )
    source.add_workspace_source(
        _SOURCE_WORKSPACE_ID,
        {
            "id": "source-1",
            "title": "Source one",
            "source_type": "document",
            "position": 0,
        },
    )
    source.add_workspace_note(
        _SOURCE_WORKSPACE_ID,
        {"title": "Research note", "content": "Grounded observation"},
    )
    source.add_workspace_artifact(
        _SOURCE_WORKSPACE_ID,
        {
            "id": "artifact-1",
            "artifact_type": "report",
            "title": "Draft report",
            "content": "Evidence summary",
        },
    )

    source_media = type("SourceMedia", (), {})()
    source_media.read_media_clone_snapshots = lambda _media_ids: {}
    target_media = type("TargetMedia", (), {})()
    target_media.read_operation_owned_clone_media_readiness = (
        lambda *, operation_id, items: {}
    )
    target_media.list_operation_owned_clone_media = (
        lambda *, operation_id, limit=100: []
    )
    target_media.read_operation_owned_clone_media_publication_state = (
        lambda *, operation_id, limit=100: OperationOwnedMediaPublicationState(
            total_count=0,
            pending_count=0,
            pending=(),
        )
    )

    @contextmanager
    def _media_session(owner_user_id: int):
        yield source_media if owner_user_id == 7 else target_media

    async def _load_chacha(owner_user_id: int):
        return source if owner_user_id == 7 else target

    access = _AccessService()
    runtime = SharedWorkspaceCloneRuntime(
        jobs=jobs,
        access_service=access,
        load_chacha_db=_load_chacha,
        media_session_factory=_media_session,
        share_repo=_ShareRepo(),
        vector_retrieval_configured=True,
    )
    principal: dict[str, Any] = {
        "value": AuthPrincipal(
            kind="user",
            user_id=9,
            username="recipient",
            permissions=["sharing.read"],
        ),
        "user": User(
            id=9,
            username="recipient",
            email="recipient@example.test",
            password_hash="hash",
        ),
    }

    async def _principal():
        return principal["value"]

    async def _user():
        return principal["user"]

    async def _rate_limit(*_args, **_kwargs):
        return None

    async def _audit(*_args, **_kwargs):
        return None

    monkeypatch.setattr(auth_deps, "enforce_rbac_rate_limit", _rate_limit)
    monkeypatch.setattr(sharing, "_audit_log_best_effort", _audit)
    monkeypatch.setattr(sharing, "_get_audit_service", lambda: object())
    app = FastAPI()
    app.include_router(sharing.router, prefix="/api/v1")
    app.dependency_overrides[auth_deps.get_auth_principal] = _principal
    app.dependency_overrides[sharing.get_request_user] = _user
    app.dependency_overrides[sharing.get_shared_workspace_access_service] = (
        lambda: access
    )
    app.dependency_overrides[sharing.try_get_job_manager] = lambda: jobs
    client = TestClient(app, raise_server_exceptions=False)

    try:
        yield _Harness(
            client=client,
            app=app,
            jobs=jobs,
            access=access,
            source=source,
            target=target,
            runtime=runtime,
            principal=principal,
            jobs_dsn=jobs_dsn,
        )
    finally:
        client.close()
        source.close_all_connections()
        target.close_all_connections()


def _post(harness: _Harness):
    return harness.client.post(
        "/api/v1/sharing/shared-with-me/42/clone",
        headers={"Idempotency-Key": _IDEMPOTENCY_KEY},
        json={"name": "Recipient evidence copy"},
    )


async def _run_worker_once(harness: _Harness) -> CloneFinalizationOutcome | bool:
    sdk = WorkerSDK(
        harness.jobs,
        WorkerConfig(
            domain="sharing",
            queue="workspace-clone",
            worker_id="clone-acceptance-worker",
            lease_seconds=60,
            retry_on_exception=False,
        ),
    )
    terminal = asyncio.Event()
    outcome: dict[str, CloneFinalizationOutcome | bool] = {}

    async def _completed(job: dict[str, Any], result: dict[str, Any]) -> None:
        outcome["value"] = await finalize_shared_workspace_clone(
            job,
            result,
            runtime=harness.runtime,
        )
        terminal.set()
        sdk.stop()

    async def _failed(job: dict[str, Any], _exc: Exception) -> None:
        outcome["value"] = await cleanup_shared_workspace_clone(
            job,
            runtime=harness.runtime,
            patch_terminal_result=True,
        )
        terminal.set()
        sdk.stop()

    worker = asyncio.create_task(
        sdk.run(
            handler=lambda job: handle_shared_workspace_clone_job(
                job,
                runtime=harness.runtime,
            ),
            job_type="workspace_clone",
            on_completed=_completed,
            on_failed=_failed,
        )
    )
    await asyncio.wait_for(terminal.wait(), timeout=10)
    await asyncio.wait_for(worker, timeout=10)
    return outcome["value"]


def _archive_job(harness: _Harness, operation_id: str) -> None:
    projection = ", ".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS))
    if harness.jobs_dsn is not None:
        import psycopg

        with psycopg.connect(harness.jobs_dsn) as conn, conn.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO jobs_archive ({projection}) "  # nosec B608
                f"SELECT {projection} FROM jobs WHERE uuid = %s",  # nosec B608
                (operation_id,),
            )
            cursor.execute("DELETE FROM jobs WHERE uuid = %s", (operation_id,))
        return
    with sqlite3.connect(harness.jobs.db_path) as connection:
        connection.execute(
            f"INSERT INTO jobs_archive ({projection}) "  # nosec B608
            f"SELECT {projection} FROM jobs WHERE uuid = ?",  # nosec B608
            (operation_id,),
        )
        connection.execute("DELETE FROM jobs WHERE uuid = ?", (operation_id,))


@pytest.mark.asyncio
async def test_api_worker_publication_and_archived_replay_are_one_durable_flow(
    clone_harness: _Harness,
) -> None:
    first = _post(clone_harness)
    response_loss_replay = _post(clone_harness)

    assert first.status_code == 202
    assert response_loss_replay.json() == first.json()

    outcome = await _run_worker_once(clone_harness)
    operation = first.json()
    status = clone_harness.client.get(operation["poll_href"])

    assert outcome is CloneFinalizationOutcome.PUBLISHED
    assert status.status_code == 200
    assert status.json()["status"] == "succeeded"
    assert status.json()["result"]["outcome"] == "partial"
    assert status.json()["result"]["readiness"]["vector_search"] == (
        "needs_indexing"
    )
    assert clone_harness.target.get_workspace(operation["workspace_id"])["id"] == (
        operation["workspace_id"]
    )
    assert len(clone_harness.target.list_workspace_sources(operation["workspace_id"])) == 1
    assert len(clone_harness.target.list_workspace_notes(operation["workspace_id"])) == 1
    assert len(clone_harness.target.list_workspace_artifacts(operation["workspace_id"])) == 1

    _archive_job(clone_harness, operation["operation_id"])
    clone_harness.access.error = SharedWorkspaceNotFound()
    archived_replay = _post(clone_harness)

    assert archived_replay.status_code == 200
    assert archived_replay.json() == status.json()

    clone_harness.principal["value"] = AuthPrincipal(
        kind="user",
        user_id=10,
        username="other-recipient",
        permissions=["sharing.read"],
    )
    clone_harness.principal["user"] = User(
        id=10,
        username="other-recipient",
        email="other@example.test",
        password_hash="hash",
    )
    assert clone_harness.client.get(operation["poll_href"]).status_code == 404


@pytest.mark.asyncio
async def test_hard_exit_reconciliation_keeps_target_hidden_until_publication(
    clone_harness: _Harness,
) -> None:
    admitted = _post(clone_harness).json()
    job = clone_harness.jobs.acquire_next_job(
        domain="sharing",
        queue="workspace-clone",
        worker_id="clone-hard-exit-worker",
        lease_seconds=60,
        job_type="workspace_clone",
    )
    assert job is not None
    result = await handle_shared_workspace_clone_job(
        job,
        runtime=clone_harness.runtime,
    )
    assert clone_harness.jobs.complete_job(
        int(job["id"]),
        result=result,
        worker_id="clone-hard-exit-worker",
        lease_id=str(job["lease_id"]),
    )

    before = clone_harness.client.get(admitted["poll_href"])
    assert before.json()["status"] == "running"
    assert before.json()["progress"]["phase"] == "finalizing"
    assert clone_harness.target.get_workspace(admitted["workspace_id"]) is None

    summary = await reconcile_shared_workspace_clone_jobs(
        jobs=clone_harness.jobs,
        runtime=clone_harness.runtime,
        limit=10,
    )
    after = clone_harness.client.get(admitted["poll_href"])

    assert summary["published"] == 1
    assert after.json()["status"] == "succeeded"
    assert clone_harness.target.get_workspace(admitted["workspace_id"])["id"] == (
        admitted["workspace_id"]
    )


@pytest.mark.asyncio
async def test_revocation_before_execution_fails_without_leaving_a_target(
    clone_harness: _Harness,
) -> None:
    admitted = _post(clone_harness).json()
    clone_harness.access.error = SharedWorkspaceNotFound()

    cleaned = await _run_worker_once(clone_harness)
    status = clone_harness.client.get(admitted["poll_href"])

    assert cleaned is True
    assert status.json()["status"] == "failed"
    assert status.json()["error"] == {
        "code": "clone_access_revoked",
        "message_key": "sharing.clone.errors.clone_access_revoked",
        "message": "Access to the shared workspace ended before the copy completed.",
        "cleanup_state": "complete",
    }
    assert clone_harness.target.get_workspace(admitted["workspace_id"]) is None


@pytest.mark.asyncio
async def test_fatal_publication_failure_is_cleaned_and_bounded(
    clone_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _post(clone_harness).json()

    def _fail_publication(**_kwargs):
        raise RuntimeError("private publication database detail")

    monkeypatch.setattr(
        clone_harness.target,
        "publish_clone_target",
        _fail_publication,
    )
    cleaned = await _run_worker_once(clone_harness)
    status = clone_harness.client.get(admitted["poll_href"])

    assert cleaned is True
    assert status.json()["status"] == "failed"
    assert status.json()["error"]["code"] == "clone_failed"
    assert status.json()["error"]["message_key"] == (
        "sharing.clone.errors.clone_failed"
    )
    assert status.json()["error"]["cleanup_state"] == "complete"
    assert "private" not in str(status.json())
    assert clone_harness.target.get_workspace(admitted["workspace_id"]) is None
