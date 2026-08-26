"""Publication and cleanup reconciliation tests for shared Workspace clones."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.repositories.clone_snapshot_repository import (
    OperationOwnedMediaReference,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    TerminalOperationResultPatchOutcome,
)
from tldw_Server_API.app.core.Sharing.share_audit_service import (
    SHARE_CLONE_FAILED,
    SHARE_CLONED,
    ShareAuditService,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
    SharedWorkspaceAccessContext,
    SharedWorkspaceNotFound,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_clone_jobs_worker import (
    CloneFinalizationOutcome,
    SharedWorkspaceCloneRuntime,
    cleanup_shared_workspace_clone,
    finalize_shared_workspace_clone,
    reconcile_shared_workspace_clone_jobs,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_clone_operations import (
    build_clone_publication_checkpoint,
    clone_request_fingerprint,
    target_workspace_id,
)
from tldw_Server_API.app.core.Sharing.unified_share_audit import (
    UnifiedShareAuditWriter,
)

pytestmark = pytest.mark.integration

OPERATION_ID = "496504b4-85ec-53eb-a0f2-172a67d5434e"
TARGET_ID = target_workspace_id(OPERATION_ID)


def _result(*, publication_confirmed: bool = False) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "outcome": "complete",
        "workspace_id": TARGET_ID,
        "name": "Recipient copy",
        "publication_confirmed": publication_confirmed,
        "counts": {
            f"{kind}_{field}": 0
            for kind in ("sources", "notes", "artifacts", "media")
            for field in ("attempted", "copied", "failed")
        }
        | {"operation_owned_media_count": 0},
        "readiness": {
            "text_search": "ready",
            "citations": "ready",
            "vector_search": "needs_indexing",
        },
        "warnings": [],
    }


def _job(*, status: str = "completed", archived: bool = False) -> dict[str, Any]:
    return {
        "id": 17,
        "uuid": OPERATION_ID,
        "owner_user_id": "9",
        "domain": "sharing",
        "queue": "workspace-clone",
        "job_type": "workspace_clone",
        "batch_group": "share:42",
        "status": status,
        "payload": {
            "schema_version": 1,
            "share_id": 42,
            "recipient_user_id": 9,
            "requested_name": "Recipient copy",
            "request_fingerprint": clone_request_fingerprint(
                share_id=42,
                recipient_user_id=9,
                requested_name="Recipient copy",
            ),
        },
        "result": _result() if status == "completed" else None,
        "archived": archived,
    }


@dataclass
class _Resources:
    chacha: MagicMock
    media: MagicMock
    events: list[str]


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
        workspace={"id": "workspace-alpha", "name": "Evidence review"},
        policy_actions={},
    )


class _AccessService:
    def __init__(self, outcomes: list[Any] | None = None) -> None:
        self.outcomes = outcomes or [_context()]
        self.calls: list[tuple[int, int]] = []

    async def resolve(self, *, share_id: int, recipient_user_id: int):
        self.calls.append((share_id, recipient_user_id))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _ShareRepo:
    async def get_share(self, share_id: int) -> dict[str, Any] | None:
        assert share_id == 42
        return {
            "id": 42,
            "workspace_id": "workspace-alpha",
            "owner_user_id": 7,
            "revoked_at": "2026-08-25T12:00:00+00:00",
        }


def _runtime(
    jobs: Any,
    resources: _Resources,
    *,
    access_service: _AccessService | None = None,
) -> SharedWorkspaceCloneRuntime:
    async def _load_chacha(owner_user_id: int):
        assert owner_user_id == 9
        return resources.chacha

    @contextmanager
    def _media(owner_user_id: int):
        assert owner_user_id == 9
        yield resources.media

    return SharedWorkspaceCloneRuntime(
        jobs=jobs,
        access_service=access_service or _AccessService(),
        load_chacha_db=_load_chacha,
        media_session_factory=_media,
    )


def _runtime_with_audit(
    jobs: Any,
    resources: _Resources,
    audit_service: Any,
    *,
    access_service: _AccessService | None = None,
    share_repo: Any | None = None,
) -> SharedWorkspaceCloneRuntime:
    async def _load_chacha(owner_user_id: int):
        assert owner_user_id == 9
        return resources.chacha

    @contextmanager
    def _media(owner_user_id: int):
        assert owner_user_id == 9
        yield resources.media

    return SharedWorkspaceCloneRuntime(
        jobs=jobs,
        access_service=access_service or _AccessService(),
        load_chacha_db=_load_chacha,
        media_session_factory=_media,
        share_repo=share_repo or _ShareRepo(),
        audit_service=audit_service,
    )


def _pending_resources() -> _Resources:
    events: list[str] = []
    chacha = MagicMock()
    media = MagicMock()
    reference = OperationOwnedMediaReference(
        media_id=12,
        media_uuid="media-12",
        source_identity="source-12",
        expected_content_hash="b" * 64,
    )
    media.list_operation_owned_clone_media.side_effect = [[reference], []]
    media.confirm_operation_owned_clone_media.side_effect = lambda **_kwargs: events.append("media") or 1
    media.delete_operation_owned_clone_media.side_effect = lambda **_kwargs: events.append("delete-media") or 1
    chacha.list_clone_targets_for_reconciliation.return_value = [
        {
            "id": TARGET_ID,
            "system_operation_id": OPERATION_ID,
            "system_operation_kind": "shared_workspace_clone",
            "system_operation_state": "publication_pending",
            "deleted": 0,
        }
    ]
    chacha.confirm_clone_target_publication.side_effect = lambda **_kwargs: (
        events.append("workspace") or {"id": TARGET_ID}
    )
    chacha.discard_clone_target.side_effect = lambda **_kwargs: events.append("discard-workspace") or True
    chacha.get_workspace.return_value = None
    return _Resources(chacha=chacha, media=media, events=events)


@pytest.mark.asyncio
async def test_completed_job_exposes_owned_media_before_workspace() -> None:
    job = _job()
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.side_effect = [
        TerminalOperationResultPatchOutcome.APPLIED,
        TerminalOperationResultPatchOutcome.APPLIED,
    ]
    resources = _pending_resources()

    finalized = await finalize_shared_workspace_clone(
        job,
        _result(),
        runtime=_runtime(jobs, resources),
    )

    assert finalized is CloneFinalizationOutcome.PUBLISHED
    assert resources.events == ["media", "workspace"]
    resources.media.confirm_operation_owned_clone_media.assert_called_once_with(
        operation_id=OPERATION_ID,
        source_identity="source-12",
        expected_content_hash="b" * 64,
    )
    resources.chacha.confirm_clone_target_publication.assert_called_once_with(
        workspace_id=TARGET_ID,
        operation_id=OPERATION_ID,
    )
    checkpoint_command, confirmation_command = (
        call.args[0] for call in jobs.patch_terminal_operation_result.call_args_list
    )
    assert checkpoint_command.replacement_result["publication_state"] == "authorized"
    assert confirmation_command.replacement_result["publication_confirmed"] is True


@pytest.mark.asyncio
async def test_success_audit_is_emitted_only_by_confirmed_result_cas_winner() -> None:
    result = _result()
    result["outcome"] = "partial"
    job = _job() | {"result": result}
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.side_effect = [
        TerminalOperationResultPatchOutcome.APPLIED,
        TerminalOperationResultPatchOutcome.APPLIED,
    ]
    resources = _pending_resources()
    audit = MagicMock()
    audit.log = AsyncMock()

    finalized = await finalize_shared_workspace_clone(
        job,
        result,
        runtime=_runtime_with_audit(jobs, resources, audit),
    )

    assert finalized is CloneFinalizationOutcome.PUBLISHED
    audit.log.assert_awaited_once_with(
        SHARE_CLONED,
        resource_type="workspace",
        resource_id="workspace-alpha",
        owner_user_id=7,
        actor_user_id=9,
        share_id=42,
        metadata={
            "operation_id": OPERATION_ID,
            "target_workspace_id": TARGET_ID,
            "outcome": "partial",
            "counts": result["counts"],
        },
    )


@pytest.mark.asyncio
async def test_success_audit_is_not_duplicated_for_idempotent_confirmation() -> None:
    job = _job()
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.side_effect = [
        TerminalOperationResultPatchOutcome.APPLIED,
        TerminalOperationResultPatchOutcome.IDEMPOTENT,
    ]
    resources = _pending_resources()
    audit = MagicMock()
    audit.log = AsyncMock()

    finalized = await finalize_shared_workspace_clone(
        job,
        _result(),
        runtime=_runtime_with_audit(jobs, resources, audit),
    )

    assert finalized is CloneFinalizationOutcome.PUBLISHED
    audit.log.assert_not_awaited()


@pytest.mark.asyncio
async def test_success_audit_failure_does_not_change_publication_outcome() -> None:
    job = _job()
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.side_effect = [
        TerminalOperationResultPatchOutcome.APPLIED,
        TerminalOperationResultPatchOutcome.APPLIED,
    ]
    resources = _pending_resources()
    audit = MagicMock()
    audit.log = AsyncMock(side_effect=RuntimeError("private audit backend detail"))

    finalized = await finalize_shared_workspace_clone(
        job,
        _result(),
        runtime=_runtime_with_audit(jobs, resources, audit),
    )

    assert finalized is CloneFinalizationOutcome.PUBLISHED


@pytest.mark.asyncio
async def test_share_lookup_failure_skips_audit_without_reverting_publication() -> None:
    job = _job()
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.side_effect = [
        TerminalOperationResultPatchOutcome.APPLIED,
        TerminalOperationResultPatchOutcome.APPLIED,
    ]
    resources = _pending_resources()
    audit = MagicMock()
    audit.log = AsyncMock()
    share_repo = MagicMock()
    share_repo.get_share = AsyncMock(
        side_effect=RuntimeError("private authorization database detail")
    )

    finalized = await finalize_shared_workspace_clone(
        job,
        _result(),
        runtime=_runtime_with_audit(
            jobs,
            resources,
            audit,
            share_repo=share_repo,
        ),
    )

    assert finalized is CloneFinalizationOutcome.PUBLISHED
    audit.log.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminal_clone_event_is_queryable_from_unified_audit_store(
    tmp_path,
) -> None:
    writer = UnifiedShareAuditWriter(db_path=str(tmp_path / "clone-audit.db"))
    audit = ShareAuditService(writer=writer)
    job = _job()
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.side_effect = [
        TerminalOperationResultPatchOutcome.APPLIED,
        TerminalOperationResultPatchOutcome.APPLIED,
    ]
    resources = _pending_resources()

    try:
        finalized = await finalize_shared_workspace_clone(
            job,
            _result(),
            runtime=_runtime_with_audit(jobs, resources, audit),
        )
        events = await audit.query(
            owner_user_id=7,
            resource_type="workspace",
            resource_id="workspace-alpha",
        )
    finally:
        await writer.stop()

    assert finalized is CloneFinalizationOutcome.PUBLISHED
    assert len(events) == 1
    assert events[0]["event_type"] == SHARE_CLONED
    assert events[0]["share_id"] == 42
    assert events[0]["actor_user_id"] == 9
    assert events[0]["metadata"]["target_workspace_id"] == TARGET_ID


@pytest.mark.asyncio
async def test_publication_replay_requires_re_read_of_public_target() -> None:
    checkpoint = build_clone_publication_checkpoint(_result())
    job = _job() | {"result": checkpoint}
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.return_value = TerminalOperationResultPatchOutcome.APPLIED
    resources = _pending_resources()
    access = _AccessService()
    resources.media.list_operation_owned_clone_media.side_effect = [[]]
    resources.chacha.list_clone_targets_for_reconciliation.return_value = []
    resources.chacha.get_workspace.return_value = {
        "id": TARGET_ID,
        "deleted": 0,
        "system_operation_id": None,
        "system_operation_kind": None,
        "system_operation_state": None,
    }

    finalized = await finalize_shared_workspace_clone(
        job,
        checkpoint,
        runtime=_runtime(jobs, resources, access_service=access),
    )

    assert finalized is CloneFinalizationOutcome.PUBLISHED
    assert access.calls == []
    resources.chacha.get_workspace.assert_called_once_with(TARGET_ID)
    resources.chacha.confirm_clone_target_publication.assert_not_called()


@pytest.mark.asyncio
async def test_authorized_checkpoint_finishes_without_reauthorizing() -> None:
    checkpoint = build_clone_publication_checkpoint(_result())
    job = _job() | {"result": checkpoint}
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.return_value = TerminalOperationResultPatchOutcome.APPLIED
    resources = _pending_resources()
    access = _AccessService([SharedWorkspaceNotFound()])

    finalized = await finalize_shared_workspace_clone(
        job,
        checkpoint,
        runtime=_runtime(jobs, resources, access_service=access),
    )

    assert finalized is CloneFinalizationOutcome.PUBLISHED
    assert access.calls == []
    assert resources.events == ["media", "workspace"]


@pytest.mark.asyncio
async def test_publication_checkpoint_conflict_does_not_expose_resources() -> None:
    job = _job()
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.return_value = TerminalOperationResultPatchOutcome.CONFLICT
    resources = _pending_resources()

    finalized = await finalize_shared_workspace_clone(
        job,
        _result(),
        runtime=_runtime(jobs, resources),
    )

    assert finalized is CloneFinalizationOutcome.DEFERRED
    assert resources.events == []


@pytest.mark.asyncio
async def test_revocation_before_publication_cleans_and_compensates() -> None:
    job = _job()
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.side_effect = [
        TerminalOperationResultPatchOutcome.APPLIED,
        TerminalOperationResultPatchOutcome.APPLIED,
    ]
    resources = _pending_resources()
    access = _AccessService([SharedWorkspaceNotFound()])
    audit = MagicMock()
    audit.log = AsyncMock()

    finalized = await finalize_shared_workspace_clone(
        job,
        _result(),
        runtime=_runtime_with_audit(
            jobs,
            resources,
            audit,
            access_service=access,
        ),
    )

    assert finalized is CloneFinalizationOutcome.COMPENSATED
    assert resources.events == ["delete-media", "discard-workspace"]
    resources.media.confirm_operation_owned_clone_media.assert_not_called()
    resources.chacha.confirm_clone_target_publication.assert_not_called()
    aborting_command, aborted_command = (call.args[0] for call in jobs.patch_terminal_operation_result.call_args_list)
    assert aborting_command.allowed_statuses == ("completed",)
    assert aborting_command.replacement_result == {
        "schema_version": 1,
        "publication_state": "aborting",
        "failure_code": "clone_access_revoked",
        "cleanup_state": "pending",
    }
    assert aborted_command.replacement_result == {
        "schema_version": 1,
        "publication_state": "aborted",
        "failure_code": "clone_access_revoked",
        "cleanup_state": "complete",
    }
    audit.log.assert_awaited_once_with(
        SHARE_CLONE_FAILED,
        resource_type="workspace",
        resource_id="workspace-alpha",
        owner_user_id=7,
        actor_user_id=9,
        share_id=42,
        metadata={
            "operation_id": OPERATION_ID,
            "target_workspace_id": TARGET_ID,
            "failure_code": "clone_access_revoked",
            "cleanup_state": "complete",
        },
    )


@pytest.mark.asyncio
async def test_losing_abort_checkpoint_does_not_delete_resources() -> None:
    job = _job()
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.return_value = TerminalOperationResultPatchOutcome.CONFLICT
    resources = _pending_resources()
    access = _AccessService([SharedWorkspaceNotFound()])

    finalized = await finalize_shared_workspace_clone(
        job,
        _result(),
        runtime=_runtime(jobs, resources, access_service=access),
    )

    assert finalized is CloneFinalizationOutcome.DEFERRED
    assert resources.events == []


@pytest.mark.asyncio
async def test_cleanup_deletes_only_enumerated_owned_media_and_exact_target() -> None:
    job = _job(status="failed")
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.return_value = TerminalOperationResultPatchOutcome.APPLIED
    resources = _pending_resources()

    cleaned = await cleanup_shared_workspace_clone(
        job,
        runtime=_runtime(jobs, resources),
        patch_terminal_result=True,
    )

    assert cleaned is True
    resources.media.delete_operation_owned_clone_media.assert_called_once_with(
        operation_id=OPERATION_ID,
        source_identity="source-12",
        expected_content_hash="b" * 64,
    )
    resources.chacha.discard_clone_target.assert_called_once_with(
        workspace_id=TARGET_ID,
        operation_id=OPERATION_ID,
    )
    command = jobs.patch_terminal_operation_result.call_args.args[0]
    assert command.job_uuid == OPERATION_ID
    assert command.owner_user_id == "9"
    assert command.operation_scope == "share:42"
    assert command.allowed_statuses == ("failed", "cancelled", "quarantined")
    assert command.replacement_result == {
        "schema_version": 1,
        "cleanup_state": "complete",
    }


@pytest.mark.asyncio
async def test_failure_audit_is_emitted_only_by_cleanup_result_cas_winner() -> None:
    job = _job(status="failed") | {"error_code": "clone_persistence_failed"}
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.return_value = (
        TerminalOperationResultPatchOutcome.APPLIED
    )
    resources = _pending_resources()
    audit = MagicMock()
    audit.log = AsyncMock()

    cleaned = await cleanup_shared_workspace_clone(
        job,
        runtime=_runtime_with_audit(jobs, resources, audit),
        patch_terminal_result=True,
    )

    assert cleaned is True
    audit.log.assert_awaited_once_with(
        SHARE_CLONE_FAILED,
        resource_type="workspace",
        resource_id="workspace-alpha",
        owner_user_id=7,
        actor_user_id=9,
        share_id=42,
        metadata={
            "operation_id": OPERATION_ID,
            "target_workspace_id": TARGET_ID,
            "failure_code": "clone_persistence_failed",
            "cleanup_state": "complete",
        },
    )


@pytest.mark.asyncio
async def test_failure_audit_is_not_duplicated_for_idempotent_cleanup() -> None:
    job = _job(status="failed") | {"error_code": "clone_persistence_failed"}
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    jobs.patch_terminal_operation_result.return_value = (
        TerminalOperationResultPatchOutcome.IDEMPOTENT
    )
    resources = _pending_resources()
    audit = MagicMock()
    audit.log = AsyncMock()

    cleaned = await cleanup_shared_workspace_clone(
        job,
        runtime=_runtime_with_audit(jobs, resources, audit),
        patch_terminal_result=True,
    )

    assert cleaned is True
    audit.log.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_replay_fails_closed_when_target_is_public() -> None:
    job = _job(status="failed")
    jobs = MagicMock()
    jobs.get_job_or_archived_by_uuid.return_value = job
    resources = _pending_resources()
    resources.media.list_operation_owned_clone_media.side_effect = [[]]
    resources.chacha.list_clone_targets_for_reconciliation.return_value = []
    resources.chacha.discard_clone_target.return_value = False
    resources.chacha.get_workspace.return_value = {"id": TARGET_ID, "deleted": 0}

    cleaned = await cleanup_shared_workspace_clone(
        job,
        runtime=_runtime(jobs, resources),
        patch_terminal_result=True,
    )

    assert cleaned is False
    jobs.patch_terminal_operation_result.assert_not_called()


@pytest.mark.asyncio
async def test_reconciliation_scans_active_and_archive_with_one_bounded_budget() -> None:
    active = _job(status="failed")
    archived = _job(status="completed", archived=True) | {"uuid": "1170b0df-c07c-5bb5-9a19-d8145e02a6b1"}
    jobs = MagicMock()
    jobs.integrity_sweep.return_value = {"fixed": 0}
    jobs.list_jobs.return_value = [active]
    jobs.list_archived_jobs.return_value = [archived]
    jobs.get_job_or_archived_by_uuid.side_effect = [active, archived]
    resources = _pending_resources()
    runtime = _runtime(jobs, resources)

    summary = await reconcile_shared_workspace_clone_jobs(
        jobs=jobs,
        runtime=runtime,
        limit=3,
    )

    assert jobs.list_jobs.call_args.kwargs["limit"] <= 3
    assert jobs.list_archived_jobs.call_args.kwargs["limit"] <= 2
    assert jobs.list_jobs.call_args.kwargs["limit"] + jobs.list_archived_jobs.call_args.kwargs["limit"] <= 3
    assert summary["scanned"] == 2
    jobs.integrity_sweep.assert_called_once_with(
        fix=True,
        domain="sharing",
        queue="workspace-clone",
        job_type="workspace_clone",
    )


@pytest.mark.asyncio
async def test_reconciliation_advances_active_cursor_past_nonterminal_rows() -> None:
    queued = {
        "id": 20,
        "uuid": "ba03c6ee-a45a-586e-b604-03384bb93e81",
        "status": "queued",
        "created_at": datetime(2026, 8, 25, 12, tzinfo=UTC),
    }
    failed = _job(status="failed") | {"created_at": datetime(2026, 8, 25, 11, tzinfo=UTC)}
    jobs = MagicMock()
    jobs.integrity_sweep.return_value = {"fixed": 0}
    jobs.list_jobs.side_effect = [[queued], [failed]]
    jobs.list_archived_jobs.return_value = []
    jobs.get_job_or_archived_by_uuid.return_value = failed
    jobs.patch_terminal_operation_result.return_value = TerminalOperationResultPatchOutcome.APPLIED
    resources = _pending_resources()
    runtime = _runtime(jobs, resources)

    first = await reconcile_shared_workspace_clone_jobs(
        jobs=jobs,
        runtime=runtime,
        limit=2,
    )
    second = await reconcile_shared_workspace_clone_jobs(
        jobs=jobs,
        runtime=runtime,
        limit=2,
    )

    assert first["scanned"] == 1
    assert second["cleaned"] == 1
    second_query = jobs.list_jobs.call_args_list[1].kwargs
    assert second_query["created_before"] == queued["created_at"]
    assert second_query["before_id"] == queued["id"]


@pytest.mark.asyncio
async def test_reconciliation_advances_archive_cursor_between_bounded_pages() -> None:
    first_archived = {
        "id": 20,
        "uuid": "ba03c6ee-a45a-586e-b604-03384bb93e81",
        "status": "processing",
        "_archive_cursor_created_at": datetime(2026, 8, 25, 12, tzinfo=UTC),
        "_archive_cursor_uuid": "ba03c6ee-a45a-586e-b604-03384bb93e81",
        "_archive_locator": 30,
    }
    failed = _job(status="failed", archived=True) | {
        "_archive_cursor_created_at": datetime(2026, 8, 25, 11, tzinfo=UTC),
        "_archive_cursor_uuid": OPERATION_ID,
        "_archive_locator": 29,
    }
    jobs = MagicMock()
    jobs.integrity_sweep.return_value = {"fixed": 0}
    jobs.list_jobs.return_value = []
    jobs.list_archived_jobs.side_effect = [[first_archived], [failed]]
    jobs.get_job_or_archived_by_uuid.return_value = failed
    jobs.patch_terminal_operation_result.return_value = TerminalOperationResultPatchOutcome.APPLIED
    resources = _pending_resources()
    runtime = _runtime(jobs, resources)

    await reconcile_shared_workspace_clone_jobs(jobs=jobs, runtime=runtime, limit=2)
    second = await reconcile_shared_workspace_clone_jobs(
        jobs=jobs,
        runtime=runtime,
        limit=2,
    )

    assert second["cleaned"] == 1
    second_query = jobs.list_archived_jobs.call_args_list[1].kwargs
    assert second_query["created_before"] == first_archived["_archive_cursor_created_at"]
    assert second_query["before_id"] == first_archived["id"]
    assert second_query["before_uuid"] == first_archived["_archive_cursor_uuid"]
    assert second_query["before_archive_locator"] == first_archived["_archive_locator"]


@pytest.mark.asyncio
async def test_reconciliation_ignores_nonterminal_and_malformed_jobs() -> None:
    jobs = MagicMock()
    jobs.integrity_sweep.return_value = {"fixed": 0}
    jobs.list_jobs.return_value = [
        _job(status="processing"),
        _job(status="completed") | {"batch_group": "share:99"},
    ]
    jobs.list_archived_jobs.return_value = []
    resources = _pending_resources()

    summary = await reconcile_shared_workspace_clone_jobs(
        jobs=jobs,
        runtime=_runtime(jobs, resources),
        limit=4,
    )

    assert summary == {
        "scanned": 2,
        "published": 0,
        "cleaned": 0,
        "deferred": 0,
        "invalid": 1,
    }
    resources.chacha.confirm_clone_target_publication.assert_not_called()
    resources.chacha.discard_clone_target.assert_not_called()


@pytest.mark.asyncio
async def test_reconciliation_counts_malformed_completed_result_as_invalid() -> None:
    malformed = _job() | {"result": {"private_path": "/owner/private.db"}}
    jobs = MagicMock()
    jobs.integrity_sweep.return_value = {"fixed": 0}
    jobs.list_jobs.return_value = [malformed]
    jobs.list_archived_jobs.return_value = []
    jobs.get_job_or_archived_by_uuid.return_value = malformed
    resources = _pending_resources()

    summary = await reconcile_shared_workspace_clone_jobs(
        jobs=jobs,
        runtime=_runtime(jobs, resources),
        limit=2,
    )

    assert summary["invalid"] == 1
    assert summary["deferred"] == 0
    assert resources.events == []
