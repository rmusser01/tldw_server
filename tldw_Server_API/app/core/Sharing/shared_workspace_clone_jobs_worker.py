"""Durable worker, publication, and reconciliation for shared Workspace clones."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import math
import os
import re
import threading
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.shared_workspace_recipient_schemas import (
    SharedWorkspaceCloneResult as SharedWorkspaceCloneResultModel,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    TerminalOperationResultPatchCommand,
    TerminalOperationResultPatchOutcome,
    terminal_operation_result_fingerprint,
)
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int, jobs_manager_from_env
from tldw_Server_API.app.core.Sharing.clone_models import (
    CloneCancelled,
    ClonePersistenceError,
    CloneSnapshotUnavailable,
    WorkspaceCloneRequest,
    WorkspaceCloneResult,
)
from tldw_Server_API.app.core.Sharing.clone_service import CloneService
from tldw_Server_API.app.core.Sharing.share_audit_service import (
    SHARE_CLONE_FAILED,
    SHARE_CLONED,
    ShareAuditService,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
    SharedWorkspaceAccessContext,
    SharedWorkspaceAccessService,
    SharedWorkspaceNotFound,
    SharedWorkspaceUnavailable,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_clone_operations import (
    CLONE_DOMAIN,
    CLONE_JOB_TYPE,
    CLONE_QUEUE,
    CLONE_SCHEMA_VERSION,
    build_clone_publication_abort,
    build_clone_publication_checkpoint,
    clone_request_fingerprint,
    normalize_clone_name,
    parse_clone_publication_abort,
    parse_clone_publication_checkpoint,
    target_workspace_id,
)

_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "share_id",
        "recipient_user_id",
        "requested_name",
        "request_fingerprint",
    }
)
_PROGRESS_PHASES = frozenset(
    {
        "queued",
        "authorizing",
        "preparing",
        "sources",
        "notes",
        "artifacts",
        "finalizing",
    }
)
_TERMINAL_FAILURE_STATUSES = ("failed", "cancelled", "quarantined")
_TERMINAL_STATUSES = frozenset({"completed", *_TERMINAL_FAILURE_STATUSES})
_FAILURE_CODE_RE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_RECONCILIATION_LIMIT = 100
_CLONE_COUNT_FIELDS = frozenset(
    {
        f"{kind}_{field}"
        for kind in ("sources", "notes", "artifacts", "media")
        for field in ("attempted", "copied", "failed")
    }
    | {"operation_owned_media_count"}
)


class SharedWorkspaceCloneJobError(RuntimeError):
    """Bounded worker failure safe for Jobs terminal metadata."""

    def __init__(
        self,
        failure_code: str,
        *,
        cleanup_state: str = "unknown",
        retryable: bool = False,
    ) -> None:
        if _FAILURE_CODE_RE.fullmatch(str(failure_code)) is None:
            failure_code = "clone_persistence_failed"
        if cleanup_state not in {"complete", "pending", "unknown"}:
            cleanup_state = "unknown"
        self.failure_code = failure_code
        self.cleanup_state = cleanup_state
        self.retryable = bool(retryable)
        super().__init__(failure_code)


class CloneFinalizationOutcome(str, Enum):
    """Bounded result of one publication reconciliation attempt."""

    PUBLISHED = "published"
    COMPENSATED = "compensated"
    DEFERRED = "deferred"


@dataclass(slots=True)
class CloneProgressState:
    """Thread-safe, content-free progress snapshot consumed by WorkerSDK."""

    _percent: float | None = field(init=False, default=None)
    _message: str | None = field(init=False, default=None)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def reset(self) -> None:
        with self._lock:
            self._percent = None
            self._message = None

    def update(self, phase: str, fraction: float) -> None:
        if phase not in _PROGRESS_PHASES:
            raise ValueError("unsupported clone progress phase")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise ValueError("clone progress fraction must be numeric")
        normalized = float(fraction)
        if not 0.0 <= normalized <= 1.0:
            raise ValueError("clone progress fraction must be between zero and one")
        with self._lock:
            if self._percent is not None and normalized * 100 < self._percent:
                raise ValueError("clone progress must be monotonic")
            self._percent = normalized * 100
            self._message = phase

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {}
            if self._percent is not None:
                result["progress_percent"] = self._percent
            if self._message is not None:
                result["progress_message"] = self._message
            return result


@dataclass(frozen=True, slots=True)
class _ActiveReconciliationCursor:
    created_before: datetime
    before_id: int


@dataclass(frozen=True, slots=True)
class _ArchiveReconciliationCursor:
    created_before: datetime
    before_id: int
    before_uuid: str
    before_archive_locator: str | int


@dataclass(slots=True)
class _CloneReconciliationState:
    active: _ActiveReconciliationCursor | None = None
    archived: _ArchiveReconciliationCursor | None = None


@dataclass(slots=True)
class SharedWorkspaceCloneRuntime:
    """Explicit runtime dependencies shared by handler and reconciler."""

    jobs: Any
    access_service: Any
    load_chacha_db: Callable[[int], Awaitable[Any]]
    media_session_factory: Callable[[int], AbstractContextManager[Any]]
    share_repo: Any | None = None
    audit_service: Any | None = None
    clone_service_factory: Callable[..., Any] = CloneService
    vector_retrieval_configured: bool = False
    authorization_timeout_seconds: float = 5.0
    progress: CloneProgressState = field(default_factory=CloneProgressState)
    reconciliation: _CloneReconciliationState = field(default_factory=_CloneReconciliationState)
    stop_event: asyncio.Event | None = None

    def __post_init__(self) -> None:
        if not callable(self.load_chacha_db):
            raise TypeError("load_chacha_db must be callable")
        if not callable(self.media_session_factory):
            raise TypeError("media_session_factory must be callable")
        if not callable(self.clone_service_factory):
            raise TypeError("clone_service_factory must be callable")
        if not isinstance(self.vector_retrieval_configured, bool):
            raise TypeError("vector_retrieval_configured must be a boolean")
        if float(self.authorization_timeout_seconds) <= 0:
            raise ValueError("authorization_timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class _CloneIdentity:
    operation_id: str
    target_workspace_id: str
    share_id: int
    recipient_user_id: int
    requested_name: str | None
    request_fingerprint: str
    operation_scope: str


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if normalized <= 0 or str(value).strip() != str(normalized):
        raise ValueError(f"{field_name} must be a positive integer")
    return normalized


def _validate_job_identity(
    job: Mapping[str, Any],
    *,
    allowed_statuses: set[str] | frozenset[str] | None = None,
) -> _CloneIdentity:
    """Validate all persisted correlation fields without trusting payload claims."""

    try:
        operation_id = str(UUID(str(job.get("uuid") or "")))
        recipient_user_id = _positive_integer(
            job.get("owner_user_id"),
            "owner_user_id",
        )
        payload = _mapping(job.get("payload"))
        if set(payload) != _PAYLOAD_FIELDS:
            raise ValueError("clone payload fields are malformed")
        share_id = _positive_integer(payload.get("share_id"), "share_id")
        payload_recipient = _positive_integer(
            payload.get("recipient_user_id"),
            "recipient_user_id",
        )
        requested_name = normalize_clone_name(payload.get("requested_name"))
        request_fingerprint = str(payload.get("request_fingerprint") or "")
        expected_fingerprint = clone_request_fingerprint(
            share_id=share_id,
            recipient_user_id=payload_recipient,
            requested_name=requested_name,
        )
        correlation_matches = (
            str(job.get("domain") or "") == CLONE_DOMAIN
            and str(job.get("queue") or "") == CLONE_QUEUE
            and str(job.get("job_type") or "") == CLONE_JOB_TYPE
            and str(job.get("batch_group") or "") == f"share:{share_id}"
            and payload.get("schema_version") == CLONE_SCHEMA_VERSION
            and payload_recipient == recipient_user_id
            and hmac.compare_digest(request_fingerprint, expected_fingerprint)
        )
        if not correlation_matches:
            raise ValueError("clone Job correlation is malformed")
        if allowed_statuses is not None and str(job.get("status") or "") not in allowed_statuses:
            raise ValueError("clone Job status is invalid for this operation")
        if job.get("max_retries") is not None and int(job["max_retries"]) != 0:
            raise ValueError("clone Jobs must disable automatic retries")
    except (TypeError, ValueError, AttributeError) as exc:
        raise SharedWorkspaceCloneJobError(
            "clone_validation_failed",
            cleanup_state="unknown",
        ) from exc
    return _CloneIdentity(
        operation_id=operation_id,
        target_workspace_id=target_workspace_id(operation_id),
        share_id=share_id,
        recipient_user_id=recipient_user_id,
        requested_name=requested_name,
        request_fingerprint=request_fingerprint,
        operation_scope=f"share:{share_id}",
    )


def _context_matches(
    context: SharedWorkspaceAccessContext,
    identity: _CloneIdentity,
) -> bool:
    workspace = _mapping(context.workspace)
    return bool(
        context.share_id == identity.share_id
        and context.recipient_user_id == identity.recipient_user_id
        and isinstance(context.owner_user_id, int)
        and context.owner_user_id > 0
        and isinstance(context.workspace_id, str)
        and bool(context.workspace_id)
        and str(workspace.get("id") or "") == context.workspace_id
    )


async def _authorize_clone(
    identity: _CloneIdentity,
    *,
    runtime: SharedWorkspaceCloneRuntime,
) -> SharedWorkspaceAccessContext:
    try:
        context = await runtime.access_service.resolve(
            share_id=identity.share_id,
            recipient_user_id=identity.recipient_user_id,
        )
    except SharedWorkspaceNotFound as exc:
        raise SharedWorkspaceCloneJobError(
            "clone_access_revoked",
            cleanup_state="complete",
        ) from exc
    except SharedWorkspaceUnavailable as exc:
        raise SharedWorkspaceCloneJobError(
            "clone_interrupted",
            cleanup_state="complete",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - authorization is a trust boundary
        raise SharedWorkspaceCloneJobError(
            "clone_interrupted",
            cleanup_state="complete",
        ) from exc
    if not isinstance(context, SharedWorkspaceAccessContext) or not _context_matches(
        context,
        identity,
    ):
        raise SharedWorkspaceCloneJobError(
            "clone_validation_failed",
            cleanup_state="complete",
        )
    if not context.allow_clone:
        raise SharedWorkspaceCloneJobError(
            "clone_permission_removed",
            cleanup_state="complete",
        )
    return context


def _same_job_identity(
    job: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> bool:
    return all(
        job.get(field_name) == reference.get(field_name)
        for field_name in (
            "uuid",
            "owner_user_id",
            "domain",
            "queue",
            "job_type",
            "batch_group",
        )
    )


async def _cancellation_reason(
    acquired_job: Mapping[str, Any],
    identity: _CloneIdentity,
    *,
    runtime: SharedWorkspaceCloneRuntime,
) -> str | None:
    if runtime.stop_event is not None and runtime.stop_event.is_set():
        return "clone_interrupted"
    try:
        current = await asyncio.to_thread(
            runtime.jobs.get_job_or_archived_by_uuid,
            identity.operation_id,
            domain=CLONE_DOMAIN,
            owner_user_id=str(identity.recipient_user_id),
        )
    except Exception:  # noqa: BLE001 - cancellation checks fail closed
        return "clone_interrupted"
    if (
        not isinstance(current, Mapping)
        or not _same_job_identity(current, acquired_job)
        or current.get("status") != "processing"
    ):
        return "clone_interrupted"
    if current.get("cancel_requested_at"):
        return "clone_cancelled"
    try:
        context = await runtime.access_service.resolve(
            share_id=identity.share_id,
            recipient_user_id=identity.recipient_user_id,
        )
    except SharedWorkspaceNotFound:
        return "clone_access_revoked"
    except SharedWorkspaceUnavailable:
        return "clone_interrupted"
    except Exception:  # noqa: BLE001 - authorization checks fail closed
        return "clone_interrupted"
    if not isinstance(context, SharedWorkspaceAccessContext) or not _context_matches(
        context,
        identity,
    ):
        return "clone_interrupted"
    if not context.allow_clone:
        return "clone_permission_removed"
    return None


def _close_chacha_thread_connection(database: Any) -> None:
    close = getattr(database, "close_connection", None)
    if callable(close):
        close()


def _is_exact_pending_target(
    row: Mapping[str, Any],
    identity: _CloneIdentity,
) -> bool:
    return bool(
        str(row.get("id") or "") == identity.target_workspace_id
        and str(row.get("system_operation_id") or "") == identity.operation_id
        and str(row.get("system_operation_kind") or "") == "shared_workspace_clone"
        and str(row.get("system_operation_state") or "") == "publication_pending"
        and not bool(row.get("deleted"))
    )


def _serialize_clone_result(
    result: WorkspaceCloneResult,
    identity: _CloneIdentity,
    target_chacha: Any,
) -> dict[str, Any]:
    if (
        not isinstance(result, WorkspaceCloneResult)
        or result.workspace_id != identity.target_workspace_id
        or result.publication_confirmed
    ):
        raise SharedWorkspaceCloneJobError(
            "clone_validation_failed",
            cleanup_state="unknown",
        )
    rows = target_chacha.list_clone_targets_for_reconciliation(
        operation_ids=[identity.operation_id],
        limit=2,
    )
    if len(rows) != 1 or not _is_exact_pending_target(rows[0], identity):
        raise SharedWorkspaceCloneJobError(
            "clone_validation_failed",
            cleanup_state="unknown",
        )
    candidate = asdict(result)
    candidate["schema_version"] = CLONE_SCHEMA_VERSION
    try:
        validated = SharedWorkspaceCloneResultModel.model_validate(candidate)
    except ValidationError as exc:
        raise SharedWorkspaceCloneJobError(
            "clone_validation_failed",
            cleanup_state="unknown",
        ) from exc
    return validated.model_dump(mode="json")


async def handle_shared_workspace_clone_job(
    job: dict[str, Any],
    *,
    runtime: SharedWorkspaceCloneRuntime,
) -> dict[str, Any]:
    """Authorize and execute one deterministic clone in a worker-owned thread."""

    identity = _validate_job_identity(job, allowed_statuses={"processing"})
    runtime.progress.reset()
    context = await _authorize_clone(identity, runtime=runtime)
    try:
        source_chacha = await runtime.load_chacha_db(context.owner_user_id)
        target_chacha = await runtime.load_chacha_db(identity.recipient_user_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - database resolution is classified
        logger.bind(exception_type=type(exc).__name__).warning("Shared Workspace clone database resolution failed")
        raise SharedWorkspaceCloneJobError(
            "clone_persistence_failed",
            cleanup_state="complete",
        ) from exc
    if source_chacha is None or target_chacha is None:
        raise SharedWorkspaceCloneJobError(
            "clone_persistence_failed",
            cleanup_state="complete",
        )

    try:
        clone_name = identity.requested_name or normalize_clone_name(_mapping(context.workspace).get("name"))
        request = WorkspaceCloneRequest(
            source_workspace_id=context.workspace_id,
            target_workspace_id=identity.target_workspace_id,
            operation_id=identity.operation_id,
            request_fingerprint=identity.request_fingerprint,
            name=clone_name,
        )
    except (TypeError, ValueError) as exc:
        raise SharedWorkspaceCloneJobError(
            "clone_validation_failed",
            cleanup_state="complete",
        ) from exc
    event_loop = asyncio.get_running_loop()
    reason_lock = threading.Lock()
    cancellation_code: str | None = None

    def _record_reason(reason: str) -> None:
        nonlocal cancellation_code
        with reason_lock:
            if cancellation_code is None:
                cancellation_code = reason

    def _should_cancel() -> bool:
        future = asyncio.run_coroutine_threadsafe(
            _cancellation_reason(job, identity, runtime=runtime),
            event_loop,
        )
        try:
            reason = future.result(timeout=float(runtime.authorization_timeout_seconds))
        except Exception:  # noqa: BLE001 - timeout and loop failure cancel safely
            future.cancel()
            reason = "clone_interrupted"
        if reason is not None:
            _record_reason(reason)
            return True
        return False

    def _run_clone() -> dict[str, Any]:
        try:
            with runtime.media_session_factory(context.owner_user_id) as source_media:
                with runtime.media_session_factory(identity.recipient_user_id) as target_media:
                    service = runtime.clone_service_factory(
                        source_chacha,
                        source_media,
                        target_chacha,
                        target_media,
                        vector_retrieval_configured=(runtime.vector_retrieval_configured),
                    )
                    result = service.clone_workspace(
                        request,
                        should_cancel=_should_cancel,
                        on_progress=runtime.progress.update,
                    )
                    return _serialize_clone_result(result, identity, target_chacha)
        finally:
            seen: set[int] = set()
            for database in (source_chacha, target_chacha):
                if id(database) in seen:
                    continue
                seen.add(id(database))
                _close_chacha_thread_connection(database)

    try:
        return await asyncio.to_thread(_run_clone)
    except asyncio.CancelledError:
        raise
    except SharedWorkspaceCloneJobError:
        raise
    except CloneCancelled as exc:
        with reason_lock:
            reason = cancellation_code or exc.code
        raise SharedWorkspaceCloneJobError(
            reason,
            cleanup_state=exc.cleanup_state,
        ) from exc
    except (CloneSnapshotUnavailable, ClonePersistenceError) as exc:
        raise SharedWorkspaceCloneJobError(
            exc.code,
            cleanup_state=exc.cleanup_state,
        ) from exc
    except (TypeError, ValueError) as exc:
        raise SharedWorkspaceCloneJobError(
            "clone_validation_failed",
            cleanup_state="unknown",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - persistence boundary is classified
        logger.bind(exception_type=type(exc).__name__).warning("Shared Workspace clone execution failed")
        raise SharedWorkspaceCloneJobError(
            "clone_persistence_failed",
            cleanup_state="unknown",
        ) from exc


def _validated_clone_result(
    result: Any,
    identity: _CloneIdentity,
    *,
    publication_confirmed: bool | None = None,
) -> dict[str, Any]:
    try:
        validated = SharedWorkspaceCloneResultModel.model_validate(result)
    except ValidationError as exc:
        raise SharedWorkspaceCloneJobError("clone_validation_failed") from exc
    if validated.workspace_id != identity.target_workspace_id or (
        publication_confirmed is not None and validated.publication_confirmed is not publication_confirmed
    ):
        raise SharedWorkspaceCloneJobError("clone_validation_failed")
    return validated.model_dump(mode="json")


def _classify_completed_result(
    result: Any,
    identity: _CloneIdentity,
) -> tuple[str, dict[str, Any] | None, tuple[str, str] | None]:
    try:
        publication_abort = parse_clone_publication_abort(result)
        if publication_abort is not None:
            return (
                "aborted" if publication_abort[1] == "complete" else "aborting",
                None,
                publication_abort,
            )
        checkpoint = parse_clone_publication_checkpoint(result)
    except (TypeError, ValueError, ValidationError) as exc:
        raise SharedWorkspaceCloneJobError("clone_validation_failed") from exc
    if checkpoint is not None:
        return (
            "authorized",
            _validated_clone_result(
                checkpoint,
                identity,
                publication_confirmed=False,
            ),
            None,
        )
    clone_result = _validated_clone_result(result, identity)
    return (
        "published" if clone_result["publication_confirmed"] else "unconfirmed",
        clone_result,
        None,
    )


def _public_target_is_exact(workspace: Any, identity: _CloneIdentity) -> bool:
    row = _mapping(workspace)
    return bool(
        str(row.get("id") or "") == identity.target_workspace_id
        and not bool(row.get("deleted"))
        and not row.get("system_operation_id")
        and not row.get("system_operation_kind")
        and not row.get("system_operation_state")
    )


def _publish_clone_resources(
    identity: _CloneIdentity,
    *,
    target_chacha: Any,
    target_media: Any,
) -> bool:
    references = target_media.list_operation_owned_clone_media(
        operation_id=identity.operation_id,
        limit=_RECONCILIATION_LIMIT,
    )
    for reference in references:
        changed = target_media.confirm_operation_owned_clone_media(
            operation_id=identity.operation_id,
            source_identity=reference.source_identity,
            expected_content_hash=reference.expected_content_hash,
        )
        if changed != 1:
            return False
    if len(references) == _RECONCILIATION_LIMIT:
        remaining = target_media.list_operation_owned_clone_media(
            operation_id=identity.operation_id,
            limit=_RECONCILIATION_LIMIT,
        )
        if remaining:
            return False

    pending = target_chacha.list_clone_targets_for_reconciliation(
        operation_ids=[identity.operation_id],
        limit=2,
    )
    if len(pending) > 1:
        return False
    if pending:
        if not _is_exact_pending_target(pending[0], identity):
            return False
        published = target_chacha.confirm_clone_target_publication(
            workspace_id=identity.target_workspace_id,
            operation_id=identity.operation_id,
        )
        if _mapping(published).get("id") == identity.target_workspace_id:
            return True
    return _public_target_is_exact(
        target_chacha.get_workspace(identity.target_workspace_id),
        identity,
    )


def _published_clone_resources_are_exact(
    identity: _CloneIdentity,
    *,
    target_chacha: Any,
    target_media: Any,
) -> bool:
    references = target_media.list_operation_owned_clone_media(
        operation_id=identity.operation_id,
        limit=1,
    )
    if references:
        return False
    pending = target_chacha.list_clone_targets_for_reconciliation(
        operation_ids=[identity.operation_id],
        limit=1,
    )
    if pending:
        return False
    return _public_target_is_exact(
        target_chacha.get_workspace(identity.target_workspace_id),
        identity,
    )


async def _run_target_resource_action(
    identity: _CloneIdentity,
    *,
    runtime: SharedWorkspaceCloneRuntime,
    action: Callable[[Any, Any], bool],
) -> bool:
    target_chacha = await runtime.load_chacha_db(identity.recipient_user_id)
    if target_chacha is None:
        return False

    def _run() -> bool:
        try:
            with runtime.media_session_factory(identity.recipient_user_id) as target_media:
                return action(target_chacha, target_media)
        finally:
            _close_chacha_thread_connection(target_chacha)

    return await asyncio.to_thread(_run)


async def _patch_completed_clone_result(
    identity: _CloneIdentity,
    *,
    runtime: SharedWorkspaceCloneRuntime,
    current_result: Mapping[str, Any],
    replacement_result: Mapping[str, Any],
) -> TerminalOperationResultPatchOutcome:
    command = TerminalOperationResultPatchCommand(
        job_uuid=identity.operation_id,
        owner_user_id=str(identity.recipient_user_id),
        domain=CLONE_DOMAIN,
        queue=CLONE_QUEUE,
        job_type=CLONE_JOB_TYPE,
        operation_scope=identity.operation_scope,
        allowed_statuses=("completed",),
        expected_result_fingerprint=terminal_operation_result_fingerprint(dict(current_result)),
        replacement_result=dict(replacement_result),
    )
    return await asyncio.to_thread(
        runtime.jobs.patch_terminal_operation_result,
        command,
    )


def _terminal_patch_succeeded(outcome: TerminalOperationResultPatchOutcome) -> bool:
    return outcome in {
        TerminalOperationResultPatchOutcome.APPLIED,
        TerminalOperationResultPatchOutcome.IDEMPOTENT,
    }


def _bounded_failure_code(value: Any) -> str:
    candidate = str(value or "").strip()
    return candidate if _FAILURE_CODE_RE.fullmatch(candidate) else "clone_failed"


def _bounded_clone_counts(result: Mapping[str, Any]) -> dict[str, int]:
    counts = _mapping(result.get("counts"))
    return {
        field: int(counts[field])
        for field in sorted(_CLONE_COUNT_FIELDS)
        if isinstance(counts.get(field), int) and not isinstance(counts.get(field), bool)
    }


async def _resolve_clone_audit_attribution(
    identity: _CloneIdentity,
    *,
    runtime: SharedWorkspaceCloneRuntime,
) -> tuple[str, int] | None:
    share_repo = runtime.share_repo
    if share_repo is None:
        return None
    try:
        share = _mapping(await share_repo.get_share(identity.share_id))
        share_id = _positive_integer(share.get("id"), "share_id")
        owner_user_id = _positive_integer(
            share.get("owner_user_id"),
            "owner_user_id",
        )
        workspace_id = share.get("workspace_id")
        if (
            share_id != identity.share_id
            or not isinstance(workspace_id, str)
            or not workspace_id
        ):
            return None
        return workspace_id, owner_user_id
    except Exception as exc:  # noqa: BLE001 - audit attribution is best effort
        logger.bind(exception_type=type(exc).__name__).warning(
            "Shared Workspace clone audit attribution failed"
        )
        return None


async def _audit_clone_transition(
    identity: _CloneIdentity,
    *,
    runtime: SharedWorkspaceCloneRuntime,
    event_type: str,
    metadata: dict[str, Any],
) -> None:
    audit_service = runtime.audit_service
    if audit_service is None:
        return
    attribution = await _resolve_clone_audit_attribution(
        identity,
        runtime=runtime,
    )
    if attribution is None:
        return
    source_workspace_id, source_owner_user_id = attribution
    try:
        await audit_service.log(
            event_type,
            resource_type="workspace",
            resource_id=source_workspace_id,
            owner_user_id=source_owner_user_id,
            actor_user_id=identity.recipient_user_id,
            share_id=identity.share_id,
            metadata={
                **metadata,
                "target_workspace_id": identity.target_workspace_id,
            },
        )
    except Exception as exc:  # noqa: BLE001 - audit is best effort after durable CAS
        logger.bind(exception_type=type(exc).__name__).warning(
            "Shared Workspace clone audit emission failed"
        )


async def _audit_clone_succeeded(
    identity: _CloneIdentity,
    *,
    runtime: SharedWorkspaceCloneRuntime,
    result: Mapping[str, Any],
) -> None:
    outcome = result.get("outcome")
    if outcome not in {"complete", "partial"}:
        outcome = "complete"
    await _audit_clone_transition(
        identity,
        runtime=runtime,
        event_type=SHARE_CLONED,
        metadata={
            "operation_id": identity.operation_id,
            "outcome": outcome,
            "counts": _bounded_clone_counts(result),
        },
    )


async def _audit_clone_failed(
    identity: _CloneIdentity,
    *,
    runtime: SharedWorkspaceCloneRuntime,
    failure_code: Any,
) -> None:
    await _audit_clone_transition(
        identity,
        runtime=runtime,
        event_type=SHARE_CLONE_FAILED,
        metadata={
            "operation_id": identity.operation_id,
            "failure_code": _bounded_failure_code(failure_code),
            "cleanup_state": "complete",
        },
    )


async def _compensate_completed_clone(
    identity: _CloneIdentity,
    *,
    runtime: SharedWorkspaceCloneRuntime,
    current_result: Mapping[str, Any],
    failure_code: str,
) -> CloneFinalizationOutcome:
    aborting = build_clone_publication_abort(
        failure_code,
        cleanup_state="pending",
    )
    aborting_outcome = await _patch_completed_clone_result(
        identity,
        runtime=runtime,
        current_result=current_result,
        replacement_result=aborting,
    )
    if not _terminal_patch_succeeded(aborting_outcome):
        return CloneFinalizationOutcome.DEFERRED
    cleaned = await _run_target_resource_action(
        identity,
        runtime=runtime,
        action=lambda target_chacha, target_media: _cleanup_clone_resources(
            identity,
            target_chacha=target_chacha,
            target_media=target_media,
        ),
    )
    if not cleaned:
        return CloneFinalizationOutcome.DEFERRED
    replacement = build_clone_publication_abort(
        failure_code,
        cleanup_state="complete",
    )
    aborted_outcome = await _patch_completed_clone_result(
        identity,
        runtime=runtime,
        current_result=aborting,
        replacement_result=replacement,
    )
    if not _terminal_patch_succeeded(aborted_outcome):
        return CloneFinalizationOutcome.DEFERRED
    if aborted_outcome is TerminalOperationResultPatchOutcome.APPLIED:
        await _audit_clone_failed(
            identity,
            runtime=runtime,
            failure_code=failure_code,
        )
    return CloneFinalizationOutcome.COMPENSATED


async def finalize_shared_workspace_clone(
    job: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    runtime: SharedWorkspaceCloneRuntime,
) -> CloneFinalizationOutcome:
    """Publish exact operation-owned media before exposing the completed Workspace."""

    identity = _validate_job_identity(job)
    try:
        stored = await asyncio.to_thread(
            runtime.jobs.get_job_or_archived_by_uuid,
            identity.operation_id,
            domain=CLONE_DOMAIN,
            owner_user_id=str(identity.recipient_user_id),
        )
        if not isinstance(stored, Mapping) or not _same_job_identity(stored, job):
            return CloneFinalizationOutcome.DEFERRED
        stored_identity = _validate_job_identity(stored, allowed_statuses={"completed"})
        if stored_identity != identity:
            return CloneFinalizationOutcome.DEFERRED
        expected_result = _mapping(result)
        stored_result = _mapping(stored.get("result"))
        if not hmac.compare_digest(
            terminal_operation_result_fingerprint(stored_result),
            terminal_operation_result_fingerprint(expected_result),
        ):
            return CloneFinalizationOutcome.DEFERRED
        state, clone_result, publication_abort = _classify_completed_result(
            stored_result,
            identity,
        )
        if state in {"aborting", "aborted"}:
            if publication_abort is None:
                return CloneFinalizationOutcome.DEFERRED
            cleaned = await _run_target_resource_action(
                identity,
                runtime=runtime,
                action=lambda target_chacha, target_media: _cleanup_clone_resources(
                    identity,
                    target_chacha=target_chacha,
                    target_media=target_media,
                ),
            )
            if not cleaned:
                return CloneFinalizationOutcome.DEFERRED
            if state == "aborting":
                completed_abort = build_clone_publication_abort(
                    publication_abort[0],
                    cleanup_state="complete",
                )
                aborted_outcome = await _patch_completed_clone_result(
                    identity,
                    runtime=runtime,
                    current_result=stored_result,
                    replacement_result=completed_abort,
                )
                if not _terminal_patch_succeeded(aborted_outcome):
                    return CloneFinalizationOutcome.DEFERRED
                if aborted_outcome is TerminalOperationResultPatchOutcome.APPLIED:
                    await _audit_clone_failed(
                        identity,
                        runtime=runtime,
                        failure_code=publication_abort[0],
                    )
            return CloneFinalizationOutcome.COMPENSATED
        if clone_result is None:
            return CloneFinalizationOutcome.DEFERRED
        if state == "published":
            verified = await _run_target_resource_action(
                identity,
                runtime=runtime,
                action=lambda target_chacha, target_media: _published_clone_resources_are_exact(
                    identity,
                    target_chacha=target_chacha,
                    target_media=target_media,
                ),
            )
            return CloneFinalizationOutcome.PUBLISHED if verified else CloneFinalizationOutcome.DEFERRED
        checkpoint = build_clone_publication_checkpoint(clone_result)
        if state == "unconfirmed":
            try:
                await _authorize_clone(identity, runtime=runtime)
            except SharedWorkspaceCloneJobError as exc:
                if exc.failure_code in {
                    "clone_access_revoked",
                    "clone_permission_removed",
                }:
                    return await _compensate_completed_clone(
                        identity,
                        runtime=runtime,
                        current_result=stored_result,
                        failure_code=exc.failure_code,
                    )
                return CloneFinalizationOutcome.DEFERRED
            checkpoint_outcome = await _patch_completed_clone_result(
                identity,
                runtime=runtime,
                current_result=stored_result,
                replacement_result=checkpoint,
            )
            if not _terminal_patch_succeeded(checkpoint_outcome):
                return CloneFinalizationOutcome.DEFERRED
        published = await _run_target_resource_action(
            identity,
            runtime=runtime,
            action=lambda target_chacha, target_media: _publish_clone_resources(
                identity,
                target_chacha=target_chacha,
                target_media=target_media,
            ),
        )
        if not published:
            return CloneFinalizationOutcome.DEFERRED
        confirmed_result = {**clone_result, "publication_confirmed": True}
        confirmation_outcome = await _patch_completed_clone_result(
            identity,
            runtime=runtime,
            current_result=checkpoint,
            replacement_result=confirmed_result,
        )
        if not _terminal_patch_succeeded(confirmation_outcome):
            return CloneFinalizationOutcome.DEFERRED
        if confirmation_outcome is TerminalOperationResultPatchOutcome.APPLIED:
            await _audit_clone_succeeded(
                identity,
                runtime=runtime,
                result=confirmed_result,
            )
        return CloneFinalizationOutcome.PUBLISHED
    except asyncio.CancelledError:
        raise
    except SharedWorkspaceCloneJobError:
        raise
    except Exception as exc:  # noqa: BLE001 - reconciliation defers uncertain state
        logger.bind(exception_type=type(exc).__name__).warning("Shared Workspace clone publication deferred")
        return CloneFinalizationOutcome.DEFERRED


def _cleanup_clone_resources(
    identity: _CloneIdentity,
    *,
    target_chacha: Any,
    target_media: Any,
) -> bool:
    references = target_media.list_operation_owned_clone_media(
        operation_id=identity.operation_id,
        limit=_RECONCILIATION_LIMIT,
    )
    for reference in references:
        changed = target_media.delete_operation_owned_clone_media(
            operation_id=identity.operation_id,
            source_identity=reference.source_identity,
            expected_content_hash=reference.expected_content_hash,
        )
        if changed != 1:
            return False
    if len(references) == _RECONCILIATION_LIMIT:
        remaining = target_media.list_operation_owned_clone_media(
            operation_id=identity.operation_id,
            limit=_RECONCILIATION_LIMIT,
        )
        if remaining:
            return False
    discarded = target_chacha.discard_clone_target(
        workspace_id=identity.target_workspace_id,
        operation_id=identity.operation_id,
    )
    if discarded:
        return target_chacha.get_workspace(identity.target_workspace_id) is None
    pending = target_chacha.list_clone_targets_for_reconciliation(
        operation_ids=[identity.operation_id],
        limit=2,
    )
    if pending:
        return False
    return target_chacha.get_workspace(identity.target_workspace_id) is None


async def cleanup_shared_workspace_clone(
    job: Mapping[str, Any],
    *,
    runtime: SharedWorkspaceCloneRuntime,
    patch_terminal_result: bool,
    allowed_statuses: set[str] | frozenset[str] | None = None,
) -> bool:
    """Delete only exact operation-owned staged resources and optionally record proof."""

    statuses = allowed_statuses or frozenset(_TERMINAL_FAILURE_STATUSES)
    identity = _validate_job_identity(job)
    try:
        stored = await asyncio.to_thread(
            runtime.jobs.get_job_or_archived_by_uuid,
            identity.operation_id,
            domain=CLONE_DOMAIN,
            owner_user_id=str(identity.recipient_user_id),
        )
        if not isinstance(stored, Mapping) or not _same_job_identity(stored, job):
            return False
        _validate_job_identity(stored, allowed_statuses=statuses)
        target_chacha = await runtime.load_chacha_db(identity.recipient_user_id)
        if target_chacha is None:
            return False

        def _cleanup() -> bool:
            try:
                with runtime.media_session_factory(identity.recipient_user_id) as target_media:
                    return _cleanup_clone_resources(
                        identity,
                        target_chacha=target_chacha,
                        target_media=target_media,
                    )
            finally:
                _close_chacha_thread_connection(target_chacha)

        if not await asyncio.to_thread(_cleanup):
            return False
        if not patch_terminal_result:
            return True
        current_result = _mapping(stored.get("result"))
        replacement_result = {
            "schema_version": CLONE_SCHEMA_VERSION,
            "cleanup_state": "complete",
        }
        command = TerminalOperationResultPatchCommand(
            job_uuid=identity.operation_id,
            owner_user_id=str(identity.recipient_user_id),
            domain=CLONE_DOMAIN,
            queue=CLONE_QUEUE,
            job_type=CLONE_JOB_TYPE,
            operation_scope=identity.operation_scope,
            allowed_statuses=tuple(
                status
                for status in (
                    "completed",
                    "failed",
                    "cancelled",
                    "quarantined",
                )
                if status in statuses
            ),
            expected_result_fingerprint=terminal_operation_result_fingerprint(current_result),
            replacement_result=replacement_result,
        )
        outcome = await asyncio.to_thread(
            runtime.jobs.patch_terminal_operation_result,
            command,
        )
        if not _terminal_patch_succeeded(outcome):
            return False
        if outcome is TerminalOperationResultPatchOutcome.APPLIED:
            await _audit_clone_failed(
                identity,
                runtime=runtime,
                failure_code=stored.get("error_code"),
            )
        return True
    except asyncio.CancelledError:
        raise
    except SharedWorkspaceCloneJobError:
        return False
    except Exception as exc:  # noqa: BLE001 - reconciliation defers uncertain state
        logger.bind(exception_type=type(exc).__name__).warning("Shared Workspace clone cleanup deferred")
        return False


def _cursor_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        return datetime.fromisoformat(normalized)
    raise ValueError("reconciliation cursor timestamp is malformed")


def _next_active_reconciliation_cursor(
    rows: list[Mapping[str, Any]],
    *,
    page_size: int,
) -> _ActiveReconciliationCursor | None:
    if len(rows) < page_size:
        return None
    try:
        last = rows[-1]
        return _ActiveReconciliationCursor(
            created_before=_cursor_datetime(last.get("created_at")),
            before_id=_positive_integer(last.get("id"), "job id"),
        )
    except (IndexError, TypeError, ValueError, AttributeError):
        logger.warning("Shared Workspace clone active reconciliation cursor was malformed")
        return None


def _next_archive_reconciliation_cursor(
    rows: list[Mapping[str, Any]],
    *,
    page_size: int,
) -> _ArchiveReconciliationCursor | None:
    if len(rows) < page_size:
        return None
    try:
        last = rows[-1]
        before_uuid = last.get("_archive_cursor_uuid")
        if not isinstance(before_uuid, str):
            raise ValueError("archive cursor UUID is malformed")
        return _ArchiveReconciliationCursor(
            created_before=_cursor_datetime(last.get("_archive_cursor_created_at")),
            before_id=_positive_integer(last.get("id"), "archived job id"),
            before_uuid=before_uuid,
            before_archive_locator=_positive_integer(
                last.get("_archive_locator"),
                "archive locator",
            ),
        )
    except (IndexError, TypeError, ValueError, AttributeError):
        logger.warning("Shared Workspace clone archive reconciliation cursor was malformed")
        return None


async def reconcile_shared_workspace_clone_jobs(
    *,
    jobs: Any,
    runtime: SharedWorkspaceCloneRuntime | None = None,
    limit: int = _RECONCILIATION_LIMIT,
) -> dict[str, int]:
    """Repair a bounded combined active/archive set without exposing partial clones."""

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if runtime is None:
        runtime = await _build_default_runtime(jobs=jobs)
    summary = {
        "scanned": 0,
        "published": 0,
        "cleaned": 0,
        "deferred": 0,
        "invalid": 0,
    }
    await asyncio.to_thread(
        jobs.integrity_sweep,
        fix=True,
        domain=CLONE_DOMAIN,
        queue=CLONE_QUEUE,
        job_type=CLONE_JOB_TYPE,
    )
    active_budget = (limit + 1) // 2
    archive_budget = limit - active_budget
    active_query: dict[str, Any] = {
        "domain": CLONE_DOMAIN,
        "queue": CLONE_QUEUE,
        "job_type": CLONE_JOB_TYPE,
        "limit": active_budget,
    }
    if runtime.reconciliation.active is not None:
        active_query.update(
            created_before=runtime.reconciliation.active.created_before,
            before_id=runtime.reconciliation.active.before_id,
        )
    active = await asyncio.to_thread(jobs.list_jobs, **active_query)
    runtime.reconciliation.active = _next_active_reconciliation_cursor(
        active,
        page_size=active_budget,
    )
    archived: list[dict[str, Any]] = []
    if archive_budget:
        archive_query: dict[str, Any] = {
            "domain": CLONE_DOMAIN,
            "queue": CLONE_QUEUE,
            "job_type": CLONE_JOB_TYPE,
            "fail_on_decryption_error": True,
            "limit": archive_budget,
        }
        if runtime.reconciliation.archived is not None:
            archive_query.update(
                created_before=runtime.reconciliation.archived.created_before,
                before_id=runtime.reconciliation.archived.before_id,
                before_uuid=runtime.reconciliation.archived.before_uuid,
                before_archive_locator=(runtime.reconciliation.archived.before_archive_locator),
            )
        archived = await asyncio.to_thread(jobs.list_archived_jobs, **archive_query)
        runtime.reconciliation.archived = _next_archive_reconciliation_cursor(
            archived,
            page_size=archive_budget,
        )
    for job in [*active, *archived][:limit]:
        summary["scanned"] += 1
        status = str(job.get("status") or "")
        if status not in _TERMINAL_STATUSES:
            continue
        try:
            _validate_job_identity(job, allowed_statuses=_TERMINAL_STATUSES)
            if status == "completed":
                outcome = await finalize_shared_workspace_clone(
                    job,
                    _mapping(job.get("result")),
                    runtime=runtime,
                )
                if outcome is CloneFinalizationOutcome.PUBLISHED:
                    summary["published"] += 1
                elif outcome is CloneFinalizationOutcome.COMPENSATED:
                    summary["cleaned"] += 1
                else:
                    summary["deferred"] += 1
            elif await cleanup_shared_workspace_clone(
                job,
                runtime=runtime,
                patch_terminal_result=True,
            ):
                summary["cleaned"] += 1
            else:
                summary["deferred"] += 1
        except SharedWorkspaceCloneJobError:
            summary["invalid"] += 1
    return summary


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_positive_float(value: Any, default: float, *, minimum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed) or parsed < minimum:
        return float(default)
    return parsed


async def _build_default_runtime(
    *,
    jobs: Any,
    stop_event: asyncio.Event | None = None,
) -> SharedWorkspaceCloneRuntime:
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
        get_chacha_db_for_owner,
    )
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import (
        managed_media_db_for_owner,
    )
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import (
        SharedWorkspaceRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo

    pool = await get_db_pool()
    share_repo = SharedWorkspaceRepo(db_pool=pool)
    access_service = SharedWorkspaceAccessService(
        share_repo,
        AuthnzUsersRepo(db_pool=pool),
        get_chacha_db_for_owner,
    )
    return SharedWorkspaceCloneRuntime(
        jobs=jobs,
        access_service=access_service,
        load_chacha_db=get_chacha_db_for_owner,
        media_session_factory=managed_media_db_for_owner,
        share_repo=share_repo,
        audit_service=ShareAuditService(),
        vector_retrieval_configured=_truthy(os.getenv("SHARED_WORKSPACE_CLONE_VECTOR_RETRIEVAL_CONFIGURED")),
        authorization_timeout_seconds=_coerce_positive_float(
            os.getenv("SHARED_WORKSPACE_CLONE_AUTHORIZATION_TIMEOUT_SECONDS"),
            5.0,
            minimum=0.1,
        ),
        stop_event=stop_event,
    )


def _build_worker_config() -> WorkerConfig:
    worker_id = (
        os.getenv("SHARED_WORKSPACE_CLONE_JOBS_WORKER_ID") or f"shared-workspace-clone-worker-{os.getpid()}"
    ).strip()
    return WorkerConfig(
        domain=CLONE_DOMAIN,
        queue=CLONE_QUEUE,
        worker_id=worker_id,
        lease_seconds=coerce_int(
            os.getenv("SHARED_WORKSPACE_CLONE_JOBS_LEASE_SECONDS"),
            120,
        ),
        renew_jitter_seconds=coerce_int(
            os.getenv("SHARED_WORKSPACE_CLONE_JOBS_RENEW_JITTER_SECONDS"),
            5,
        ),
        renew_threshold_seconds=coerce_int(
            os.getenv("SHARED_WORKSPACE_CLONE_JOBS_RENEW_THRESHOLD_SECONDS"),
            15,
        ),
        backoff_base_seconds=coerce_int(
            os.getenv("SHARED_WORKSPACE_CLONE_JOBS_BACKOFF_BASE_SECONDS"),
            2,
        ),
        backoff_max_seconds=coerce_int(
            os.getenv("SHARED_WORKSPACE_CLONE_JOBS_BACKOFF_MAX_SECONDS"),
            30,
        ),
        retry_on_exception=False,
        completion_callback_timeout_seconds=_coerce_positive_float(
            os.getenv("SHARED_WORKSPACE_CLONE_COMPLETION_TIMEOUT_SECONDS"),
            30.0,
            minimum=0.1,
        ),
    )


async def _wait_interruptibly(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=max(0.1, seconds))
    except TimeoutError:
        return


async def run_shared_workspace_clone_jobs_worker(
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run WorkerSDK and hard-exit reconciliation under one stop signal."""

    shared_stop = stop_event or asyncio.Event()
    if shared_stop.is_set():
        return
    jobs = jobs_manager_from_env()
    runtime = await _build_default_runtime(jobs=jobs, stop_event=shared_stop)
    sdk = WorkerSDK(jobs, _build_worker_config())

    async def _handler(job: dict[str, Any]) -> dict[str, Any]:
        return await handle_shared_workspace_clone_job(job, runtime=runtime)

    async def _completed(job: dict[str, Any], result: dict[str, Any]) -> None:
        await finalize_shared_workspace_clone(job, result, runtime=runtime)

    async def _completion_rejected(
        job: dict[str, Any],
        _result: dict[str, Any],
    ) -> None:
        await cleanup_shared_workspace_clone(
            job,
            runtime=runtime,
            patch_terminal_result=False,
            allowed_statuses={"queued", "processing", *_TERMINAL_FAILURE_STATUSES},
        )

    async def _failed(job: dict[str, Any], _exc: Exception) -> None:
        await cleanup_shared_workspace_clone(
            job,
            runtime=runtime,
            patch_terminal_result=True,
        )

    async def _watch_stop() -> None:
        await shared_stop.wait()
        sdk.stop()

    async def _reconcile_loop() -> None:
        interval = _coerce_positive_float(
            os.getenv("SHARED_WORKSPACE_CLONE_RECONCILE_SECONDS"),
            30.0,
            minimum=1.0,
        )
        while not shared_stop.is_set():
            try:
                await reconcile_shared_workspace_clone_jobs(
                    jobs=jobs,
                    runtime=runtime,
                    limit=_RECONCILIATION_LIMIT,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - loop remains available
                logger.bind(exception_type=type(exc).__name__).warning(
                    "Shared Workspace clone reconciliation pass failed"
                )
            await _wait_interruptibly(shared_stop, interval)

    stop_task = asyncio.create_task(
        _watch_stop(),
        name="shared_workspace_clone_stop_waiter",
    )
    worker_task = asyncio.create_task(
        sdk.run(
            handler=_handler,
            progress_cb=runtime.progress.snapshot,
            job_type=CLONE_JOB_TYPE,
            on_completed=_completed,
            on_completion_rejected=_completion_rejected,
            on_failed=_failed,
        ),
        name="shared_workspace_clone_worker",
    )
    reconciliation_task = asyncio.create_task(
        _reconcile_loop(),
        name="shared_workspace_clone_reconciler",
    )
    logger.info("Shared Workspace clone Jobs worker starting")
    try:
        await asyncio.gather(worker_task, reconciliation_task)
    finally:
        sdk.stop()
        shared_stop.set()
        for task in (stop_task, worker_task, reconciliation_task):
            if not task.done():
                task.cancel()
        for task in (stop_task, worker_task, reconciliation_task):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if runtime.audit_service is not None:
            try:
                await runtime.audit_service.stop()
            except Exception as exc:  # noqa: BLE001 - shutdown remains best effort
                logger.bind(exception_type=type(exc).__name__).warning(
                    "Shared Workspace clone audit shutdown failed"
                )
        logger.info("Shared Workspace clone Jobs worker stopped")


__all__ = [
    "CloneFinalizationOutcome",
    "CloneProgressState",
    "SharedWorkspaceCloneJobError",
    "SharedWorkspaceCloneRuntime",
    "cleanup_shared_workspace_clone",
    "finalize_shared_workspace_clone",
    "handle_shared_workspace_clone_job",
    "reconcile_shared_workspace_clone_jobs",
    "run_shared_workspace_clone_jobs_worker",
]


if __name__ == "__main__":  # pragma: no cover - standalone operator entry point
    asyncio.run(run_shared_workspace_clone_jobs_worker())
