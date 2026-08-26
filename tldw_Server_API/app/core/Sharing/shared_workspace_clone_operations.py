"""Canonical request identity and public projection for shared Workspace clones."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.shared_workspace_recipient_schemas import (
    SharedWorkspaceCloneError,
    SharedWorkspaceCloneOperationResponse,
    SharedWorkspaceCloneProgress,
    SharedWorkspaceCloneResult,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    CreateJobCommand,
    IdempotentOperationCommand,
)

CLONE_DOMAIN = "sharing"
CLONE_QUEUE = "workspace-clone"
CLONE_JOB_TYPE = "workspace_clone"
CLONE_COMMAND = "shared_workspace_clone"
CLONE_SCHEMA_VERSION = 1
CLONE_RECEIPT_RETENTION_DAYS = 31

_IDEMPOTENCY_KEY_RE = re.compile(r"[A-Za-z0-9._~-]{16,200}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
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
_TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled", "quarantined"})
_RETRYABLE_FAILURE_CODES = frozenset(
    {
        "clone_interrupted",
        "clone_persistence_failed",
        "lease_expired",
        "source_snapshot_unavailable",
    }
)
_ERROR_COPY: dict[str, tuple[str, str]] = {
    "clone_access_revoked": (
        "sharing.clone.errors.clone_access_revoked",
        "Access to the shared workspace ended before the copy completed.",
    ),
    "clone_cancelled": (
        "sharing.clone.errors.clone_cancelled",
        "The workspace copy was cancelled.",
    ),
    "clone_interrupted": (
        "sharing.clone.errors.clone_interrupted",
        "The workspace copy was interrupted. Try the copy again.",
    ),
    "clone_permission_removed": (
        "sharing.clone.errors.clone_permission_removed",
        "The owner disabled copying before the operation completed.",
    ),
    "clone_persistence_failed": (
        "sharing.clone.errors.clone_persistence_failed",
        "The workspace copy could not be saved. Try the copy again.",
    ),
    "lease_expired": (
        "sharing.clone.errors.clone_interrupted",
        "The workspace copy was interrupted. Try the copy again.",
    ),
    "source_snapshot_unavailable": (
        "sharing.clone.errors.source_snapshot_unavailable",
        "The source snapshot was unavailable. Try the copy again.",
    ),
}
_GENERIC_ERROR = (
    "sharing.clone.errors.clone_failed",
    "The workspace copy could not be completed.",
)


class CloneOperationNotFound(LookupError):
    """The requested operation is not owned by the authenticated recipient."""


class CloneOperationUnavailable(RuntimeError):
    """The persisted Job cannot be projected without guessing."""


def validate_idempotency_key(value: str) -> str:
    """Validate the exact clone idempotency header without normalization."""

    if not isinstance(value, str) or _IDEMPOTENCY_KEY_RE.fullmatch(value) is None:
        raise ValueError(
            "Idempotency-Key must contain 16-200 ASCII characters from "
            "[A-Za-z0-9._~-]"
        )
    return value


def normalize_clone_name(name: str | None) -> str | None:
    """Apply the Workspace clone-name normalization once at admission."""

    if name is None:
        return None
    if not isinstance(name, str):
        raise ValueError("name must be a string or null")
    normalized = " ".join(name.split())
    if not normalized:
        raise ValueError("name must not be blank")
    if len(normalized) > 255:
        raise ValueError("name must contain at most 255 characters")
    return normalized


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def build_clone_admission_command(
    *,
    share_id: int,
    recipient_user_id: int,
    requested_name: str | None,
    idempotency_key: str,
    now: datetime | None = None,
) -> IdempotentOperationCommand:
    """Build one bounded receipt-backed Jobs admission command."""

    if isinstance(share_id, bool) or not isinstance(share_id, int) or share_id <= 0:
        raise ValueError("share_id must be a positive integer")
    if (
        isinstance(recipient_user_id, bool)
        or not isinstance(recipient_user_id, int)
        or recipient_user_id <= 0
    ):
        raise ValueError("recipient_user_id must be a positive integer")
    key = validate_idempotency_key(idempotency_key)
    name = normalize_clone_name(requested_name)
    accepted_at = now or datetime.now(timezone.utc)
    if accepted_at.tzinfo is None or accepted_at.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    identity = {
        "schema_version": CLONE_SCHEMA_VERSION,
        "share_id": share_id,
        "recipient_user_id": recipient_user_id,
        "requested_name": name,
    }
    fingerprint = _sha256(_canonical_json(identity))
    key_digest = _sha256(f"{recipient_user_id}\0{key}".encode("ascii"))
    operation_scope = f"share:{share_id}"
    payload = {**identity, "request_fingerprint": fingerprint}
    return IdempotentOperationCommand(
        job=CreateJobCommand(
            domain=CLONE_DOMAIN,
            queue=CLONE_QUEUE,
            job_type=CLONE_JOB_TYPE,
            payload=payload,
            owner_user_id=str(recipient_user_id),
            batch_group=operation_scope,
            priority=10,
            max_retries=0,
        ),
        key_digest=key_digest,
        request_fingerprint=fingerprint,
        operation_scope=operation_scope,
        receipt_expires_at=accepted_at + timedelta(
            days=CLONE_RECEIPT_RETENTION_DAYS
        ),
    )


def target_workspace_id(operation_id: str) -> str:
    """Derive one deterministic target Workspace UUID from a durable Job UUID."""

    try:
        normalized = str(UUID(str(operation_id)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("operation_id must be a UUID") from exc
    return str(uuid5(NAMESPACE_URL, f"tldw:shared-workspace-clone:{normalized}"))


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        value = value.isoformat()
    if not isinstance(value, str) or not value or len(value) > 80:
        raise CloneOperationUnavailable("clone operation timestamp is invalid")
    return value


def _validate_job_scope(
    job: Mapping[str, Any],
    *,
    share_id: int,
    recipient_user_id: int,
) -> dict[str, Any]:
    expected_scope = f"share:{share_id}"
    payload = _mapping(job.get("payload"))
    correlation_matches = (
        str(job.get("owner_user_id") or "") == str(recipient_user_id)
        and str(job.get("domain") or "") == CLONE_DOMAIN
        and str(job.get("queue") or "") == CLONE_QUEUE
        and str(job.get("job_type") or "") == CLONE_JOB_TYPE
        and str(job.get("batch_group") or "") == expected_scope
        and payload.get("share_id") == share_id
        and payload.get("recipient_user_id") == recipient_user_id
    )
    if not correlation_matches:
        raise CloneOperationNotFound()
    if set(payload) != {
        "schema_version",
        "share_id",
        "recipient_user_id",
        "requested_name",
        "request_fingerprint",
    }:
        raise CloneOperationUnavailable("clone operation payload is malformed")
    name = payload.get("requested_name")
    fingerprint = payload.get("request_fingerprint")
    if (
        payload.get("schema_version") != CLONE_SCHEMA_VERSION
        or normalize_clone_name(name) != name
        or not isinstance(fingerprint, str)
        or _SHA256_RE.fullmatch(fingerprint) is None
    ):
        raise CloneOperationUnavailable("clone operation payload is malformed")
    return payload


def _progress(job: Mapping[str, Any], status: str) -> SharedWorkspaceCloneProgress:
    if status == "queued":
        phase = "queued"
        percent = 0
    else:
        raw_phase = job.get("progress_message")
        phase = raw_phase if isinstance(raw_phase, str) and raw_phase else "authorizing"
        if phase not in _PROGRESS_PHASES:
            raise CloneOperationUnavailable("clone progress phase is invalid")
        raw_percent = job.get("progress_percent")
        if raw_percent is None:
            percent = 1
        elif (
            isinstance(raw_percent, bool)
            or not isinstance(raw_percent, (int, float))
            or not math.isfinite(float(raw_percent))
            or not 0 <= float(raw_percent) <= 100
        ):
            raise CloneOperationUnavailable("clone progress percent is invalid")
        else:
            percent = int(round(float(raw_percent)))
    return SharedWorkspaceCloneProgress(
        phase=phase,
        percent=percent,
        message_code=f"clone_{phase}",
    )


def _cleanup_state(job: Mapping[str, Any]) -> str:
    result = _mapping(job.get("result"))
    if not result:
        return "unknown"
    if set(result) != {"schema_version", "cleanup_state"}:
        return "unknown"
    state = result.get("cleanup_state")
    if result.get("schema_version") != 1 or state not in {
        "complete",
        "pending",
        "unknown",
    }:
        return "unknown"
    return str(state)


def _failure(job: Mapping[str, Any], status: str) -> tuple[SharedWorkspaceCloneError, bool]:
    raw_code = job.get("error_code")
    if not isinstance(raw_code, str) or _SAFE_ERROR_CODE_RE.fullmatch(raw_code) is None:
        raw_code = "clone_interrupted" if status == "cancelled" else "clone_failed"
    code = raw_code if raw_code in _ERROR_COPY else "clone_failed"
    message_key, message = _ERROR_COPY.get(code, _GENERIC_ERROR)
    return (
        SharedWorkspaceCloneError(
            code=code,
            message_key=message_key,
            message=message,
            cleanup_state=_cleanup_state(job),
        ),
        code in _RETRYABLE_FAILURE_CODES,
    )


def project_clone_operation(
    job: Mapping[str, Any],
    *,
    share_id: int,
    recipient_user_id: int,
) -> SharedWorkspaceCloneOperationResponse:
    """Project one exact owner-scoped Job into the public clone envelope."""

    if not isinstance(job, Mapping):
        raise CloneOperationNotFound()
    try:
        _validate_job_scope(
            job,
            share_id=share_id,
            recipient_user_id=recipient_user_id,
        )
        operation_id = str(UUID(str(job.get("uuid") or "")))
        workspace_id = target_workspace_id(operation_id)
        started_at = _timestamp(job.get("created_at"))
        updated_at = _timestamp(job.get("updated_at") or job.get("created_at"))
        status = str(job.get("status") or "")
        progress = None
        result = None
        error = None
        retryable = False
        if status in {"queued", "processing"}:
            public_status = "queued" if status == "queued" else "running"
            progress = _progress(job, status)
        elif status == "completed":
            public_status = "succeeded"
            result = SharedWorkspaceCloneResult.model_validate(_mapping(job.get("result")))
            if not result.publication_confirmed or result.workspace_id != workspace_id:
                raise CloneOperationUnavailable("clone result is not publication-confirmed")
        elif status in _TERMINAL_FAILURE_STATUSES:
            public_status = "failed"
            error, retryable = _failure(job, status)
        else:
            raise CloneOperationUnavailable("clone Job status is unsupported")
        return SharedWorkspaceCloneOperationResponse(
            schema_version=1,
            operation_id=operation_id,
            workspace_id=workspace_id,
            command=CLONE_COMMAND,
            status=public_status,
            started_at=started_at,
            updated_at=updated_at,
            retryable=retryable,
            diagnostics={},
            poll_href=(
                f"/api/v1/sharing/shared-with-me/{share_id}/clone/{operation_id}"
            ),
            share_id=share_id,
            progress=progress,
            result=result,
            error=error,
        )
    except CloneOperationNotFound:
        raise
    except CloneOperationUnavailable:
        raise
    except (TypeError, ValueError, ValidationError) as exc:
        raise CloneOperationUnavailable("clone operation is malformed") from exc


__all__ = [
    "CLONE_COMMAND",
    "CLONE_DOMAIN",
    "CLONE_JOB_TYPE",
    "CLONE_QUEUE",
    "CloneOperationNotFound",
    "CloneOperationUnavailable",
    "build_clone_admission_command",
    "normalize_clone_name",
    "project_clone_operation",
    "target_workspace_id",
    "validate_idempotency_key",
]
