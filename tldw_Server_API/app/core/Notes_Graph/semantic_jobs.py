"""Receipt-backed Jobs contracts for Notes semantic-index operations."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from uuid import UUID

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    CreateJobCommand,
    IdempotentOperationCommand,
    IdempotentOperationConflict,
    IdempotentOperationConflictReason,
    IdempotentOperationDisposition,
    IdempotentOperationUnavailableError,
)

from .semantic_settings import DEFAULT_SEMANTIC_INDEX_SETTINGS, SemanticIndexSettings

JOB_DOMAIN = "notes"
JOB_QUEUE = "semantic-index"
JOB_TYPE = "note_semantic_index"
JOB_SCHEMA_VERSION = 1
JOB_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "configuration_revision",
        "generation_id",
        "mode",
    }
)

_JOB_MODES = frozenset({"build", "rebuild", "retry_failed", "maintain", "delete"})
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "quarantined"})
_RESULT_KEYS = frozenset(
    {
        "state",
        "indexed_notes",
        "excluded_notes",
        "failed_notes",
        "published_chunks",
        "cleanup_complete",
        "error_code",
    }
)
_QUOTA_ERROR_MARKERS = (
    "jobs_owner_scope_admission_limit_exceeded",
    "maximum concurrent job limit",
    "quota",
)


class SemanticJobsError(RuntimeError):
    """Stable, content-free semantic Jobs failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SemanticJobCancelled(SemanticJobsError):
    """Raised when cancellation fences semantic provider work."""

    def __init__(self) -> None:
        super().__init__("notes_semantic_run_cancelled")


@dataclass(frozen=True, slots=True)
class SemanticJobCommand:
    """Opaque identity required to execute one semantic operation."""

    dataset_id: str
    configuration_revision: int
    mode: str
    generation_id: str | None = None

    def __post_init__(self) -> None:
        _bounded_identity(self.dataset_id, "dataset_id")
        if (
            isinstance(self.configuration_revision, bool)
            or not isinstance(self.configuration_revision, int)
            or self.configuration_revision < 0
        ):
            raise ValueError("notes_semantic_configuration_revision_invalid")
        if self.mode not in _JOB_MODES:
            raise ValueError("notes_semantic_job_mode_invalid")
        if self.generation_id is not None:
            _bounded_identity(self.generation_id, "generation_id")

    def payload(self) -> dict[str, object]:
        """Return the exact content-free payload admitted to Jobs."""

        return {
            "schema_version": JOB_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "configuration_revision": self.configuration_revision,
            "generation_id": self.generation_id,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class SemanticJobAdmission:
    """Domain view of one durable Jobs admission."""

    run_id: str
    job: dict[str, Any]
    disposition: str


class SemanticJobs(Protocol):
    """Narrow Jobs surface used by the semantic coordinator."""

    def replay_idempotent_operation(self, command: IdempotentOperationCommand) -> Any: ...

    def admit_idempotent_operation(self, command: IdempotentOperationCommand) -> Any: ...

    def get_job_or_archived_by_uuid(self, job_uuid: str, **kwargs: Any) -> dict[str, Any] | None: ...

    def cancel_job(self, job_id: int, **kwargs: Any) -> bool: ...


def _bounded_identity(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"notes_semantic_{field}_invalid")
    normalized = value.strip()
    if not normalized or normalized != value or len(normalized.encode("utf-8")) > 256:
        raise ValueError(f"notes_semantic_{field}_invalid")
    return normalized


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_fingerprint(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _digest(encoded)


def _idempotency_digest(value: str) -> str:
    normalized = _bounded_identity(value, "idempotency_key")
    return _digest(normalized.encode("utf-8"))


def _operation_scope(command: SemanticJobCommand) -> str:
    raw = f"{command.dataset_id}\0{command.configuration_revision}".encode()
    return f"notes-semantic:{_digest(raw)}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SemanticJobCoordinator:
    """Admit, resolve, and cancel owner-scoped semantic root Jobs."""

    def __init__(
        self,
        *,
        jobs: SemanticJobs,
        owner_user_id: str,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._jobs = jobs
        self._owner_user_id = _bounded_identity(owner_user_id, "owner_user_id")
        self._clock = clock

    def _operation(
        self,
        command: SemanticJobCommand,
        *,
        idempotency_key: str,
        request_identity: Mapping[str, object] | None,
    ) -> IdempotentOperationCommand:
        payload = command.payload()
        scope = _operation_scope(command)
        return IdempotentOperationCommand(
            job=CreateJobCommand(
                domain=JOB_DOMAIN,
                queue=JOB_QUEUE,
                job_type=JOB_TYPE,
                payload=payload,
                owner_user_id=self._owner_user_id,
                priority=5,
                max_retries=0,
                batch_group=scope,
            ),
            key_digest=_idempotency_digest(idempotency_key),
            request_fingerprint=_canonical_fingerprint(
                payload if request_identity is None else request_identity
            ),
            operation_scope=scope,
            receipt_expires_at=self._clock() + timedelta(days=31),
        )

    def _admission(self, admission: Any) -> SemanticJobAdmission:
        admitted_job = dict(admission.job)
        job = self._jobs.get_job_or_archived_by_uuid(
            str(admitted_job.get("uuid") or ""),
            domain=JOB_DOMAIN,
            owner_user_id=self._owner_user_id,
        )
        if job is None:
            raise SemanticJobsError("notes_semantic_jobs_unavailable")
        self._validate_authoritative_job(job)
        disposition = admission.disposition
        if isinstance(disposition, IdempotentOperationDisposition):
            disposition_value = disposition.value
        else:
            disposition_value = str(disposition)
        return SemanticJobAdmission(
            run_id=str(job["uuid"]),
            job=job,
            disposition=disposition_value,
        )

    def replay(
        self,
        command: SemanticJobCommand,
        *,
        idempotency_key: str,
        request_identity: Mapping[str, object] | None = None,
    ) -> SemanticJobAdmission | None:
        """Resolve an exact durable receipt without admitting new work."""

        operation = self._operation(
            command,
            idempotency_key=idempotency_key,
            request_identity=request_identity,
        )
        try:
            admission = self._jobs.replay_idempotent_operation(operation)
        except IdempotentOperationConflict as exc:
            raise SemanticJobsError("notes_semantic_idempotency_conflict") from exc
        except (
            IdempotentOperationUnavailableError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            raise SemanticJobsError("notes_semantic_jobs_unavailable") from exc
        return None if admission is None else self._admission(admission)

    def admit(
        self,
        command: SemanticJobCommand,
        *,
        idempotency_key: str,
        request_identity: Mapping[str, object] | None = None,
    ) -> SemanticJobAdmission:
        """Atomically create, replay, or converge one semantic writer."""

        operation = self._operation(
            command,
            idempotency_key=idempotency_key,
            request_identity=request_identity,
        )
        try:
            admission = self._jobs.admit_idempotent_operation(operation)
        except IdempotentOperationConflict as exc:
            if exc.reason is IdempotentOperationConflictReason.KEY_REUSED:
                code = "notes_semantic_idempotency_conflict"
            else:
                code = "notes_semantic_writer_conflict"
            raise SemanticJobsError(code) from exc
        except ValueError as exc:
            message = str(exc).lower()
            code = (
                "notes_semantic_quota_exceeded"
                if any(marker in message for marker in _QUOTA_ERROR_MARKERS)
                else "notes_semantic_jobs_unavailable"
            )
            raise SemanticJobsError(code) from exc
        except (IdempotentOperationUnavailableError, OSError, RuntimeError) as exc:
            raise SemanticJobsError("notes_semantic_jobs_unavailable") from exc
        return self._admission(admission)

    def get_job_for_run(self, run_id: str) -> dict[str, Any] | None:
        """Return an exact owner-scoped semantic Job without foreign disclosure."""

        try:
            normalized = str(UUID(str(run_id)))
        except (TypeError, ValueError, AttributeError):
            return None
        job = self._jobs.get_job_or_archived_by_uuid(
            normalized,
            domain=JOB_DOMAIN,
            owner_user_id=self._owner_user_id,
        )
        if job is None:
            return None
        try:
            self._validate_authoritative_job(job)
        except SemanticJobsError:
            return None
        return job

    def cancel(self, run_id: str, *, expected_revision: int) -> dict[str, Any]:
        """Cancel one exact semantic Job after configuration-revision fencing."""

        job = self.get_job_for_run(run_id)
        if job is None:
            raise SemanticJobsError("notes_semantic_run_not_found")
        payload = job["payload"]
        if payload["configuration_revision"] != expected_revision:
            raise SemanticJobsError("notes_semantic_run_revision_conflict")
        if str(job.get("status")) not in _TERMINAL_STATUSES:
            self._jobs.cancel_job(
                int(job["id"]),
                reason="requested",
                expected_uuid=str(job["uuid"]),
                expected_domain=JOB_DOMAIN,
                expected_job_type=JOB_TYPE,
                cascade_dependents=False,
            )
        refreshed = self.get_job_for_run(run_id)
        if refreshed is None:
            raise SemanticJobsError("notes_semantic_run_not_found")
        return refreshed

    def _validate_authoritative_job(self, job: Mapping[str, Any]) -> None:
        if (
            job.get("owner_user_id") != self._owner_user_id
            or job.get("domain") != JOB_DOMAIN
            or job.get("queue") != JOB_QUEUE
            or job.get("job_type") != JOB_TYPE
        ):
            raise SemanticJobsError("notes_semantic_job_authority_invalid")
        _validated_payload(job.get("payload"))


class SemanticRuntime(Protocol):
    """Pinned production runtime resolved from owner-side authority."""

    async def recover(self, **kwargs: Any) -> dict[str, Any] | None: ...

    async def execute(self, **kwargs: Any) -> dict[str, Any]: ...


CancellationCheck = Callable[[], bool | Awaitable[bool]]
SemanticRuntimeFactory = Callable[..., SemanticRuntime | Awaitable[SemanticRuntime]]


def _validated_payload(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != JOB_PAYLOAD_KEYS:
        raise SemanticJobsError("notes_semantic_job_payload_invalid")
    if raw.get("schema_version") != JOB_SCHEMA_VERSION:
        raise SemanticJobsError("notes_semantic_job_payload_invalid")
    try:
        return SemanticJobCommand(
            dataset_id=raw.get("dataset_id"),  # type: ignore[arg-type]
            configuration_revision=raw.get("configuration_revision"),  # type: ignore[arg-type]
            generation_id=raw.get("generation_id"),  # type: ignore[arg-type]
            mode=raw.get("mode"),  # type: ignore[arg-type]
        ).payload()  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise SemanticJobsError("notes_semantic_job_payload_invalid") from exc


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _is_cancelled(check: CancellationCheck) -> bool:
    return bool(await _resolve(check()))


def _validated_result(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != _RESULT_KEYS:
        raise SemanticJobsError("notes_semantic_job_result_invalid")
    result = dict(raw)
    for key in (
        "indexed_notes",
        "excluded_notes",
        "failed_notes",
        "published_chunks",
    ):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SemanticJobsError("notes_semantic_job_result_invalid")
    if type(result["cleanup_complete"]) is not bool:
        raise SemanticJobsError("notes_semantic_job_result_invalid")
    if result["error_code"] is not None and not isinstance(result["error_code"], str):
        raise SemanticJobsError("notes_semantic_job_result_invalid")
    return result


class SemanticJobHandler:
    """Validate Jobs authority and execute one bounded pinned runtime."""

    def __init__(
        self,
        *,
        runtime_factory: SemanticRuntimeFactory,
        settings: SemanticIndexSettings = DEFAULT_SEMANTIC_INDEX_SETTINGS,
    ) -> None:
        self._runtime_factory = runtime_factory
        self._settings = settings

    async def handle(
        self,
        job: Mapping[str, Any],
        *,
        cancellation_requested: CancellationCheck,
    ) -> dict[str, Any]:
        """Recover or execute an exact semantic operation without fallback."""

        if (
            job.get("domain") != JOB_DOMAIN
            or job.get("queue") != JOB_QUEUE
            or job.get("job_type") != JOB_TYPE
        ):
            raise SemanticJobsError("notes_semantic_job_authority_invalid")
        owner_user_id = _bounded_identity(job.get("owner_user_id"), "owner_user_id")
        try:
            root_job_id = str(UUID(str(job.get("uuid"))))
        except (TypeError, ValueError, AttributeError) as exc:
            raise SemanticJobsError("notes_semantic_job_authority_invalid") from exc
        payload = _validated_payload(job.get("payload"))

        if await _is_cancelled(cancellation_requested):
            raise SemanticJobCancelled()
        runtime = await _resolve(
            self._runtime_factory(
                owner_user_id=owner_user_id,
                dataset_id=payload["dataset_id"],
                configuration_revision=payload["configuration_revision"],
                generation_id=payload["generation_id"],
                root_job_id=root_job_id,
                mode=payload["mode"],
            )
        )
        recovered = await runtime.recover(
            mode=payload["mode"],
            payload=payload,
            root_job_id=root_job_id,
        )
        if recovered is not None:
            return _validated_result(recovered)
        if await _is_cancelled(cancellation_requested):
            raise SemanticJobCancelled()
        result = await runtime.execute(
            mode=payload["mode"],
            payload=payload,
            root_job_id=root_job_id,
            max_batch_retries=self._settings.max_retries,
            cancellation_requested=cancellation_requested,
        )
        return _validated_result(result)


__all__ = [
    "JOB_DOMAIN",
    "JOB_PAYLOAD_KEYS",
    "JOB_QUEUE",
    "JOB_TYPE",
    "SemanticJobAdmission",
    "SemanticJobCancelled",
    "SemanticJobCommand",
    "SemanticJobCoordinator",
    "SemanticJobHandler",
    "SemanticJobsError",
]
