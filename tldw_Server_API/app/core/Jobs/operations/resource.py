"""Shared resource admission decisions; backend stores own locking and SQL.

This is admission state only. Task 3 lifecycle operations must take the same
head-before-exact-row locks and preserve immutable bindings when archiving.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import resource_contracts as contracts
from ..migrations import SLIDES_ARCHIVE_EXACT_FIELDS, normalize_slides_archive_projection
from ..resource_policy import resource_requirement_for
from .contracts import AdmissionRejectionReason, AdmissionResult, CreateJobCommand


@dataclass(frozen=True)
class ResourceCreation:
    """Complete immutable metadata for INSERT, never an unbound-row UPDATE."""

    binding: contracts.ResourceBinding
    settings_json: str
    settings_fingerprint: str

    def values(self) -> tuple[object, ...]:
        """Return values in RESOURCE_METADATA_FIELDS order."""
        return (
            self.binding.scope.resource,
            self.binding.authority_uuid,
            self.binding.generation,
            self.settings_json,
            self.settings_fingerprint,
        )


class ResourceAdmissionStore:
    """Small shared SQL store with fixed backend syntax and parameterized values."""

    def __init__(self, executor: Any, *, postgres: bool = False) -> None:
        self.executor = executor
        self.postgres = postgres
        self.marker = "%s" if postgres else "?"

    def execute(self, statement: str, values: tuple = ()) -> Any:
        """Execute a fixed internal statement; callers never supply SQL fragments."""
        return self.executor.execute(statement, values)

    def timestamp(self, now: datetime) -> datetime | str:
        """Match each backend's persisted UTC timestamp representation."""
        now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
        return now if self.postgres else now.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

    def scope_values(self, scope: contracts.ResourceScope) -> tuple[str, str, str]:
        """Return the single shared identity order used by all head operations."""
        return scope.domain, scope.owner_user_id, scope.resource

    @property
    def scope_where(self) -> str:
        """Render only trusted fixed column names and backend placeholders."""
        p = self.marker
        return f"domain={p} AND owner_user_id={p} AND resource={p}"

    def lock_head(self, scope: contracts.ResourceScope) -> dict[str, Any]:
        """Implemented by backend modules: caller already owns the transaction."""
        raise NotImplementedError

    def resolve_job(self, scope: contracts.ResourceScope, job_uuid: str) -> dict[str, Any]:
        """Lock exact live/archive targets after the head, rejecting ambiguity."""
        p = self.marker
        fields = ",".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS, *contracts.RESOURCE_METADATA_FIELDS))
        lock = " FOR UPDATE" if self.postgres else ""
        values = (job_uuid, scope.domain, scope.owner_user_id)
        live = self.execute(
            f"SELECT {fields}, NULL AS payload_compressed, NULL AS result_compressed, "  # nosec B608 - fixed projection/syntax; values bound.
            f"0 AS archived FROM jobs WHERE uuid={p} AND domain={p} AND owner_user_id={p}{lock}",
            values,
        ).fetchall()
        archived = self.execute(
            f"SELECT {fields}, payload_compressed, result_compressed, 1 AS archived "  # nosec B608 - fixed projection/syntax; values bound.
            f"FROM jobs_archive WHERE uuid={p} AND domain={p} AND owner_user_id={p}{lock}",
            values,
        ).fetchall()
        rows = [*live, *archived]
        if len(rows) != 1:
            raise RuntimeError("resource Job target is missing or ambiguous")
        return normalize_slides_archive_projection(rows[0])

    def active_jobs(self, scope: contracts.ResourceScope, command: CreateJobCommand) -> list[dict[str, Any]]:
        """Detect bound or legacy active candidates across all queues/methods."""
        p = self.marker
        lock = " FOR UPDATE" if self.postgres else ""
        alias_prefix = command.job_type.replace("!", "!!").replace("%", "!%").replace("_", "!_") + ".v%"
        rows = self.execute(
            f"SELECT * FROM jobs WHERE domain={p} AND owner_user_id={p} "  # nosec B608 - backend placeholders only; values bound.
            f"AND (resource={p} OR job_type={p} OR job_type LIKE {p} ESCAPE '!') "
            f"AND status IN ('queued','processing') ORDER BY id{lock}",
            (*self.scope_values(scope), command.job_type, alias_prefix),
        ).fetchall()
        active = []
        for row in rows:
            try:
                required = resource_requirement_for(row["domain"], row["job_type"])
            except ValueError:
                # Reserved but unsupported aliases cannot hide a legacy active Job.
                required = scope.resource
            if row["resource"] == scope.resource or required == scope.resource:
                active.append(dict(row))
        return active

    def has_bound_history(self, scope: contracts.ResourceScope) -> bool:
        """A missing head must not mint a new authority over retained bound Jobs."""
        row = self.execute(
            f"SELECT 1 FROM jobs WHERE {self.scope_where} "  # nosec B608 - fixed predicate/placeholders; values bound.
            f"UNION ALL SELECT 1 FROM jobs_archive WHERE {self.scope_where} LIMIT 1",
            (*self.scope_values(scope), *self.scope_values(scope)),
        ).fetchone()
        return row is not None

    def receipt(self, scope: contracts.ResourceScope, digest: str, now: datetime) -> dict[str, Any] | None:
        """Read only retained keys under the resource lock."""
        p = self.marker
        row = self.execute(
            f"SELECT * FROM job_resource_request_receipts WHERE {self.scope_where} "  # nosec B608 - fixed predicate/placeholders; values bound.
            f"AND request_key_digest={p} AND expires_at>{p}",
            (*self.scope_values(scope), digest, self.timestamp(now)),
        ).fetchone()
        return dict(row) if row else None

    def attach_receipt(self, request: contracts.ResourceAdmissionRequest, job: dict[str, Any], now: datetime) -> bool:
        """Attach a bounded 24-hour alias; existing retained aliases were read first."""
        if request.request_key is None:
            return True
        p = self.marker
        scope = self.scope_values(request.scope)
        row = self.execute(
            f"SELECT COUNT(*) AS c FROM job_resource_request_receipts WHERE {self.scope_where} "  # nosec B608 - fixed predicate/placeholders; values bound.
            f"AND job_uuid={p} AND expires_at>{p}",
            (*scope, job["uuid"], self.timestamp(now)),
        ).fetchone()
        count = row["c"] if hasattr(row, "keys") else row[0]
        if count >= contracts.RESOURCE_REQUEST_MAX_ALIASES:
            return False
        digest = hashlib.sha256(request.request_key.encode("ascii")).hexdigest()
        self.execute(
            f"DELETE FROM job_resource_request_receipts WHERE {self.scope_where} "  # nosec B608 - fixed predicate/placeholders; values bound.
            f"AND request_key_digest={p} AND expires_at<={p}",
            (*scope, digest, self.timestamp(now)),
        )
        self.execute(
            "INSERT INTO job_resource_request_receipts "  # nosec B608 - fixed columns/placeholders; values bound.
            "(domain,owner_user_id,resource,request_key_digest,settings_json,settings_fingerprint,job_uuid,expires_at) "
            f"VALUES ({','.join([p] * 8)})",
            (
                *scope,
                digest,
                job["resource_settings_json"],
                job["resource_settings_fingerprint"],
                job["uuid"],
                self.timestamp(now + timedelta(seconds=contracts.RESOURCE_REQUEST_RETENTION_SECONDS)),
            ),
        )
        return True

    def attach_created(self, request: contracts.ResourceAdmissionRequest, job: dict[str, Any], now: datetime) -> None:
        """Publish the new head and alias in the existing insert/event transaction."""
        p = self.marker
        self.execute(
            f"UPDATE job_resource_heads SET generation={p},active_job_uuid={p} WHERE {self.scope_where}",  # nosec B608 - fixed predicate/placeholders; values bound.
            (job["resource_generation"], job["uuid"], *self.scope_values(request.scope)),
        )
        if not self.attach_receipt(request, job, now):
            raise RuntimeError("resource alias limit for newly created Job")


def _validate_target(
    job: dict[str, Any], head: dict[str, Any], command: CreateJobCommand, request: contracts.ResourceAdmissionRequest
) -> None:
    """Fail closed on mismatched identity, authority, generation or settings encoding."""
    try:
        binding = contracts.ResourceBinding(request.scope, job["resource_authority_uuid"], job["resource_generation"])
        settings = json.loads(job["resource_settings_json"])
        valid = (
            job["domain"] == request.scope.domain
            and job["owner_user_id"] == request.scope.owner_user_id
            and job["resource"] == request.scope.resource
            and job["job_type"] == command.job_type
            and binding.authority_uuid == head["authority_uuid"]
            and binding.generation <= head["generation"]
            and job["resource_settings_json"] == contracts.canonical_settings_json(settings)
            and job["resource_settings_fingerprint"] == contracts.settings_fingerprint(settings)
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError("resource Job authority is invalid")


def _conflict_reference(job: dict[str, Any]) -> dict[str, Any]:
    """Project an already validated target without payload, result or raw errors."""
    return {field: job[field] for field in ("id", "uuid", "domain", "owner_user_id", "job_type", "resource")}


def lookup_resource_admission(
    store: ResourceAdmissionStore,
    *,
    command: CreateJobCommand,
    request: contracts.ResourceAdmissionRequest,
    now: datetime,
) -> contracts.ResourceAdmission | ResourceCreation:
    """Resolve retained replay/active reuse before any creation-only rejection."""
    head = store.lock_head(request.scope)
    # Validate even an empty head, without allocating a logical generation yet.
    contracts.ResourceBinding(request.scope, head["authority_uuid"], head["generation"] + 1)
    encoded = contracts.canonical_settings_json(request.canonical_settings)
    fingerprint = contracts.settings_fingerprint(request.canonical_settings)
    if request.request_key is not None:
        digest = hashlib.sha256(request.request_key.encode("ascii")).hexdigest()
        receipt = store.receipt(request.scope, digest, now)
        if receipt is not None:
            job = store.resolve_job(request.scope, receipt["job_uuid"])
            _validate_target(job, head, command, request)
            if (receipt["settings_json"], receipt["settings_fingerprint"]) != (
                job["resource_settings_json"],
                job["resource_settings_fingerprint"],
            ):
                raise RuntimeError("resource receipt authority is invalid")
            equal = (receipt["settings_json"], receipt["settings_fingerprint"]) == (encoded, fingerprint)
            return contracts.ResourceAdmission(
                "request_replayed" if equal else "request_conflict", job if equal else _conflict_reference(job), None
            )

    if head["generation"] == 0 and store.has_bound_history(request.scope):
        raise RuntimeError("resource head is missing for retained bound Jobs")
    active = store.active_jobs(request.scope, command)
    target = head["active_job_uuid"]
    if target is not None:
        job = store.resolve_job(request.scope, target)
        _validate_target(job, head, command, request)
        if (
            len(active) != 1
            or active[0]["uuid"] != target
            or job.get("archived")
            or job["status"] not in ("queued", "processing")
        ):
            raise RuntimeError("resource active authority is invalid")
        if (job["resource_settings_json"], job["resource_settings_fingerprint"]) != (encoded, fingerprint):
            return contracts.ResourceAdmission("active_conflict", _conflict_reference(job), None)
        if not store.attach_receipt(request, job, now):
            return contracts.ResourceAdmission(
                "rejected",
                None,
                AdmissionResult.rejected(
                    AdmissionRejectionReason.QUOTA_EXCEEDED,
                    message="Resource request alias quota exceeded",
                ),
            )
        return contracts.ResourceAdmission("active_reused", job, None)
    if active:
        raise RuntimeError("resource has ambiguous or unbound active Jobs")
    return ResourceCreation(
        contracts.ResourceBinding(request.scope, head["authority_uuid"], head["generation"] + 1), encoded, fingerprint
    )
